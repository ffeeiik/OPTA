#!/usr/bin/env python3
"""
python search_server.py --host 0.0.0.0 --port 8000
"""

import os
import time
import pickle
import re
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty, Full
import signal
import sys
import threading

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.disabled = True

MODEL_NAME = 'Qwen/Qwen3-Embedding-8B'
CORPUS_DATASET = "Tevatron/browsecomp-plus-corpus"
CORPUS_EMBEDDING_DATASET = "your-hf-namespace/browsecomp-plus"
CORPUS_EMBEDDING_FILE = "corpus_embeddings.pkl"
DEFAULT_CORPUS_DATA_PATH = "/your/path/data/corpus"
DEFAULT_CORPUS_EMBEDDINGS_PATH = os.path.join(DEFAULT_CORPUS_DATA_PATH, CORPUS_EMBEDDING_FILE)

class EmbeddingModel(ABC):
    @abstractmethod
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        pass

    @abstractmethod
    def get_corpus_embeddings(self, corpus_texts: List[str], docids: List[str]):
        pass

class LocalEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if device.type == "cuda":
            # RTX 6000 (Turing) does not support bfloat16, use float16
            try:
                major, _ = torch.cuda.get_device_capability(device)
                dtype = torch.bfloat16 if major >= 8 else torch.float16
            except:
                dtype = torch.float16
            
            print(f"Loading local embedding model with {dtype}...")
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype,
                                                  attn_implementation="sdpa").to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        self.model.eval()
        self.device = device
        
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        batch_dict = self.tokenizer(queries, padding=True, truncation=True, max_length=8192, return_tensors="pt")
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            query_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        return query_embeddings

    def get_corpus_embeddings(self, corpus_texts: List[str], docids: List[str]):
         pass

@dataclass
class SearchRequest:
    query: str
    k: int = 20
    request_id: str = None


@dataclass
class SearchBatch:
    requests: List[SearchRequest]


class QueryRequest(BaseModel):
    query: str
    k: int = 20

class OpenRequest(BaseModel):
    docid: Optional[str] = None
    url: Optional[str] = None



class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    took_ms: float


def last_token_pool(last_hidden_states, attention_mask):
    """Pool embeddings using last token"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def keep_first_n_words(text: str, n: int = 1000) -> str:
    if not text:
        return ""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == n:
            return text[:m.end()] + '\n[Document is truncated.]'
    return text


def load_corpus():
    """Load the corpus dataset from local path or HuggingFace"""
    # Priority: local arrow dir > HuggingFace download
    local_data_path = os.getenv("CORPUS_DATA_PATH", DEFAULT_CORPUS_DATA_PATH)
    if local_data_path and os.path.isdir(local_data_path):
        import glob
        arrow_files = sorted(glob.glob(os.path.join(local_data_path, "*.arrow")))
        if arrow_files:
            print(f"Loading corpus dataset from local arrow files: {local_data_path} ({len(arrow_files)} shards)")
            ds = load_dataset("arrow", data_files=arrow_files, split="train")
        else:
            print(f"No arrow files found in {local_data_path}, falling back to HuggingFace...")
            ds = load_dataset(CORPUS_DATASET, split='train')
    else:
        print(f"Loading corpus dataset from {CORPUS_DATASET}...")
        ds = load_dataset(CORPUS_DATASET, split='train')

    docid_to_text = {row["docid"]: {
        'raw': keep_first_n_words(row["text"], 15000),
        'content': keep_first_n_words(row["text"], 1000),
        'url': row['url'],
        'docid': row['docid']
    } for row in ds}
    url_to_docid = {row["url"]: row['docid'] for row in ds}
    print(f"Loaded {len(docid_to_text)} documents")
    return docid_to_text, url_to_docid


def encode_corpus():
    """Load corpus embeddings from local path or HuggingFace dataset"""
    # Priority: local pickle > hf_hub_download
    local_path = os.getenv("CORPUS_EMBEDDINGS_PATH", DEFAULT_CORPUS_EMBEDDINGS_PATH)
    if local_path and os.path.exists(local_path):
        print(f"Loading corpus embeddings from local path: {local_path}")
        with open(local_path, 'rb') as f:
            return pickle.load(f)

    from huggingface_hub import hf_hub_download
    print(f"Downloading corpus embeddings from {CORPUS_EMBEDDING_DATASET}...")
    embeddings_path = hf_hub_download(
        repo_id=CORPUS_EMBEDDING_DATASET,
        filename=CORPUS_EMBEDDING_FILE,
        repo_type="dataset"
    )
    print(f"Loading corpus embeddings from {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        return pickle.load(f)


def high_speed_batcher(request_queue: mp.Queue, batch_queue: mp.Queue,
                       max_batch_size: int = 512, batch_timeout: float = 0.005):
    """Ultra-fast batcher optimized for high throughput"""

    batch_requests = []
    last_batch_time = time.time()

    while True:
        try:
            # Very short timeout for maximum responsiveness
            request = request_queue.get(timeout=batch_timeout)

            if request is None:  # Shutdown
                if batch_requests:
                    batch_queue.put(SearchBatch(requests=batch_requests))
                break

            batch_requests.append(request)
            current_time = time.time()

            # Send batch if full or timeout exceeded
            if (len(batch_requests) >= max_batch_size or (current_time - last_batch_time) >= batch_timeout):
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Batcher] Sending batch size={len(batch_requests)}")
                batch_queue.put_nowait(SearchBatch(requests=batch_requests))
                batch_requests = []
                last_batch_time = current_time

        except Empty:
            # Process any pending requests immediately for low latency
            if batch_requests:
                current_time = time.time()
                if (current_time - last_batch_time) >= batch_timeout:
                    batch_queue.put_nowait(SearchBatch(requests=batch_requests))
                    batch_requests = []
                    last_batch_time = current_time
            continue

        except Full:
            # If batch queue is full, just reset and continue
            batch_requests = []
            last_batch_time = time.time()
            continue

        except Exception:
            # On any error, reset batch to avoid getting stuck
            batch_requests = []
            last_batch_time = time.time()


def optimized_worker(worker_id: int, batch_queue: mp.Queue, result_queue: mp.Queue,
                     corpus_data: Dict, device_str: str = "cuda"):
    """Optimized worker for maximum throughput"""
    try:
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        embedding_model = LocalEmbeddingModel(MODEL_NAME, device)
        
        corpus_embeddings = corpus_data['embeddings']
        corpus_embeddings = corpus_embeddings.to(device)
        if hasattr(embedding_model, 'model'):
            target_dtype = embedding_model.model.dtype
            if corpus_embeddings.dtype != target_dtype:
                print(f"Worker {worker_id}: Converting corpus from {corpus_embeddings.dtype} to {target_dtype}")
                corpus_embeddings = corpus_embeddings.to(target_dtype)
        
        corpus_docids = corpus_data['docids']
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'

        print(f"Worker {worker_id}: Ready on {device}")

        while True:
            try:
                batch = batch_queue.get(timeout=0.01)

                if batch is None:
                    break
                
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Worker {worker_id}] Processing batch size={len(batch.requests)}")
                fast_process_batch(batch, embedding_model,
                                   corpus_embeddings, corpus_docids, task_description, result_queue)
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Worker {worker_id}] Batch done")

            except Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

    except Exception as e:
        print(f"Worker {worker_id} init error: {e}")


def fast_process_batch(batch: SearchBatch, embedding_model: EmbeddingModel,
                       corpus_embeddings, corpus_docids, task_description, result_queue):
    """Ultra-fast batch processing"""
    try:
        queries = [get_detailed_instruct(task_description, req.query) for req in batch.requests]

        query_embeddings = embedding_model.encode_queries(queries)
        query_embeddings = query_embeddings.to(corpus_embeddings.device)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        
        similarities = torch.mm(query_embeddings, corpus_embeddings.T)

        for i, request in enumerate(batch.requests):
            scores, indices = torch.topk(similarities[i], k=min(request.k, len(corpus_docids)))

            results = []
            for score, idx in zip(scores.cpu().tolist(), indices.cpu().tolist()):
                results.append({'docid': corpus_docids[idx], 'score': float(score)})

            result_queue.put({
                'request_id': request.request_id,
                'results': results,
                'status': 'success'
            })

    except Exception as e:
        for request in batch.requests:
            result_queue.put({
                'request_id': request.request_id,
                'results': [],
                'status': 'error',
                'error': str(e)
            })
        import traceback
        traceback.print_exc()
        print(f"BATCH ERROR: {e}")


class HighThroughputSearchServer:
    def __init__(self, num_workers: int = 1, max_batch_size: int = 2048, batch_timeout: float = 0.005,
                 device: str = "cuda"):

        # Optimized for 10k+ requests: Large queues, fast timeouts
        self.request_queue = mp.Queue(maxsize=20000)  # Large queue for burst traffic
        self.batch_queue = mp.Queue(maxsize=1000)  # Large batch queue
        self.result_queue = mp.Queue(maxsize=20000)  # Large result queue

        self.pending_requests = {}

        self.docid_to_text, self.url_to_docid = load_corpus()

        self.corpus_data = encode_corpus()

        # Start high-speed batcher
        self.batcher = mp.Process(target=high_speed_batcher,
                                  args=(self.request_queue, self.batch_queue, max_batch_size, batch_timeout))
        self.batcher.start()

        # Start optimized workers
        self.workers = []
        for i in range(num_workers):
            worker = mp.Process(target=optimized_worker,
                                args=(i, self.batch_queue, self.result_queue, self.corpus_data, device))
            worker.start()
            self.workers.append(worker)

        print(f"Started {num_workers} worker(s) on {device} with max_batch_size={max_batch_size}")

        # Fast result collector
        self.result_thread = threading.Thread(target=self._fast_collect_results, daemon=True)
        self.result_thread.start()

    def _fast_collect_results(self):
        """Optimized result collection"""
        while True:
            try:
                result = self.result_queue.get(timeout=0.1)
                request_id = result['request_id']

                if request_id in self.pending_requests:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Collector] Result for {request_id} received")
                    future = self.pending_requests.pop(request_id)
                    future.set_result(result)

            except Empty:
                continue
            except Exception:
                continue

    async def search(self, query: str, k: int = 20) -> Dict:
        """Fast search with minimal overhead"""
        import asyncio
        import uuid

        request_id = str(uuid.uuid4())
        request = SearchRequest(query=query, k=k, request_id=request_id)

        # Create future
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_requests[request_id] = future

        try:
            # Submit request
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Search] Queuing request {request_id} (k={k})")
            self.request_queue.put_nowait(request)
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Search] Request {request_id} queued")

            # Wait for result with generous timeout for high load
            result = await asyncio.wait_for(future, timeout=600.0)
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Search] Got result for {request_id}")

            return result

        except Full:
            self.pending_requests.pop(request_id, None)
            raise HTTPException(status_code=503, detail="Server overloaded")
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise HTTPException(status_code=408, detail="Timeout")

    def shutdown(self):
        """Clean shutdown"""
        print("Shutting down...")

        # Stop batcher
        self.request_queue.put(None)
        self.batcher.join(timeout=2.0)
        if self.batcher.is_alive():
            self.batcher.terminate()

        # Stop workers
        for _ in self.workers:
            self.batch_queue.put(None)

        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()


# Global server
search_server: HighThroughputSearchServer = None

# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup_event():
    global search_server
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 1
    max_batch_size = 2048
    batch_timeout = 0.005

    print(f"Starting search server: device={device}, workers={num_workers}, batch={max_batch_size}, timeout={batch_timeout}s")

    search_server = HighThroughputSearchServer(
        num_workers=num_workers,
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        device=device,
    )


@app.on_event("shutdown")
async def shutdown_event():
    if search_server:
        search_server.shutdown()


@app.post("/search", response_model=QueryResponse)
async def search_endpoint(request: QueryRequest):
    """Optimized search endpoint"""
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Endpoint] SEARCH request received: {request.query[:50]}...")
    result = await search_server.search(request.query, request.k)
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] [Endpoint] SEARCH request finished in {(time.time() - start_time) * 1000:.2f}ms")

    if result['status'] == 'error':
        raise HTTPException(status_code=500, detail=result.get('error', 'Error'))

    # Add document text
    enriched_results = []
    for res in result['results']:
        text = 'Fetch document error.'
        url = 'ERROR'
        if res['docid'] in search_server.docid_to_text:
            text = search_server.docid_to_text.get(res['docid'])['content']
            url = search_server.docid_to_text.get(res['docid'])['url']
        enriched_results.append({
            'docid': res['docid'],
            'url': url,
            'text':text,
            'score': res['score']
        })

    return QueryResponse(results=enriched_results, took_ms=(time.time() - start_time) * 1000)


@app.post("/open", response_model=QueryResponse)
async def open_page(request: OpenRequest):
    """Optimized search endpoint"""
    start_time = time.time()
    docid = request.docid or (search_server.url_to_docid.get(request.url) if request.url else None)
    if not docid:
        return QueryResponse(
            results=[{'docid': docid, 'url': request.url, 'text': "Missing docid and url, or url not indexed."}],
            took_ms=(time.time() - start_time) * 1000
        )

    item = search_server.docid_to_text.get(docid)
    if not item:
        return QueryResponse(
            results=[{'docid': docid, 'url': request.url, 'text': "Document not found for given docid."}],
            took_ms=(time.time() - start_time) * 1000
        )

    enriched_results = [{
        'docid': docid,
        'url': item.get('url', request.url),
        'text': item.get('raw', item.get('text', "")),
    }]
    return QueryResponse(results=enriched_results, took_ms=(time.time() - start_time) * 1000)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "workers": len(search_server.workers) if search_server else 0}


def signal_handler(sig, frame):
    if search_server:
        search_server.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=MODEL_NAME)
    parser.add_argument('--corpus', type=str, default=CORPUS_DATASET)
    parser.add_argument('--corpus-embedding-dataset', type=str, default=CORPUS_EMBEDDING_DATASET)
    parser.add_argument('--corpus-embedding-file', type=str, default=CORPUS_EMBEDDING_FILE)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    MODEL_NAME = args.model
    CORPUS_DATASET = args.corpus
    CORPUS_EMBEDDING_DATASET = args.corpus_embedding_dataset
    CORPUS_EMBEDDING_FILE = args.corpus_embedding_file

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)
