"""OPTA evaluator.

OPTA (Online Partial Trajectory Aggregating) runs several search trajectories
for the same question.  Whenever a trajectory folds its context, it emits a
chunk report.  The controller accumulates each trajectory's report, builds a
shared state across active trajectories, injects useful cross-trajectory
information back into each trajectory, and finally selects one answer.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import traceback
from collections import defaultdict

import httpx
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from opta.agent import process_item
from opta.config import create_config
from opta.data import DataProto
from opta.prompts import SUMMARY_PROMPT_SEARCH
from opta.search_env import judge
from opta.shared_state_prompts import (
    FINAL_INTEGRATE_PROMPT,
    PER_TRAJ_INJECTION_PROMPT,
    PER_TRAJ_INJECTION_SYSTEM_PROMPT,
    SHARED_STATE_BUILD_PROMPT,
    SHARED_STATE_SYSTEM_PROMPT,
)
from opta.utils import CallAPI, TaskContext


AGENT_MODEL_NAME = os.getenv("AGENT_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", AGENT_MODEL_NAME)


CHUNK_MERGE_PROMPT = """**Original Question:** {question}

The following chunk-level reports come from one trajectory.

{trajectory_status}

Merge them into one concise accumulated report. Keep only information relevant
to solving the original question, deduplicate repeated findings, and clearly state
what remains unresolved.

Below are the chunk-level reports:
{chunk_reports}

Present the merged report in Markdown and wrap it within <report> </report> tags."""


MERGE_STATUS_IN_PROGRESS = (
    "These reports cover a partial trajectory. Do not claim the final answer "
    "has been reached unless it is explicitly present."
)
MERGE_STATUS_COMPLETE = (
    "These reports cover the complete trajectory. Extract the trajectory's "
    "final answer if it is present."
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def extract_tag_content(text: str | None, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.S | re.I)
    return match.group(1).strip() if match else ""


def extract_block_keep_tags(text: str | None, tag: str) -> str:
    match = re.search(rf"(<{tag}>\s*.*?\s*</{tag}>)", text or "", re.S | re.I)
    return match.group(1).strip() if match else ""


def parse_selected_trajectory_id(text: str | None, eligible_ids) -> int | None:
    raw = extract_tag_content(text, "trajectory_id")
    if not raw or raw.strip().lower() in {"none", "n/a", "na", "no prediction"}:
        return None
    match = re.search(r"\d+", raw)
    if not match:
        return None
    tid = int(match.group(0))
    return tid if tid in set(eligible_ids) else None


def infer_selected_trajectory_id(answer: str | None, usable: list[dict]) -> int | None:
    if not answer or answer == "[No Prediction]":
        return None
    norm = str(answer).strip().casefold()
    matches = [
        int(item["trajectory_id"])
        for item in usable
        if str(item.get("answer") or "").strip().casefold() == norm
    ]
    return matches[0] if len(matches) == 1 else None


def format_messages_for_chunk(messages: list[dict]) -> str:
    chunks = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content", "") or "").strip()
        if not content:
            continue
        role = msg.get("role", "unknown")
        source = msg.get("source")
        label = f"{role}" + (f"/{source}" if source else "")
        chunks.append(f"[{label}]\n{content}")
    return "\n\n---\n\n".join(chunks).strip() or "[No new content.]"


def normalize_extra_info(row, idx: int):
    if "query" in row and "extra_info" not in row:
        return {
            "query": row["query"],
            "answer": row["answer"],
            "instance_id": row.get("instance_id", row.get("query_id", f"val_{idx}")),
            "workflow": "search",
        }

    extra_info = dict(row["extra_info"])
    extra_info["workflow"] = "search"
    return extra_info


def build_item(row, idx: int):
    item = DataProto()
    extra_info = normalize_extra_info(row, idx)
    item.non_tensor_batch = {
        "ability": np.array(["search"], dtype=object),
        "extra_info": np.array([extra_info], dtype=object),
        "uid": np.array([extra_info.get("instance_id", "unknown")], dtype=object),
        "reward_model": np.array([{}], dtype=object),
    }
    item.meta_info = {"generation_kwargs": {}, "max_turn": None}
    return item, extra_info


def extract_result_payload(output):
    if output is None:
        return {
            "recent_messages": [],
            "score": 0.0,
            "predicted_answer": None,
        }

    extra_data = output.non_tensor_batch.get("extra_data", [{}])[0]
    recent_messages = output.non_tensor_batch.get("recent_messages", [[]])[0]
    if isinstance(recent_messages, np.ndarray):
        recent_messages = recent_messages.tolist()
    if not isinstance(recent_messages, list):
        recent_messages = []
    predicted_answer = output.non_tensor_batch.get("predicted_answer", [None])[0]
    return {
        "recent_messages": recent_messages,
        "score": float(extra_data.get("score", 0.0) or 0.0),
        "predicted_answer": predicted_answer or None,
    }


class OPTAController:
    def __init__(
        self,
        *,
        question: str,
        sample_count: int,
        min_summary_rounds: int,
        merge_max_tokens: int,
        shared_state_max_tokens: int,
        injection_max_tokens: int,
        integration_max_tokens: int,
    ):
        self.question = question
        self.sample_count = sample_count
        self.min_summary_rounds = min_summary_rounds
        self.merge_max_tokens = merge_max_tokens
        self.shared_state_max_tokens = shared_state_max_tokens
        self.injection_max_tokens = injection_max_tokens
        self.integration_max_tokens = integration_max_tokens

        self.alive_ids = set(range(1, sample_count + 1))
        self.finished_ids = set()
        self.latest_accumulated = {}
        self.round_accumulated_snapshot = defaultdict(dict)
        self.round_in_progress = set()
        self.round_decisions = {}

        self.condition = asyncio.Condition()
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=os.getenv("OPTA_LLM_BASE_URL"),
            http_client=httpx.AsyncClient(proxy=None, trust_env=False),
        )

    async def close(self):
        await self.client.close()

    async def call_llm(self, messages, max_tokens, *, system_prompt=None, temperature=0.2):
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        response = await self.client.chat.completions.create(
            model=os.getenv("MODEL_NAME", AGENT_MODEL_NAME),
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content or ""

    async def mark_finished(self, trajectory_id: int):
        async with self.condition:
            self.finished_ids.add(trajectory_id)
            self.condition.notify_all()

    async def merge_chunk_reports(
        self,
        *,
        trajectory_id: int,
        previous_accumulated: str,
        new_chunk_report: str,
        is_complete: bool = False,
    ) -> str:
        blocks = [
            "[Previous Accumulated Report]\n"
            f"{previous_accumulated or '[No previous accumulated report.]'}",
            f"[New Chunk Report]\n{new_chunk_report}",
        ]
        prompt = CHUNK_MERGE_PROMPT.format(
            question=self.question,
            trajectory_status=MERGE_STATUS_COMPLETE if is_complete else MERGE_STATUS_IN_PROGRESS,
            chunk_reports="\n\n".join(blocks),
        )
        try:
            content = await self.call_llm(
                [{"role": "user", "content": prompt}],
                self.merge_max_tokens,
                system_prompt="Follow the user's instructions exactly.",
            )
            return extract_tag_content(content, "report") or content.strip() or new_chunk_report
        except Exception as exc:
            print(f"[OPTA merge error] trajectory={trajectory_id}: {exc}")
            return new_chunk_report or previous_accumulated

    async def build_shared_state(self, round_idx: int, active_ids: list[int], snapshot: dict[int, str]) -> str:
        if not active_ids:
            return ""
        trajectory_reports = "\n\n---\n\n".join(
            f"### [Trajectory {tid}]\n{snapshot.get(tid) or '[No accumulated report]'}"
            for tid in active_ids
        )
        prompt = SHARED_STATE_BUILD_PROMPT.format(
            sample_count=len(active_ids),
            round_idx=round_idx,
            question=self.question,
            trajectory_reports=trajectory_reports,
        )
        try:
            content = await self.call_llm(
                [{"role": "user", "content": prompt}],
                self.shared_state_max_tokens,
                system_prompt=SHARED_STATE_SYSTEM_PROMPT,
            )
            return extract_block_keep_tags(content, "shared_state") or content.strip()
        except Exception as exc:
            print(f"[OPTA shared-state error] round={round_idx}: {exc}")
            return ""

    async def inject_for_trajectory(
        self,
        *,
        trajectory_id: int,
        round_idx: int,
        own_report: str,
        shared_state_xml: str,
    ):
        if not shared_state_xml:
            return own_report

        prompt = PER_TRAJ_INJECTION_PROMPT.format(
            trajectory_id=trajectory_id,
            round_idx=round_idx,
            question=self.question,
            own_report=own_report or "[No accumulated report]",
            shared_state_xml=shared_state_xml,
        )
        try:
            content = await self.call_llm(
                [{"role": "user", "content": prompt}],
                self.injection_max_tokens,
                system_prompt=PER_TRAJ_INJECTION_SYSTEM_PROMPT,
            )
        except Exception as exc:
            print(f"[OPTA injection error] trajectory={trajectory_id}: {exc}")
            return own_report

        enriched = extract_tag_content(content, "enriched_accumulated_report")
        if not enriched:
            return own_report
        return enriched

    async def run_aggregator_round(self, round_idx: int, active_ids: list[int], snapshot: dict[int, str]):
        if round_idx < self.min_summary_rounds or len(active_ids) <= 1:
            return {tid: snapshot[tid] for tid in active_ids}

        shared_state_xml = await self.build_shared_state(round_idx, active_ids, snapshot)

        async def inject(tid):
            enriched = await self.inject_for_trajectory(
                trajectory_id=tid,
                round_idx=round_idx,
                own_report=snapshot.get(tid, ""),
                shared_state_xml=shared_state_xml,
            )
            return tid, enriched

        results = await asyncio.gather(*[inject(tid) for tid in active_ids])
        return {tid: enriched for tid, enriched in results}

    async def on_chunk_report(
        self,
        *,
        trajectory_id: int,
        chunk_report: str,
        previous_accumulated_report: str,
        round_idx: int,
        instance_id=None,
        question=None,
    ):
        merged_report = await self.merge_chunk_reports(
            trajectory_id=trajectory_id,
            previous_accumulated=previous_accumulated_report
            or self.latest_accumulated.get(trajectory_id, ""),
            new_chunk_report=chunk_report,
            is_complete=False,
        )

        async with self.condition:
            self.latest_accumulated[trajectory_id] = merged_report
            self.round_accumulated_snapshot[round_idx][trajectory_id] = merged_report
            self.condition.notify_all()

        async with self.condition:
            while True:
                active_ids = sorted(self.alive_ids - self.finished_ids)
                if round_idx in self.round_decisions:
                    return {
                        "accumulated_report": self.round_decisions[round_idx].get(
                            trajectory_id, merged_report
                        )
                    }
                missing = [
                    tid for tid in active_ids
                    if tid not in self.round_accumulated_snapshot[round_idx]
                ]
                if missing:
                    await self.condition.wait()
                    continue
                if round_idx in self.round_in_progress:
                    await self.condition.wait()
                    continue
                self.round_in_progress.add(round_idx)
                break

        active_ids = sorted(self.alive_ids - self.finished_ids)
        snapshot = {tid: self.round_accumulated_snapshot[round_idx][tid] for tid in active_ids}
        decisions = await self.run_aggregator_round(
            round_idx, active_ids, snapshot
        )
        async with self.condition:
            self.round_decisions[round_idx] = decisions
            self.round_in_progress.discard(round_idx)
            self.condition.notify_all()
            return {"accumulated_report": decisions.get(trajectory_id, merged_report)}

    async def finalize_accumulated_reports(self, trajectory_payloads: dict[int, dict]):
        for tid, payload in trajectory_payloads.items():
            previous = self.latest_accumulated.get(tid, "")
            recent_messages = payload.get("recent_messages", [])
            candidate_answer = payload.get("predicted_answer") or "[No Prediction]"
            recent_text = format_messages_for_chunk(recent_messages)

            current_context = (
                f"{recent_text}\n\n"
                f"[Trajectory final candidate answer]: {candidate_answer}"
            ).strip()
            prompt = (
                "Below is the CURRENT context window to summarize. Use only this content.\n\n"
                f"{current_context}\n\n"
                + SUMMARY_PROMPT_SEARCH.format(question=self.question)
            )
            try:
                content = await self.call_llm(
                    [{"role": "user", "content": prompt}],
                    self.merge_max_tokens,
                    system_prompt="Follow the user's instructions exactly.",
                )
            except Exception as exc:
                print(f"[OPTA final chunk error] trajectory={tid}: {exc}")
                content = ""
            final_chunk_report = extract_tag_content(content, "report") or content.strip() or "[Empty]"
            merged = await self.merge_chunk_reports(
                trajectory_id=tid,
                previous_accumulated=previous,
                new_chunk_report=final_chunk_report,
                is_complete=True,
            )
            self.latest_accumulated[tid] = merged

    async def select_final_answer(self, trajectory_payloads: dict[int, dict], eligible_ids: list[int]):
        usable = []
        for tid in sorted(set(eligible_ids)):
            payload = trajectory_payloads.get(tid, {}) or {}
            usable.append(
                {
                    "trajectory_id": tid,
                    "summary": self.latest_accumulated.get(tid, ""),
                    "answer": payload.get("predicted_answer"),
                }
            )
        usable = [item for item in usable if item["summary"] or item["answer"]]
        if not usable:
            return {
                "num_considered": 0,
                "selected_trajectory_id": None,
                "answer": "[No Prediction]",
                "raw_response": "",
            }
        if len(usable) == 1 and usable[0]["answer"]:
            return {
                "num_considered": 1,
                "selected_trajectory_id": int(usable[0]["trajectory_id"]),
                "answer": usable[0]["answer"],
                "raw_response": "",
                "selection_method": "single_usable_trajectory",
            }

        trajectory_reports = "\n\n---\n\n".join(
            f"### [Trajectory {item['trajectory_id']}]\n"
            f"Candidate Answer: {item['answer'] or '[No Prediction]'}\n\n"
            f"Final Accumulated Report:\n{item['summary'] or '[No accumulated report]'}"
            for item in usable
        )
        try:
            content = await self.call_llm(
                [
                    {
                        "role": "user",
                        "content": FINAL_INTEGRATE_PROMPT.format(
                            question=self.question,
                            trajectory_reports=trajectory_reports,
                        ),
                    }
                ],
                self.integration_max_tokens,
                system_prompt="Follow the user's instructions exactly.",
            )
        except Exception as exc:
            print(f"[OPTA final integration error] {exc}")
            return {
                "num_considered": len(usable),
                "selected_trajectory_id": None,
                "answer": "[No Prediction]",
                "raw_response": str(exc),
            }

        answer = extract_tag_content(content, "answer") or "[No Prediction]"
        selected_tid = parse_selected_trajectory_id(content, eligible_ids)
        method = "explicit_trajectory_id" if selected_tid is not None else ""
        if selected_tid is None:
            selected_tid = infer_selected_trajectory_id(answer, usable)
            if selected_tid is not None:
                method = "inferred_from_answer_match"
        return {
            "num_considered": len(usable),
            "selected_trajectory_id": selected_tid,
            "answer": answer,
            "raw_response": content,
            "selection_method": method,
        }


async def run_opta_for_sample(
    *,
    idx: int,
    row,
    tokenizer,
    config,
):
    base_item, extra_info = build_item(row, idx)
    question = extra_info.get("query", "")
    answer = extra_info.get("answer", "")
    instance_id = extra_info.get("instance_id", f"val_{idx}")

    sample_count = env_int("OPTA_TRAJECTORIES", 10)
    min_summary_rounds = env_int("OPTA_MIN_SUMMARY_ROUNDS", 3)
    sample_temperature = env_float("OPTA_SAMPLE_TEMPERATURE", 0.2)
    sample_top_p = env_float("OPTA_SAMPLE_TOP_P", 1.0)

    controller = OPTAController(
        question=question,
        sample_count=sample_count,
        min_summary_rounds=min_summary_rounds,
        merge_max_tokens=env_int("OPTA_MERGE_MAX_TOKENS", 4096),
        shared_state_max_tokens=env_int("OPTA_SHARED_STATE_MAX_TOKENS", 4096),
        injection_max_tokens=env_int("OPTA_INJECTION_MAX_TOKENS", 4096),
        integration_max_tokens=env_int("OPTA_FINAL_INTEGRATION_MAX_TOKENS", 1024),
    )
    trajectory_payloads = {}

    async def hook_adapter(**kwargs):
        return await controller.on_chunk_report(**kwargs)

    async def run_one_trajectory(traj_id: int):
        item = copy.deepcopy(base_item)
        item.meta_info = {
            "generation_kwargs": {},
            "max_turn": config.actor_rollout_ref.rollout.plugin.max_turn,
        }
        ctx = TaskContext(
            config=config,
            global_step=0,
            server_host=AGENT_MODEL_NAME,
            server_port=0,
            is_train=False,
            tokenizer=tokenizer,
        )
        try:
            output = await process_item(
                item,
                ctx,
                CallAPI,
                trajectory_id=traj_id,
                summary_hook=hook_adapter,
                trajectory_sampling_overrides={
                    "temperature": sample_temperature,
                    "top_p": sample_top_p,
                },
            )
            payload = extract_result_payload(output)
            payload["trajectory_id"] = traj_id
            trajectory_payloads[traj_id] = payload
            return payload
        finally:
            await controller.mark_finished(traj_id)

    try:
        results = await asyncio.gather(
            *[run_one_trajectory(tid) for tid in range(1, sample_count + 1)]
        )
        all_ids = list(range(1, sample_count + 1))
        await controller.finalize_accumulated_reports(trajectory_payloads)
        final_selection = await controller.select_final_answer(
            trajectory_payloads,
            eligible_ids=all_ids,
        )
        final_answer = final_selection["answer"]
        selected_tid = final_selection.get("selected_trajectory_id")
        try:
            selected_tid = int(selected_tid) if selected_tid is not None else None
        except (TypeError, ValueError):
            selected_tid = None
        if selected_tid not in all_ids:
            selected_tid = None
            final_selection["selected_trajectory_id"] = None

        final_score = await judge(question, answer, final_answer if final_answer != "[No Prediction]" else "")
        return {
            "instance_id": instance_id,
            "query": question,
            "answer": answer,
            "selected_answer": final_answer,
            "score": float(final_score),
            "selected_trajectory_id": selected_tid,
            "num_trajectories": sample_count,
        }
    finally:
        await controller.close()


async def run_batch(samples, tokenizer, output_file, config):
    max_workers = env_int("OPTA_WORKERS", 8)
    total_samples = len(samples)
    print(f"Starting OPTA evaluation on {total_samples} samples with {max_workers} workers.")
    print(f"Results will be appended to: {output_file}")

    semaphore = asyncio.Semaphore(max_workers)
    log_lock = asyncio.Lock()
    progress = {"done": 0, "scores": []}
    progress_lock = asyncio.Lock()

    async def evaluate_single(idx, row):
        async with semaphore:
            try:
                log_data = await run_opta_for_sample(
                    idx=idx,
                    row=row,
                    tokenizer=tokenizer,
                    config=config,
                )
            except Exception as exc:
                traceback.print_exc()
                extra_info = normalize_extra_info(row, idx)
                log_data = {
                    "instance_id": extra_info.get("instance_id", f"val_{idx}"),
                    "query": extra_info.get("query", ""),
                    "answer": extra_info.get("answer", ""),
                    "selected_answer": "[No Prediction]",
                    "score": 0.0,
                    "selected_trajectory_id": None,
                    "num_trajectories": env_int("OPTA_TRAJECTORIES", 10),
                    "error": str(exc),
                }

            async with log_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data, ensure_ascii=False, cls=NumpyEncoder) + "\n")

            async with progress_lock:
                progress["done"] += 1
                progress["scores"].append(float(log_data.get("score", 0.0)))
                done = progress["done"]
                if done % 5 == 0 or done == total_samples:
                    avg = float(np.mean(progress["scores"])) if progress["scores"] else 0.0
                    print(f"Progress: {done}/{total_samples} | avg_score={avg:.4f}")

    await asyncio.gather(*[evaluate_single(idx, row) for idx, row in samples.iterrows()])


def load_results(jsonl_path: str) -> list[dict]:
    rows = []
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_summary(output_file: str):
    results = load_results(output_file)
    if not results:
        return
    scores = [float(item.get("score", 0.0) or 0.0) for item in results]
    summary = {
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "success_rate": float(np.mean([1 if score == 1.0 else 0 for score in scores])) if scores else 0.0,
        "total_samples": len(results),
        "results": results,
    }
    with open(output_file.replace(".jsonl", ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(description="Run OPTA on a search dataset.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results_opta.jsonl")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(args.data_path)

    df = pd.read_parquet(args.data_path)
    if args.num_samples:
        df = df.head(args.num_samples)

    print(f"Loaded {len(df)} samples.")
    print(f"Loading tokenizer from {LOCAL_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    config = create_config()

    asyncio.run(run_batch(df, tokenizer, output_file, config))
    write_summary(output_file)
    print(f"Done. Output: {output_file}")


if __name__ == "__main__":
    main()
