import os
import copy
import difflib
import numpy as np
import re, unicodedata
from collections import Counter
import ast
import asyncio, json, httpx
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, contains all the essential information from [correct_answer], is equivalent despite minor wording/order differences (such as name order, inclusion or omission of middle names/initials, common honorifics, standard shortenings of first names, inclusion/omission of non-contradictory date parts like year, minor articles like "a"/"the", extra descriptive context, non-essential descriptive prefixes/suffixes such as "Restaurant", "Inc.", "Ltd.", or sports suffixes like "FC", "CF", "SC", inclusion/omission of subtitles in titles, minor spacing/punctuation differences — including presence/absence of quotation marks, interchangeable punctuation such as ":" / "-" / "–", case-only differences, or presence/absence of diacritics), or is within a small margin of error for numerical problems. Answer 'no' only if the extracted answer is incorrect, missing essential identifying information, or contradicts the [correct_answer].

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    answer_match = re.search(r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)", judge_response,
                             re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)", judge_response,
                                 re.IGNORECASE | re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"extracted_final_answer:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    reasoning_match = re.search(r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
                                judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
                                    judge_response, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        reasoning_match = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    confidence_match = re.search(r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    if result["correct"] is None:
        result["parse_error"] = True

    return result


def extract_citations_from_response(response_text: str):
    if not response_text:
        return []

    single_citation_pattern = r'\[(\d+)\]'
    single_matches = re.findall(single_citation_pattern, response_text)

    multi_citation_pattern = r'\[([^\[\]]*?)\]'
    multi_matches = re.findall(multi_citation_pattern, response_text)

    single_fullwidth_pattern = r'【(\d+)】'
    single_fullwidth_matches = re.findall(single_fullwidth_pattern, response_text)

    multi_fullwidth_pattern = r'【([^【】]*?)】'
    multi_fullwidth_matches = re.findall(multi_fullwidth_pattern, response_text)

    all_docids = set()

    all_docids.update(single_matches)
    all_docids.update(single_fullwidth_matches)

    for match in multi_matches:
        if match in single_matches:
            continue
        docids = re.findall(r'\d+', match)
        all_docids.update(docids)

    for match in multi_fullwidth_matches:
        if match in single_fullwidth_matches:
            continue
        docids = re.findall(r'\d+', match)
        all_docids.update(docids)

    return list(all_docids)


def em_score(label: str, pred: str) -> bool:
    ign = {'a', 'an', 'the', 'of', 'on', 'in', 'and', '&', 'for', 'to', 'by', 'with'}
    deacc = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    def norm(s: str) -> str:
        s = deacc(s).lower()
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)
        s = re.sub(r'[“”"\'`]+', '', s)
        s = re.sub(r'[:–—\-_/.,;!()?]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    strip = lambda s: re.sub(r'\s+', '', norm(s))
    toks = lambda s: [t for t in norm(s).split() if t not in ign and not re.fullmatch(r'\d{4}', t)]
    if strip(label) == strip(pred): return True
    lt, pt = toks(label), toks(pred)
    if not lt or not pt: return False
    if Counter(lt) == Counter(pt): return True
    if len(lt) >= 2 and len(pt) >= 2 and lt[-1] == pt[-1]:
        f1, f2 = lt[0], pt[0]
        if f1 == f2 or (min(len(f1), len(f2)) >= 4 and (f1.startswith(f2) or f2.startswith(f1))): return True
    head = lambda s: strip(re.split(r'[:–—-]', norm(s), 1)[0])
    if head(label) == head(pred): return True
    return False

def relaxed_em(label: str, pred: str) -> bool:
    deacc = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    norm  = lambda s: re.sub(r'\s+',' ',re.sub(r'\s*\([^)]*\)\s*',' ',re.sub(r'[“”"\'`]+','',re.sub(r'[:–—\-_/.,;!()?]+',' ',deacc(s).lower())))).strip()
    strip = lambda s: re.sub(r'\s+','',norm(s))
    if not label or not pred: return False
    A,B = strip(label), strip(pred)
    if A==B or A in B or B in A: return True
    if difflib.SequenceMatcher(None,A,B).ratio()>=0.9: return True
    ca,cb=Counter(A),Counter(B);
    if sum((ca&cb).values())/min(len(A),len(B) or 1)>=0.9: return True
    return False


async def call_judge_model(messages, model=None, max_retries=3):
    from openai import AsyncOpenAI
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]
    
    model = model or os.getenv("MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507")
    
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=os.getenv("OPTA_LLM_BASE_URL")
    )
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[LLM] Error after {max_retries} attempts: {e}")
                return f"Error: {e}"
            await asyncio.sleep(1 * (attempt + 1))
    return ""

async def judge(question, correct_answer, predicted_answer):
    if em_score(correct_answer, predicted_answer):
        score = 1
    elif len(predicted_answer.strip()) == 0:
        score = 0
    else:
        judge_prompt = GRADER_TEMPLATE.format(
            question=question,
            response=predicted_answer,
            correct_answer=correct_answer
        )
        messages = [{'role': 'user', 'content': judge_prompt}]
        score = 0
        for _ in range(3):
            response = await call_judge_model(messages)
            grade_report = parse_judge_response(response)
            if grade_report['parse_error']:
                continue
            score = int(grade_report['correct'])
            break
        if score == 0 and relaxed_em(correct_answer, predicted_answer):
            response = await call_judge_model(messages)
            grade_report = parse_judge_response(response)
            score = int(grade_report.get('correct', 0))

    print(f"[Judged] score={score}\nLabel: " + correct_answer + '\nModel: ' + predicted_answer.split('\n')[0])
    return score


def keep_first_n_words(text: str, n: int = 1000) -> str:
    if not text:
        return ""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == n:
            return text[:m.end()] + '\n[Document is truncated.]'
    return text


class AsyncSearchClient:
    def __init__(self, base_url: str, timeout: float = 600.0, retries: int = 3, backoff: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def close(self):
        await self._client.aclose()

    async def _post(self, path: str, payload: dict):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = await self._client.post(path, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("results", data)  # convenience: unwrap "results" if present
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == self.retries:
                    raise
                await asyncio.sleep(self.backoff * attempt)
        raise last_exc  # should not reach

    async def search(self, query: str, k: int = 10):
        return await self._post("/search", {"query": query, "k": k})

    async def open(self, url: str | None = None, docid: str | None = None):
        return await self._post("/open", {"url": url, "docid": docid})


def extract_json_tool(text: str):
    """Return [{"name": ..., "arguments": {...}}, ...] from <tool_call> and <answer> blocks; ignore others."""
    calls = []
    def parse_obj(s):
        for p in (json.loads, ast.literal_eval):
            try: return p(s)
            except Exception: pass
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            frag = m.group(0)
            for p in (json.loads, ast.literal_eval):
                try: return p(frag)
                except Exception: pass
        return None
    for kind, body in re.findall(r"<(tool_call|answer)>\s*(.*?)\s*</\1>", text, flags=re.S):
        body = body.strip()
        if kind == "tool_call":
            if body.startswith("```") and body.endswith("```"):
                body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S).strip()
            obj = parse_obj(body)
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                calls.append({"function": obj["name"], "arguments": args if isinstance(args, dict) else {}})
        elif kind == "answer":
            calls.append({"function": "finish", "arguments": {"answer": body}})
    aligned_calls = []
    for fn in calls:
        if fn['function'] == "search":
            topk = max(10 // (len(fn['arguments'].get('query', [])) + 1), 2)
            for q in fn['arguments'].get('query', []):
                aligned_calls.append({"function": "search", "arguments": {"query": q, "topk": topk}})
        elif fn['function'] == "visit":
            for url in fn['arguments'].get('url', []):
                aligned_calls.append({"function": "open_page", "arguments": {"url": url}})
        else:
            aligned_calls.append(fn)
    return aligned_calls

def extract_fn_call(text, all_functions_in_step=False):
    """
    Parse <function=name>...</function> blocks (and optional <tool_call> JSON).

    By default, only the last "cluster" of calls is returned: consecutive blocks separated by
    fewer than 4 newlines form one cluster; a larger gap starts a new cluster (legacy behavior).

    If all_functions_in_step is True, every matched <function=...> in the response is returned in
    source order, so analysis text / large gaps between tools do not drop earlier calls.
    """
    if not text:
        return None
    if '<tool_call>' in text or '<answer>' in text:
        json_tool = extract_json_tool(text)
        print(json_tool)
        if len(json_tool) > 0:
            return json_tool
        else:
            print(text)
    text = re.split(r'<\[[^\]]+\]>', text)[-1].strip()
    matches = list(re.finditer(r'(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>',
                               text, re.DOTALL))
    if not matches:
        return None
    if all_functions_in_step:
        selected = matches
    else:
        groups = [[matches[0]]]
        for m in matches[1:]:
            prev = groups[-1][-1]
            line_gap = text.count('\n', prev.end(), m.start())
            groups[-1].append(m) if line_gap < 4 else groups.append([m])
        selected = groups[-1]
    return [
        {
            'function': m.group(1),
            'arguments': dict(re.findall(r'<parameter[=>]?([^>]+)>(.*?)</parameter>',
                                         m.group(2), re.DOTALL))
        }
        for m in selected
    ]


class LocalSearch:
    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.search_count = 0
        self.env_fail = False

        base_url = os.getenv("LOCAL_SEARCH_URL")

        self.client = AsyncSearchClient(base_url=base_url)
        self.question = None
        self.label_answer = None
        self.predicted_answer = None
        self.donotgiveup = False
        self.visited_pages = set()
        self.is_finish = False
        self._current_agent_name = 'main'

    async def close(self):
        if hasattr(self, 'client'):
            await self.client.close()

    async def init_env(self, item):
        self.question = item.non_tensor_batch['extra_info'][0]['query']
        self.label_answer = item.non_tensor_batch['extra_info'][0]['answer']
        self.predicted_answer = None

    async def get_data(self, item, context):
        if 'prompt' in item.non_tensor_batch['extra_info'][0]:
            prompt = item.non_tensor_batch['extra_info'][0]['prompt']
            conversations = [
                {'role': 'system', 'content': prompt[0]['content']},
                {'role': 'user', 'content': prompt[-1]['content']},
            ]

        else:
            conversations = [
                {'role': 'system', 'content': ''},
                {'role': 'user', 'content': ''},
            ]

        self.instance_info = copy.deepcopy(item.non_tensor_batch['extra_info'][0])
        self.instance_info['problem_statement'] = self.instance_info['query']
        meta_info = copy.copy(item.meta_info)
        meta_info['uid'] = item.non_tensor_batch['uid'][0]
        meta_info['reward_model'] = item.non_tensor_batch['reward_model'][0]

        max_turn = item.meta_info.get("max_turn")
        if max_turn is None:
            max_turn = self.config.plugin.max_turn
        return conversations, {'max_turn': max_turn, 'meta_info': meta_info}

    async def run_action(self, response):
        fn_call = extract_fn_call(response)
        if fn_call is None or len(fn_call) == 0:
            return {'observation': 'No function call was detected in the model response.'}
        else:
            observation = ''
            for fn in fn_call:
                name = fn['function']
                if name == 'search':
                    self.search_count += 1
                    query = fn['arguments'].get('query', '')
                    topk = (lambda v: int(v) if str(v).isdigit() else 10)(fn['arguments'].get('topk', 10))
                    if not query:
                        observation += '[Error] The "search" function requires a "query" argument.'
                    else:
                        observation += f'[Search Results for "{query}"]\n'
                        serp = await self.client.search(query, 50)
                        show_topk = 0
                        for i, page in enumerate(serp, 1):
                            if page['docid'] in self.visited_pages:
                                page['text'] = "(This page was already seen in a previous search. Here, a shorter snippet is shown. If you find this page relevant, please use the open_page tool to inspect the full content) " + " ".join(page['text'].split()[:128])
                                show_topk += 0.25
                            else:
                                self.visited_pages.add(page['docid'])
                                page['text'] = " ".join(page['text'].split()[:512])
                                show_topk += 1
                            observation += (
                                f"\n--- #{i}: {page['docid']}---\n"
                                f"docid: {page['docid']}\n"
                                f"url: {page['url']}\n"
                                f"content: {page['text']}\n"
                            )
                            if show_topk >= topk:
                                break
                        observation += "\n"
                elif name == 'open_page':
                    url = fn['arguments'].get('url', None)
                    docid = fn['arguments'].get('docid', None)
                    if not docid and not url:
                        observation += '[Error] The "open_page" function requires either a "docid" or a "url".'
                    else:
                        open_pages = await self.client.open(url, docid)
                        for page in open_pages:
                            page['text'] = keep_first_n_words(page['text'], 4096)
                            observation += (
                                f"[Opened Page Content]\n"
                                f"docid: {page['docid']}\n"
                                f"url: {page['url']}\n"
                                f"content: {page['text']}\n"
                            )
                        observation += "\n"
                elif name == 'finish':
                    self.is_finish = True
                    answer = fn['arguments'].get('answer', "")
                    explanation = fn['arguments'].get('explanation', None)
                    confidence = fn['arguments'].get('confidence', None)
                    if len(answer.strip()) == 0:
                        print(response)
                        observation = ("Fail to parse answer. Please resubmit with the correct tool call format, eg\n"
                                       "<function=finish>\n" "<parameter=answer>YOUR ANSWER</parameter>\n"
                                       "<parameter=explanation>YOUR EXPLANATION</parameter>\n"
                                       "<parameter=confidence>YOUR CONFIDENCE</parameter>\n" "</function>\n")
                        return {'observation': observation.strip()}
                    if self.search_count == 0:
                        answer = ""
                    self.predicted_answer = (answer, explanation, confidence)

                    if 'insufficient' in answer.lower() and self.donotgiveup:
                        observation = "The answer is guaranteed to be found through sufficient search and reading. Do not give up; try searching deeper or using alternative approaches."
                        self.donotgiveup = False
                        return {'observation': observation.strip()}

                    if '<q1>' in self.label_answer:
                        label_answer_dict = extract_q_dict(self.label_answer)
                        predicted_answer_dict = extract_q_dict(self.predicted_answer[0])
                        missing = []
                        for k in label_answer_dict:
                            if k not in predicted_answer_dict:
                                missing.append(k)

                        if len(missing) > 0:
                            observation = f"Answer submission failed. The answer is missing the following questions: {', '.join(missing)}. Make sure submit answer for all the questions. Ensure all the answers are submitted in one finish tool call."
                            return {'observation': observation.strip()}

                    return {'action': 'finish'}

                else:
                    observation = f'[Error] The function "{name}" is not supported.'
            observation += "\n\n* Please reflect on the information we have obtained, and keep searching for additional information if we still can not answer the question. Do not give the answer if the information is still not enough."

        return {'observation': observation.strip()}

    async def get_reward(self, item, messages, context):
        if self.env_fail:
            return "", 0, {}
        if self.predicted_answer is None:
            return "", 0, {}

        if '<q1>' in self.label_answer:
            label_answer_dict = extract_q_dict(self.label_answer)
            predicted_answer_dict = extract_q_dict(self.predicted_answer[0])
            all_reward = []
            for k in label_answer_dict:
                if k in predicted_answer_dict:
                    reward = await judge(self.question, label_answer_dict[k], predicted_answer_dict[k])
                    all_reward.append(reward)
                else:
                    all_reward.append(0)
            reward = sum(all_reward) / len(all_reward)
            return "", reward, {}
            
        reward = await judge(self.question, self.label_answer, self.predicted_answer[0])
        return "", reward, {}

    async def update_dataproto(self, out, item, score):
        final_score = score[1]
        out.meta_info["generation_kwargs"] = item.meta_info['generation_kwargs']
        out.non_tensor_batch = copy.deepcopy(item.non_tensor_batch)
        extra_data = {"score": final_score}
        out.non_tensor_batch['extra_data'] = np.array([extra_data, ], dtype=object)
        return out

def extract_q_dict(s: str) -> dict[str, str]:
    return {k: v.strip() for k, v in re.findall(r'<(q\d+)>(.*?)</\1>', s, flags=re.S)}
