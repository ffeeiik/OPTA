import os
import time
import copy
import uuid
from itertools import groupby
import re, unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer
import aiohttp
import asyncio, httpx, inspect
from opta.data import DataProto
from opta.search_env import LocalSearch


def select_env():
    return LocalSearch


def decode_conversation(input_ids: list[int], tokenizer) -> tuple[list[dict[str, str]], str]:
    decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)
    pattern = re.compile(
        re.escape(tokenizer.bos_token)
        + r'(system|user|assistant|tool)\n'
        + r'(.*?)'
        + r'(?=' + re.escape(tokenizer.eos_token) + r')',
        re.DOTALL,
    )
    matches = pattern.findall(decoded_str)
    conversation = [{'role': role, 'content': content} for role, content in matches]
    return conversation, decoded_str

def truncate_text(
        text: str,
        max_lines: int | None = None,
        max_length: int | None = None,
        merge_repeat: bool = False,
        merge_num: int = 128,
        keep_tail_lines: int = 5,
) -> str:
    lines = text.splitlines()

    if merge_repeat:
        merged: list[str] = []
        for line, group in groupby(lines):
            grp = list(group)
            cnt = len(grp)
            if cnt > merge_num:
                merged += [line] * 2
                merged.append(f"[This line repeated {cnt - 4} more times]")
                merged += [line] * 2
            else:
                merged += grp
        lines = merged

    if max_lines is not None and len(lines) > max_lines:
        total = len(lines)
        if max_lines <= keep_tail_lines + 1:
            lines = lines[:max_lines]
        else:
            head_count = max_lines - keep_tail_lines - 1
            head = lines[:head_count]
            tail = lines[-keep_tail_lines:]
            omitted = total - head_count - keep_tail_lines
            lines = head + [f"… {omitted} lines omitted …"] + tail

    if max_length is not None:
        truncated_lines: list[str] = []
        for line in lines:
            if len(line) > max_length:
                truncated_lines.append(line[:max_length] + "… (truncated)")
            else:
                truncated_lines.append(line)
        lines = truncated_lines
    return "\n".join(lines)


def is_weird(text, repeat_n=128, cjk_limit=128):
    s = unicodedata.normalize('NFKC', text)
    if re.search(rf'(.)\1{{{repeat_n - 1},}}|(.{{2,12}})\2{{{repeat_n - 1},}}', s):
        return True
    CJK = ((0x4E00, 0x9FFF), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF))
    cjk_count = sum(any(a <= ord(c) <= b for a, b in CJK) for c in s)
    return cjk_count >= cjk_limit or (len(s) > 0 and cjk_count / len(s) > 0.8)


class CallLLM:
    def __init__(self, host, port, tokenizer, config, meta_info):
        if ':' in host:
            host = f'[{host}]'
        url = f"http://{host}:{port}/chat/completions"

        self.url = url
        self.tokenizer = tokenizer
        self.config = config
        self.meta_info = meta_info

    async def _create_completion(self, input_ids, **kwargs):
        generation_kwargs = self.meta_info['generation_kwargs']
        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_len = min(max_len, self.config.prompt_length + self.config.response_length)
        max_tokens = max_len - len(input_ids)
        if getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_tokens = min(max_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_tokens = min(max_tokens, kwargs['max_new_tokens'])

        if max_tokens < 10:
            print(f"[DEBUG] max_tokens {max_tokens}, skip rollout")
            return None

        uid = kwargs.pop('uid', self.meta_info.get('uid', None))

        request_data = {
            "model": "rollout",
            "messages": {'prompt': input_ids},
            "top_p": generation_kwargs['top_p'],
            "top_k": generation_kwargs['top_k'],
            "temperature": generation_kwargs['temperature'],
            "max_tokens": max_tokens,
            "max_length": max_len,
            "meta_info": self.meta_info | {'uid': uid},
        }

        import asyncio

        for attempt in range(10):
            try:
                timeout = aiohttp.ClientTimeout(total=9600)
                session = aiohttp.ClientSession(timeout=timeout)
                async with session.post(url=self.url,
                                        json=request_data,
                                        timeout=timeout) as response:
                    completion = await response.json()
                    completion['choices'][0]['message']['extra_data']['input_ids'] = input_ids
                    assert response.status == 200, f"completion request failed: {completion}"
                    await session.close()
                    return completion

            except Exception as e:
                print(f"[CallLLM ERROR] {e}")
                await session.close()
                await asyncio.sleep(2 ** attempt)
                if attempt < 2:
                    pass
                else:
                    import traceback
                    traceback.print_exc()
        return completion

    async def create_completion(self, input_ids, **kwargs):
        return await self._create_completion(input_ids, **kwargs)

class CallAPI:
    def __init__(self, host, port, tokenizer, config, meta_info):
        self.tokenizer = tokenizer
        self.config = config
        self.meta_info = meta_info
        self.model = host
        from openai import AsyncOpenAI
        import os
        _http_client = httpx.AsyncClient(proxy=None)
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=os.getenv("OPTA_LLM_BASE_URL", None),
            http_client=_http_client,
        )

    async def close(self):
        if hasattr(self, 'client'):
            await self.client.close()

    async def create_completion(self, input_ids, **kwargs):
        model_budget = int(os.environ.get(
            'MAX_MODEL_LEN',
            self.config.prompt_length + self.config.response_length
        ))
        max_len = kwargs.pop('max_len', None) or model_budget
        effective_budget = model_budget - 16
        max_tokens = min(max_len, effective_budget) - len(input_ids)

        if getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_tokens = min(max_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_tokens = min(max_tokens, kwargs.pop('max_new_tokens'))

        if max_tokens < 10:
            return None
        messages = kwargs.pop('messages', None) or decode_conversation(input_ids, self.tokenizer)[0]

        if self.meta_info and 'generation_kwargs' in self.meta_info:
            gen_kwargs = self.meta_info['generation_kwargs']
            if 'temperature' in gen_kwargs:
                kwargs['temperature'] = gen_kwargs['temperature']
            if 'top_p' in gen_kwargs:
                kwargs['top_p'] = gen_kwargs['top_p']
            if 'top_k' in gen_kwargs:
                pass

        for attempt in range(5):
            try:
                kwargs.pop('enable_thinking', None)
                kwargs.pop('uid', None)
                
                if 'extra_body' in kwargs:
                     if isinstance(kwargs['extra_body'], dict):
                         kwargs['extra_body'].pop('enable_thinking', None)
                         kwargs['extra_body'].pop('reasoning_content', None)
                
                if 'extra_body' not in kwargs:
                    kwargs['extra_body'] = {}
                kwargs['extra_body']['enable_thinking'] = False

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    **kwargs
                )

                text = response.choices[0].message.content or ""
                text_ids = self.tokenizer.encode(text, add_special_tokens=False)

                return {
                    "choices": [{
                        "message": {
                            "content": text,
                            "raw_output_ids": text_ids,
                            "extra_data": {"input_ids": input_ids},
                        }
                    }]
                }
            except Exception as e:
                import traceback
                error_type = type(e).__name__
                print(f"[CallAPI] Attempt {attempt + 1} failed. Error Type: {error_type}. Error: {e}")
                if "429" in str(e) or "quota" in str(e).lower():
                     print("[CallAPI] POSSIBLE QUOTA EXCEEDED OR RATE LIMIT REACHED.")

                if attempt == 4:
                    print(f"[CallAPI ERROR] Failed after 5 attempts. Last Error: {e}")
                    traceback.print_exc()
                    return None
                wait_time = 2 ** attempt
                print(f"[CallAPI] Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        return None


def truncate_prompt(chat, prompt_length, tokenizer, prompt_turn):
    exceed_len = len(tokenizer.apply_chat_template(chat[:prompt_turn])) + 8 - prompt_length
    _cut_idx = 0
    while exceed_len > 0:
        print('[PROMPT] now exceed', exceed_len, 'work on cut turn', _cut_idx)
        chat[_cut_idx]['content'] = tokenizer.decode(
            tokenizer.encode(chat[_cut_idx]['content'], add_special_tokens=False)[
                exceed_len + 4:], add_special_tokens=False)
        exceed_len = len(tokenizer.apply_chat_template(chat[:prompt_turn])) + 8 - prompt_length
        _cut_idx = _cut_idx + 1
        if _cut_idx >= prompt_turn:
            break
    return chat

class AgentContext:
    def __init__(self, chat, tokenizer, config, prompt_turn=2):
        self.tokenizer = tokenizer
        self.config = config
        self.init_len = len(chat)
        self.prompt_turn = prompt_turn
        self.prompt_length = config.prompt_length
        self.response_length = config.response_length
        self.context_uid = str(uuid.uuid4())

        self.chat = copy.deepcopy(chat)
        self.chat = truncate_prompt(self.chat, config.prompt_length, tokenizer, prompt_turn)
        self.chat_ids = [self.get_turn_context(i) for i in range(len(self.chat))]
        self.generation_prompt = None
        self.prompt_ids_len = len(sum(self.chat_ids[:prompt_turn], []))

    def get_turn_context(self, i):
        tokens = self.tokenizer.apply_chat_template(self.chat[:i + 1], add_generation_prompt=False, tokenize=True)
        prev = self.tokenizer.apply_chat_template(self.chat[:i], add_generation_prompt=False,
                                                  tokenize=True) if i > 0 else []
        turn_tokens = tokens[len(prev):]
        return turn_tokens

    def get_generation_prompt(self):
        if self.generation_prompt is None:
            tokens = self.tokenizer.apply_chat_template(self.chat, add_generation_prompt=False, tokenize=True)
            add_tokens = self.tokenizer.apply_chat_template(self.chat, add_generation_prompt=True,
                                                            tokenize=True)
            self.generation_prompt = add_tokens[len(tokens):]
        return self.generation_prompt

    def messages(self):
        return self.chat

    def context_ids(self, messages=None):
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def context(self, turn_cut: int=None):
        if turn_cut is not None:
            return sum(self.chat_ids[:turn_cut], []) + self.get_generation_prompt()
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def append(self, turn, completion=None):
        self.chat.append(turn)
        if completion is None:
            self.chat_ids.append(self.get_turn_context(len(self.chat) - 1))
        else:
            completion_tokens = completion["choices"][0]["message"]["raw_output_ids"]
            self.chat_ids.append(self.get_generation_prompt() + completion_tokens)
            if len(completion_tokens) == 0 or completion_tokens[-1] != self.tokenizer.eos_token_id:
                self.chat_ids[-1].append(self.tokenizer.eos_token_id)

    def rollback(self, k=1):
        self.chat = self.chat[:-k]
        self.chat_ids = self.chat_ids[:-k]

    def refresh_token_ids_from(self, start_index: int):
        if start_index < 0 or start_index >= len(self.chat):
            return
        self.generation_prompt = None
        for i in range(start_index, len(self.chat)):
            self.chat_ids[i] = self.get_turn_context(i)
        self.prompt_ids_len = len(sum(self.chat_ids[: self.prompt_turn], []))

    async def dataproto(self):
        return DataProto()


class Agent(AgentContext):
    def __init__(self, llm_client, conversations, tokenizer, config, prompt_turn=2):
        super().__init__(conversations, tokenizer, config, prompt_turn=prompt_turn)
        self.llm_client = llm_client
        self.retry_cjk = getattr(config.plugin, "retry_cjk", 0)

    async def step(self, max_new_tokens=None, retry_cjk=0, max_len=None):
        prompt = self.context()
        if max_len is None:
            max_len = len(prompt) + self.config.response_length
        if max_new_tokens is not None:
            max_len = min(len(prompt) + max_new_tokens, max_len)
        completion = await self.llm_client.create_completion(
            prompt, uid=self.context_uid, max_len=max_len, messages=self.chat)
        if completion is None:
            return None
        if max(self.retry_cjk, retry_cjk):
            if is_weird(completion["choices"][0]["message"]["content"]):
                for _ in range(int(max(self.retry_cjk, retry_cjk))):
                    completion = await self.llm_client.create_completion(
                        prompt, uid=self.context_uid, max_len=max_len, messages=self.chat)
                    if is_weird(completion["choices"][0]["message"]["content"]):
                        continue
                    else:
                        break
        response = completion["choices"][0]["message"]["content"]
        print(f"\n[Agent {self.context_uid[:4]}] Think:\n{response[:200]}...\n")
        self.append({'role': 'assistant', 'content': response, 'source': 'model_generation'}, completion)
        return response
@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    server_host: str
    server_port: int
    is_train: bool
    tokenizer: Optional[PreTrainedTokenizer] = None


async def run_action(env, response):
    try:
        try:
            act = time.time()
            env_return = await asyncio.wait_for(env.run_action(response), timeout=600.0)
            if time.time() - act > 10:
                print('Action Cost', time.time() - act)
        except asyncio.TimeoutError:
            print('[ACTION] Action timed out after 600 seconds')
            print(f'[ACTION] Timeout Content: {response}')
            env_return = {'observation': 'Action timed out after 300 seconds'}
        if 'action' in env_return:
            action, arguments = env_return['action'], env_return.get('arguments', {})
            if action == 'finish':
                return None
        observation = env_return.pop('observation', 'Empty')
        print(f"\n[Action Result]: {observation[:200]}...\n")
    except Exception as e:
        observation = f"Error: {e}"
    return observation
