"""Search-only rollout agent used by OPTA."""
import os
import re
import time
import copy
import asyncio

import numpy as np

from opta.data import DataProto
from .utils import CallLLM, Agent, select_env, TaskContext, CallAPI, run_action
from .prompts import SUMMARY_PROMPT_SEARCH, create_search_chat


def extract_fn_call(text):
    if text is None:
        return None
    func_matches = re.findall(r'<function=([^>]+)>', text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f'<function={last_function}>')
    text_after_last_func = text[last_func_pos:]
    params = dict(re.findall(r'<parameter[=>]?([^>]+)>(.*?)</parameter>', text_after_last_func, re.DOTALL))
    return {'function': last_function, 'arguments': params}


def extract_report(text: str) -> str:
    matches = re.findall(r'<report>(.*?)</report>', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r'<summary>(.*?)</summary>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def latest_unsummarized_messages(messages):
    last_summary_idx = -1
    for idx, msg in enumerate(messages):
        if msg.get("source") == "model_summary_generation":
            last_summary_idx = idx
    return copy.deepcopy(messages[last_summary_idx + 1:])


async def process_item(
        item: DataProto,
        context: TaskContext,
        LLMClass=CallLLM,
        **kwargs
) -> DataProto:
    """Evaluate one search item with chunk reports and an optional OPTA hook."""
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    ability = "search"

    EnvClass = select_env()
    print(is_train, EnvClass)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = "search"
    item.non_tensor_batch['extra_info'][0]['workflow'] = workflow

    user_prompt = create_search_chat(env.instance_info['problem_statement'])

    question_text = env.instance_info.get('problem_statement', '') or ''
    summary_prompt = SUMMARY_PROMPT_SEARCH.format(question=question_text)

    TRUNCATION_NOTICE = "\n\n[Note: The above observation was truncated due to context token limit.]"
    _summary_prompt_tokens = len(tokenizer.encode(summary_prompt, add_special_tokens=False))
    _notice_tokens = len(tokenizer.encode(TRUNCATION_NOTICE, add_special_tokens=False))
    SUMMARY_TOKEN_RESERVE = _summary_prompt_tokens + 256 + 40
    try:
        chunk_report_output_tokens = int(os.environ.get("OPTA_CHUNK_REPORT_OUTPUT_TOKENS", "4096"))
    except (TypeError, ValueError):
        chunk_report_output_tokens = 4096

    max_turn = agent_config.get("max_turn", 64)
    max_session = getattr(config.plugin, "max_session", 5)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    enable_summary = getattr(config.plugin, "enable_summary", False)

    host = context.server_host
    port = context.server_port

    llm_client = LLMClass(host, port, tokenizer, config, meta_info=agent_config.get("meta_info", {}))
    sampling_overrides = kwargs.get("trajectory_sampling_overrides") or {}
    if sampling_overrides and hasattr(llm_client, "meta_info"):
        generation_kwargs = copy.deepcopy(llm_client.meta_info.get("generation_kwargs", {}))
        for key in ("temperature", "top_p", "top_k", "max_tokens"):
            if sampling_overrides.get(key) is not None:
                generation_kwargs[key] = sampling_overrides[key]
        llm_client.meta_info["generation_kwargs"] = generation_kwargs

    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    init_len = len(agent['main'].context())
    current = 'main'
    session_start_time = time.time()
    iteration = 0
    session_message = []
    summary_round_idx = 0
    latest_accumulated_report = ""

    # Cap total sequence length to fit within vLLM's max_model_len.
    # Use the actual max_model_len from env (set by --max-model-len in vLLM),
    # falling back to prompt_length + response_length if not set.
    max_model_budget = int(os.environ.get(
        'MAX_MODEL_LEN',
        config.prompt_length + config.response_length
    ))
    obs_budget_raw = os.environ.get("OPTA_OBS_TRUNCATION_BUDGET")
    if obs_budget_raw:
        try:
            obs_truncation_budget = int(obs_budget_raw)
        except (TypeError, ValueError):
            obs_truncation_budget = max_model_budget - chunk_report_output_tokens
    else:
        obs_truncation_budget = max_model_budget - chunk_report_output_tokens
    obs_truncation_budget = max(min(obs_truncation_budget, max_model_budget), 1024)

    def _safe_summary_step(cur_agent, max_desired_tokens=4096, min_tokens=256):
        """Calculate summary generation parameters that fit within max_model_budget.

        Returns (new_tokens, safe_max_len) clamped so that
        ctx_len + new_tokens <= max_model_budget, avoiding the case
        where create_completion gets max_tokens < 10 and returns None.
        """
        ctx_len = len(cur_agent.context())
        available = max_model_budget - ctx_len
        new_tokens = min(max_desired_tokens, max(available, 0))
        safe_max_len = min(ctx_len + new_tokens, max_model_budget)
        print(f"[Summary Budget] ctx={ctx_len}, budget={max_model_budget}, "
              f"available={available}, new_tokens={new_tokens}, safe_max_len={safe_max_len}")
        return new_tokens, safe_max_len

    while iteration < max_turn:
        if time.time() - session_start_time > session_timeout:
            print('[SESSION] Session Timeout')
            session_message.append({'role': 'user', 'content': '[SESSION] Session Timeout', 'source': 'system_error'})
            
            agent[current].append({'role': 'user', 'content': f"Execution time limit exceeded. {summary_prompt}"})
            s_tokens, s_max_len = _safe_summary_step(agent[current])
            summary_response = await agent[current].step(max_new_tokens=s_tokens, max_len=s_max_len)
            if summary_response:
                 session_message.append({'role': 'assistant', 'content': summary_response, 'source': 'model_summary_generation'})
            break

        iteration += 1
        if iteration >= max_turn:
            msg = '[SESSION] Max Turn Limit Reached'
            print(msg)
            session_message.append({'role': 'user', 'content': msg, 'source': 'system_error'})
            
            agent[current].append({'role': 'user', 'content': f"Interaction turn limit reached. {summary_prompt}"})
            s_tokens, s_max_len = _safe_summary_step(agent[current])
            summary_response = await agent[current].step(max_new_tokens=s_tokens, max_len=s_max_len)
            if summary_response:
                 session_message.append({'role': 'assistant', 'content': summary_response, 'source': 'model_summary_generation'})
            break

        # Context Folding (Summary) — the only agent switching mechanism
        if enable_summary and len(agent[current].context()) - init_len > config.prompt_length * float(os.environ.get('SUMMARY_THRESHOLD', '0.95')):

            print(f"\n[Context Folding] Triggered at length {len(agent[current].context()) - init_len}/{config.prompt_length}...\n")
            
            if len(agent) >= max_session:
                msg = f'[SESSION] Session OOC after session {len(agent)}'
                print(msg)
                session_message.append({'role': 'user', 'content': msg, 'source': 'system_error'})

                agent[current].append({'role': 'user', 'content': f"Maximum context folding sessions ({max_session}) reached. {summary_prompt}"})
                s_tokens, s_max_len = _safe_summary_step(agent[current])
                summary_response = await agent[current].step(max_new_tokens=s_tokens, max_len=s_max_len)
                if summary_response:
                    session_message.append({'role': 'assistant', 'content': summary_response, 'source': 'model_summary_generation'})
                break

            agent[current].append({'role': 'assistant', 'content': ""})
            final_summary_prompt = summary_prompt
            agent[current].append({'role': 'user', 'content': final_summary_prompt})
            session_message.append({'role': 'user', 'content': final_summary_prompt, 'source': 'system_summary_request'})
            s_tokens, s_max_len = _safe_summary_step(
                agent[current],
                max_desired_tokens=chunk_report_output_tokens,
            )
            response = await agent[current].step(max_new_tokens=s_tokens, max_len=s_max_len)

            if response is None:
                session_message.append({'role': 'user', 'content': '[SYSTEM] Summary Generation Failed (LLM returned None)', 'source': 'system_error'})
                break
            session_message.append({'role': 'assistant', 'content': response, 'source': 'model_summary_generation'})
            chunk_report = extract_report(response) or response
            summary_round_idx += 1
            accumulated_report = chunk_report

            summary_hook = kwargs.get("summary_hook")
            if summary_hook is not None:
                try:
                    hook_result = await summary_hook(
                        trajectory_id=kwargs.get("trajectory_id"),
                        chunk_report=chunk_report,
                        previous_accumulated_report=latest_accumulated_report,
                        round_idx=summary_round_idx,
                        instance_id=item.non_tensor_batch['extra_info'][0].get('instance_id'),
                        question=question_text,
                    )
                    if isinstance(hook_result, dict) and hook_result.get("accumulated_report"):
                        accumulated_report = str(hook_result["accumulated_report"])
                except Exception as exc:
                    print(f"[OPTA HOOK ERROR] {exc}")

            latest_accumulated_report = accumulated_report
            next_session_prompt = (
                "For this question, you have already made the following progress "
                "in previous session(s), summarized as an accumulated report:\n\n"
                f"{accumulated_report}\n\nNow continue work on it."
            )
            current = current + '+'
            agent[current] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn) 
            agent[current].append({'role': 'assistant', 'content': ""})
            agent[current].append({'role': 'user', 'content': next_session_prompt})
            session_message.append({'role': 'user', 'content': next_session_prompt, 'source': 'system_summary_injection'})
            continue

        # Generate Response (Think / Act)
        response = await agent[current].step()

        if response is None:
            session_message.append({'role': 'user', 'content': '[SYSTEM] LLM Error', 'source': 'system_error'})

            try:
                agent[current].append({'role': 'user', 'content': f"A system error occurred during generation. {summary_prompt}"})
                s_tokens, s_max_len = _safe_summary_step(agent[current])
                summary_response = await agent[current].step(max_new_tokens=s_tokens, max_len=s_max_len)
                if summary_response:
                    session_message.append({'role': 'assistant', 'content': summary_response, 'source': 'model_summary_generation'})
            except Exception:
                pass
            break

        session_message.append({'role': 'assistant', 'content': response, 'source': 'model_generation'})

        fn_call = extract_fn_call(response)

        if fn_call is not None and fn_call['function'] not in ('search', 'open_page', 'finish'):
            print(f"[SEARCH_ONLY] Rejected unavailable '{fn_call['function']}' tool call")
            observation = (
                f"Error: The '{fn_call['function']}' tool is not available. "
                f"You only have access to: search, open_page, finish. "
                f"Please use these tools to continue your research."
            )
            agent[current].append({'role': 'assistant', 'content': response})
            agent[current].append({'role': 'user', 'content': observation, 'source': 'system_error'})
            session_message.append({'role': 'environment', 'content': observation, 'source': 'system_error'})
            continue

        env._current_agent_name = current
        observation = await run_action(env, response)
        
        if observation is None:
            session_message.append({'role': 'user', 'content': '[SYSTEM] Agent called finish() tool', 'source': 'system_exit'})
            break

        if agent[current].chat[-1]['role'] == 'user':
            print('[ROLE ERROR]')
            print(agent[current].chat[-1])
            agent[current].append({'role': 'assistant', 'content': str(response)})

        ctx_before = len(agent[current].context())
        obs_ids = tokenizer.encode(observation, add_special_tokens=False)
        chat_overhead = 20
        max_obs_tokens = obs_truncation_budget - ctx_before - SUMMARY_TOKEN_RESERVE - _notice_tokens - chat_overhead
        if len(obs_ids) > max_obs_tokens:
            if max_obs_tokens > 0:
                observation = tokenizer.decode(obs_ids[:max_obs_tokens], skip_special_tokens=True) + TRUNCATION_NOTICE
            else:
                observation = TRUNCATION_NOTICE.strip()
            print(f"[Observation Truncated] {len(obs_ids)} -> ~{max(max_obs_tokens, 0) + _notice_tokens} tokens "
                  f"(ctx_before={ctx_before}, obs_budget={obs_truncation_budget}, reserve={SUMMARY_TOKEN_RESERVE})")

        agent[current].append({'role': 'user', 'content': observation, 'source': 'environment'})
        session_message.append({'role': 'environment', 'content': observation, 'source': 'environment'})
    

    print('[TASK] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent[current].messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    outs = []

    is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
    if not is_finish:
        score = ('', 0)

    for name in agent if is_train else ['main']:
        out = await agent[name].dataproto()
        out = await env.update_dataproto(out, item, score)
        out.non_tensor_batch['recent_messages'] = np.array(
            [latest_unsummarized_messages(session_message)],
            dtype=object,
        )
        predicted_answer = ""
        if getattr(env, "predicted_answer", None):
            predicted_answer = env.predicted_answer[0]
        out.non_tensor_batch['predicted_answer'] = np.array([predicted_answer], dtype=object)
        outs.append(copy.deepcopy(out))

    try:
        res = DataProto.concat(outs)
        return res
    except Exception as e:
        breakpoint()
        return
    finally:
         if hasattr(llm_client, 'close'):
             await llm_client.close()
         if 'env' in locals() and hasattr(env, 'close'):
             await env.close()
