"""Configuration helpers for OPTA."""

from __future__ import annotations

import os

from omegaconf import OmegaConf


def create_config():
    prompt_len = int(os.getenv("OPTA_PROMPT_LEN", 54000))
    response_len = int(os.getenv("OPTA_RESPONSE_LEN", 4096))
    max_turn = int(os.getenv("OPTA_MAX_TURN", 300))
    max_session = int(os.getenv("OPTA_MAX_SESSION", 100))
    session_timeout = int(os.getenv("OPTA_SESSION_TIMEOUT", 10000))

    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": prompt_len,
                    "response_length": response_len,
                    "plugin": {
                        "workflow": "search",
                        "max_turn": max_turn,
                        "max_session": max_session,
                        "session_timeout": session_timeout,
                        "enable_summary": True,
                    },
                }
            }
        }
    )
