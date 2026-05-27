# OPTA

OPTA stands for **Online Partial Trajectory Aggregating**.

This folder contains a cleaned, search-only version of the shared-state
aggregator experiment.  It keeps the basic workflow needed to run OPTA:

1. start a local retrieval server;
2. start one local vLLM OpenAI-compatible server;
3. run multiple search trajectories per question;
4. summarize each trajectory into chunk reports when context folding happens;
5. aggregate partial trajectory reports into a shared state;
6. inject useful shared-state information back into each trajectory;
7. select the final answer from all completed trajectories.

Only the search OPTA workflow and its basic JSON/JSONL outputs are included.

## Layout

- `run_opta.sh`: launch script.
- `opta/evaluator.py`: OPTA controller and evaluation loop.
- `opta/agent.py`: search-only rollout agent with chunk-report context folding.
- `opta/search_env.py`: search/open/finish environment and answer judge.
- `opta/search_server.py`: local retrieval server.
- `opta/shared_state_prompts.py`: prompts for shared-state construction,
  trajectory-specific injection, and final answer integration.

## Run

Edit paths as needed, then run:

```bash
cd OPTA
bash run_opta.sh
```

Common overrides:

```bash
DATA_PATH=/path/to/data.parquet \
MODEL_PATH=/path/to/instruct-model \
MODEL_NAME=Your-Served-Model-Name \
EMBEDDING_MODEL_PATH=/path/to/embedding-model \
OPTA_TRAJECTORIES=10 \
NUM_SAMPLES=all \
bash run_opta.sh
```

The script writes one run directory under `evaluation_results/` with
`results_opta.jsonl` and `results_opta.json`. These files keep only minimal
evaluation fields such as instance id, query, reference answer, selected answer,
score, selected trajectory id, and number of trajectories. They do not save raw
trajectory histories, chunk reports, shared states, or injection records.

## Dataset Format

The evaluator expects a parquet file with at least:

- `query`: question text;
- `answer`: reference answer;
- optionally `instance_id` or `query_id`.

Rows with the older `extra_info` schema are also accepted, but the workflow is
forced to `search`.
