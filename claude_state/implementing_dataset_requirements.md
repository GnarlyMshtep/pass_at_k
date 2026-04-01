# Dataset Hyperparam Requirements

## Problem
Preprocessing scripts produce datasets that require specific training hyperparams (e.g. `shuffle=false`, `total_epochs=1`). Without enforcement, running with wrong settings silently corrupts training (e.g. prompt-phase boundaries shift when `filter_overlong_prompts=true` removes rows from an ordered dataset).

## Solution: Sidecar `dataset_requirements.json`

Preprocessing scripts write a `dataset_requirements.json` alongside their parquet files:

```
apps_multiphase_regular_t160_b40_p500_filt1024/
  train.parquet
  test.parquet
  dataset_requirements.json   ← checked by validate_env
```

### Format
```json
{
    "data.shuffle": false,
    "data.filter_overlong_prompts": false,
    "trainer.total_epochs": 1,
    "trainer.test_freq": 20
}
```

Keys are dot-separated Hydra paths matching the merged config structure. Values are the required settings.

### Enforcement
`validate_env.py` (`check_dataset_requirements`):
1. Derives dataset dir from `--train-path` (parent directory)
2. Looks for `dataset_requirements.json`
3. If found, checks each requirement against the merged config (passed via `--merged-config-json`)
4. On mismatch: hard error with clear message
5. No requirements file: passes silently

### Data Flow
```
1. Preprocessing script saves train.parquet + dataset_requirements.json
2. VFH override config points to the dataset
3. Orchestrator resolves config, passes --merged-config-json to validate_env
4. validate_env reads requirements file, compares against merged config
5. Mismatch → validation fails, run does not start
```

### How to Add Requirements to a New Preprocessing Script
```python
import json

requirements = {
    "data.shuffle": False,
    "data.filter_overlong_prompts": False,
    "trainer.total_epochs": 1,
    "trainer.test_freq": 20,
}
with open(os.path.join(output_dir, "dataset_requirements.json"), "w") as f:
    json.dump(requirements, f, indent=2)
```

### Reference Implementation
`custom/data_preprocessing/APPS/preprocess_apps_multiphase.py` — writes requirements for the multi-phase APPS dataset.

### Key Files
- `validate_env.py:check_dataset_requirements()` — the check function
- `vfh/orchestrator.py:_run_validation()` — passes `--merged-config-json`
