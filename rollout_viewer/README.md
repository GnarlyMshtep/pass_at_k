Rollout Viewer

A lightweight FastAPI + vanilla JS app to browse rollout directories and inspect JSONL files interactively.

Features
- Navigate folders/files under a configurable base root
- Open `.jsonl` files and page through entries (infinite scroll)
- Renders arbitrary/dynamic JSON fields without hardcoding
- Long text values shown in resizable, scrollable boxes

Configure base root
Set `ROLLOUT_ROOT` (defaults to `/scratch/m000122/stalaei/logs/pass_at_k/rollouts`). Example:

```bash
export ROLLOUT_ROOT=/scratch/m000122/stalaei/logs/pass_at_k/rollouts
```

Install and run
```bash
cd /users/stalaei/code/pass_at_k/rollout_viewer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ROLLOUT_ROOT=/scratch/m000122/stalaei/logs/pass_at_k/rollouts uvicorn main:app --host 0.0.0.0 --port 8090 --reload
```

Open `http://localhost:8090`.

Notes
- Only `.jsonl` files are loadable in the viewer.
- Pagination reads sequentially to avoid loading large files into memory.
- The UI attempts to auto-detect long text fields (e.g., prompts/outputs) and display them in text boxes; other scalars are rendered as small pills.

