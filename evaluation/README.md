# evaluation

G2-G6 gate evaluator. `gates.py` runs the scripted scenarios through the real pipeline
(live mode, `make gates`) or scores the committed mock telemetry (offline mode,
`make gates-offline`) and returns one `GateResult` per gate.

It is **production code**, not test scaffolding: `POST /api/eval` — the dashboard's
"Verify Gates" button — imports `run_offline_evaluation` from here. That is why it does
not live under `tests/`, which `.dockerignore` excludes from the image; while it did, the
endpoint returned `500 ModuleNotFoundError: No module named 'tests'` in production and
nowhere else.

```bash
python -m evaluation.gates                  # live mode
python -m evaluation.gates --mode offline   # over logs/mock_run.jsonl
```

`tests/` keeps the pytest suite and imports nothing from here.
