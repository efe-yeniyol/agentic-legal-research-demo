# Moved

The demo package now lives at [`src/northbridge/`](../src/northbridge/).

```bash
PYTHONPATH=src python -m northbridge.runner --method deterministic
PYTHONPATH=src uvicorn app.main:app --reload --port 8000
```

See the root [README.md](../README.md).
