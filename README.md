# Quantitative Analysis Program

A multi-model quantitative analysis ecosystem. Twenty models — spanning
structural (layer 2), ML / signal generation (layer 4), execution (layer 5),
and risk (layer 6) — plug into a shared core. Market data is sourced from
[yfinance](https://github.com/ranaroussi/yfinance).

## Layout

```
src/
  core/                 # data access, model contract, registry, config — shared
  models/               # one subfolder per model — added incrementally
tests/
  core/                 # core unit tests (must stay green)
  integration/          # cross-model integration tests
pyproject.toml          # ruff / mypy / pytest config
CONTRIBUTING.md         # one-chat-per-model workflow
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Adding a model

Read [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Open a fresh chat session for that model.
2. Create `src/models/<model_name>/model.py` containing a `BaseModel` subclass
   decorated with `@register_model`.
3. Add tests under `tests/models/<model_name>/`.
4. Do **not** modify `src/core/`.
