# Contributing

This repo hosts the **shared core** plus **twenty quant models**. The architecture
deliberately separates the two so that each model can be developed in isolation
by a different contributor — including, in our case, twenty separate Claude
Code chat sessions running in parallel.

## Architecture in one paragraph

`src/core/` defines the data access layer (`DataProvider`), the model contract
(`BaseModel`), the shared result dataclasses (`Signal`, `Forecast`, `RiskMetric`,
`CalibrationResult`), and a decorator-based registry (`@register_model`). Every
quant model is a `BaseModel` subclass that lives in its own subfolder under
`src/models/<model_name>/`. The orchestration layer (not in this PR) iterates
the registry, asks each model for its inputs via the `DataProvider`, and routes
the output of `predict()` downstream.

## The rules

### One chat per model
Each model is implemented in a single, focused Claude Code chat. That chat is
responsible for:

- Creating `src/models/<model_name>/` with `__init__.py` and `model.py`
- Implementing a `BaseModel` subclass decorated with `@register_model`
- Adding tests under `tests/models/<model_name>/`
- Wiring the model into `src/models/__init__.py` so the registry sees it on import

Do **not** open a chat that touches multiple models. If you find yourself needing
to, that's a signal the core needs a change — see below.

### The core is read-only from a model chat
Model chats may **read** anything under `src/core/`, but they may **not modify**
it. The core contract is the load-bearing interface twenty models depend on;
changing it from inside a model chat will silently break the others.

If a model genuinely needs a core change:

1. Stop that chat.
2. Open a separate "core change" chat describing the missing capability.
3. Land the core change first, with tests.
4. Resume model work against the updated core.

### What lives where
| Path                                  | Owner             | Mutability from model chat |
|---------------------------------------|-------------------|----------------------------|
| `src/core/`                           | core maintainer   | read-only                  |
| `src/models/<model_name>/`            | that model's chat | full ownership             |
| `tests/core/`                         | core maintainer   | read-only                  |
| `tests/models/<model_name>/`          | that model's chat | full ownership             |
| `tests/integration/`                  | core maintainer   | additions allowed          |
| `pyproject.toml`                      | core maintainer   | additions allowed          |

### Adding a dependency
Append it to `dependencies` in `pyproject.toml` (or `[project.optional-dependencies].dev`).
Justify the addition in your commit message — every dependency is something the
other nineteen models will inherit.

## The model contract (cheat sheet)

```python
from src.core import BaseModel, DataProvider, Signal, register_model, CalibrationResult
from datetime import datetime

@register_model
class MyModel(BaseModel):
    name = "my_model"          # globally unique, snake_case
    layer = 4                  # 2=structural, 4=signals/ML, 5=execution, 6=risk
    refit_frequency = "daily"  # tick | intraday | daily | weekly | monthly

    def fetch_data(self, provider: DataProvider):
        return provider.fetch_prices("AAPL", "5y", "1d")

    def calibrate(self, data) -> CalibrationResult:
        ...

    def predict(self, data) -> Signal:
        return Signal("AAPL", "long", 0.7, datetime.utcnow(), "1d")

    def validate(self, data) -> dict:
        return {"residual_mean": 0.0}
```

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

pytest                       # tests/core/ must stay green; add your own under tests/models/
ruff check .
mypy src/core
```

CI runs the same three commands. A PR that breaks `pytest`/`ruff`/`mypy` on
the core will be rejected regardless of how the model itself behaves.
