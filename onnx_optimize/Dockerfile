FROM python:3.12.5-slim AS builder

RUN pip install --upgrade pip==24.2.0 && \
    pip install poetry==1.8.3

WORKDIR /app

COPY pyproject.toml poetry.toml poetry.lock ./

RUN poetry install

FROM python:3.12.5-slim AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv .venv