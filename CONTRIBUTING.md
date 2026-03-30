# Contributing

Thanks for contributing to this repository.

## Before You Start

- Use Python 3.11 or newer.
- Install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

- Copy the environment template when you need local environment variables:

```bash
cp .env.example .env
```

## Local Workflow

Typical workflow:

1. Sync or prepare local data under `storage/`
2. Run the relevant CLI command or web flow
3. Run tests before opening a change

Example:

```bash
python3 -m pytest
```

## Scope Expectations

This repository is intentionally narrow in scope:

- A-share daily research and backtesting
- local single-user workflows
- protocol-constrained strategy execution

Please avoid broadening the project into a general-purpose quant platform unless the change is explicitly discussed first.

## Pull Request Notes

When opening a pull request, include:

- what changed
- why it changed
- how you validated it
- whether the change affects data layout, configs, or web-console behavior

## Data And Secrets

- Do not commit `.env`, tokens, or other credentials.
- Do not commit large local research artifacts from `results/`, `research/factors/`, `research/models/`, or private `storage/` snapshots unless the change is specifically about tracked demo data.
- If you add a public demo dataset, keep it small and document exactly what it is for.
