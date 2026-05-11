# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Install optional image understanding support only when needed:

```bash
pip install -r requirements-vision.txt
```

## Run Tests

```bash
.venv/bin/python -m pytest
```

## Run the App

```bash
docflow serve
```

## Frontend Styles

Committed CSS is enough to run the app. Rebuild styles only after frontend style changes:

```bash
npm install
npm run build:css
```

## Project Rule

Normal browser UI must feel like a finished personal knowledge product. Developer-only commands, repair instructions, scripts, and recovery details belong in docs, CLI, tests, or internal implementation, not in the user-facing app.
