# DocFlow Roadmap

DocFlow is moving from a capable local RAG app toward a mature personal knowledge product. The roadmap is intentionally focused on trust, usability, and maintainability.

## 1. Public Project Readiness

- Keep README concise, evidence-based, and aligned with real commands.
- Maintain public docs for features, architecture, privacy, CLI, development, evaluation, and status.
- Keep contribution, security, code of conduct, changelog, and issue templates current.

## 2. First-Run Experience

- Provide a full `docker compose up` path for the app and local services.
- Offer a small demo library so new users can see answers with sources quickly.
- Make empty states useful: import demo data, choose a local folder, or upload files.

## 3. Code Quality and Architecture

- Make full lint, type checking, and tests required in CI.
- Split large API, storage, and retrieval modules into smaller focused pieces.
- Keep public command behavior stable while internal modules improve.

## 4. Trustworthy Answers

- Strengthen citations with stable chunk identity and source spans.
- Highlight the cited source text in the browser preview.
- Reject or mark citations that cannot be matched to retrieved evidence.

## 5. Measured Quality

- Expand retrieval and parsing evaluation sets.
- Add incremental indexing checks.
- Track latency, storage use, and large-library behavior.
- Publish measured results instead of subjective maturity scores.

## 6. Local Privacy

- Keep default behavior local and auditable.
- Make model downloads and cloud model use explicit.
- Keep offline checks covering startup, ingest, query fallback, model status, and source preview.

## 7. Knowledge Workspace Depth

- Add topic views, similar-document detection, knowledge cards, and periodic reviews.
- Connect answers, notes, and source documents into a stronger knowledge workflow.
