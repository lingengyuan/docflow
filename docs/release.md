# Release Process

Use this checklist before tagging a public DocFlow release.

## 1. Prepare

- Confirm `pyproject.toml` has the release version.
- Confirm `CHANGELOG.md` has a dated entry for the release.
- Confirm `README.md` and `docs/status.md` show the latest measured validation results.
- Confirm screenshots in `docs/assets/` still match the current browser UI when the UI changed.
- Confirm known limitations are listed in `docs/status.md`.
- Confirm install cost, model size boundaries, and upgrade notes are current in `README.md`, `docs/development.md`, and `docs/architecture.md`.
- Confirm Docker image release notes name the exact image tag, for example `ghcr.io/lingengyuan/docflow:v0.58.0`.

## 2. Validate

Run the release checks from a clean working tree:

```bash
scripts/run_ci.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 docflow eval public --write-results
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 docflow eval retrieval --refresh-sources --source-filter --write-results
docflow eval parsing --write-results
docflow browser-acceptance
docflow doctor --offline
```

For visible UI changes, open the browser workspace and click the main flows for chat, library, notes, source preview, and settings.

## 3. Update Status

When a validation number changes, update both places in the same commit:

- `README.md` Current Verification / 当前验证结果
- `docs/status.md` Latest Local Validation

Use measured command output only. Do not replace these results with subjective maturity scores.

## 4. Tag

Create a tag after the release commit is pushed:

```bash
git tag v0.58.0
git push origin v0.58.0
```

Use the actual version for the release you are publishing.

Tagged releases build:

- Python wheel and source archive artifacts through `.github/workflows/python-package.yml`.
- GHCR Docker images through `.github/workflows/docker-image.yml`.

DocFlow is not published to PyPI yet. Before enabling PyPI publishing, verify package data includes browser assets, config templates, docs needed at runtime, and that optional heavy dependencies remain optional.

## 5. Release Notes

GitHub release notes should include:

- What changed.
- Who should upgrade.
- Validation results.
- Screenshots when UI changed.
- Known limitations.
- Privacy or network behavior changes.
- Install and upgrade notes, including whether users need to rebuild Qdrant vectors or update local models.

## 6. After Release

- Confirm CI, CodeQL, and Dependabot are active.
- Confirm dependency audit checks pass for Python and frontend dependencies.
- Confirm issue templates and pull request template still match the project workflow.
- Start the next changelog entry only after the next user-facing or maintainer-facing change lands.
