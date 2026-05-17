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
- Confirm `docs/threat-model.md` and `docs/model-licenses.md` still match the release behavior.
- Confirm OpenSSF Scorecard has a recent run and review any high-risk findings before tagging.
- Confirm workflow actions, Docker bases, and Qdrant service images are still pinned to the intended commits or image digests.
- Confirm `scripts/run_release_surface_check.py` passes so the public docs, Docker files, workflows, package data, and ignored internal history are aligned.
- Confirm `scripts/package_smoke.py` passes before treating wheel artifacts as releasable.
- Confirm the scheduled evaluation workflow has a recent successful run before quoting public retrieval numbers in release notes.

## 2. Validate

Run the release checks from a clean working tree:

```bash
scripts/run_ci.sh
python scripts/run_release_surface_check.py
docflow dev eval public --write-results
docflow dev eval retrieval --refresh-sources --source-filter --write-results
docflow dev eval parsing --write-results
docflow dev browser-acceptance
docflow doctor --offline
```

Review the latest OpenSSF Scorecard workflow result in GitHub Actions before tagging. Treat repository-setting findings separately from code changes: branch protection, required reviews, and merge policy must be configured in GitHub, not in this repository.

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

The main branch publishes `ghcr.io/lingengyuan/docflow:edge` for no-build smoke usage after CI. Tagged releases build:

- Python wheel and source archive artifacts through `.github/workflows/python-package.yml`.
- `SHA256SUMS` for Python package artifacts through `.github/workflows/python-package.yml`.
- GHCR Docker images through `.github/workflows/docker-image.yml`.
- Docker image SBOM and provenance attestations through `.github/workflows/docker-image.yml`.
- OpenSSF Scorecard SARIF through `.github/workflows/scorecard.yml`.
- The release surface check, parsing eval, performance smoke, and package smoke test are part of GitHub CI and the local CI script so public docs, install paths, package data, and internal-file exclusions are checked before release work.
- The scheduled evaluation workflow runs the full public retrieval eval with Qdrant and model downloads isolated from normal pull-request CI.

DocFlow is not published to PyPI yet. Wheel artifacts now include browser assets, config templates, and runtime docs, and the installed-wheel smoke test must pass before a release. Before enabling PyPI publishing, review optional heavy dependencies and publish policy separately.
If PyPI publishing is enabled later, use Trusted Publishing instead of a long-lived token.

Release artifacts are not signed yet. Do not describe a release as signed until a real signing flow is added and validated. Current integrity coverage is limited to package checksums, Docker SBOM/provenance output, dependency audit, CodeQL, and OpenSSF Scorecard review.

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
