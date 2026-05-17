# Threat Model

This page describes DocFlow's security boundaries for a local desktop deployment.
It is not a promise that every host is hardened; it is the checklist maintainers
use when changing storage, parsing, model, network, or release behavior.

## Assets

- Local source documents in watched folders and uploaded files.
- SQLite metadata, Qdrant vectors, generated notes, saved answers, and backups.
- Prompts, retrieved snippets, model responses, citations, and source-preview text.
- `config.yaml`, optional cloud model keys, local model cache paths, and logs.
- Release artifacts: Docker images, Python package artifacts, and GitHub Actions outputs.

## Trust Boundaries

- Browser to DocFlow API: intended for local use on `localhost:8000`; do not expose it
  to an untrusted network without an external access-control layer.
- DocFlow to filesystem: watched folders and upload paths can contain untrusted files,
  so parsers must fail visibly and avoid silent data loss.
- DocFlow to Qdrant and SQLite: local metadata and vectors are trusted only as local
  state; deletion, rebuild, and migration paths must be explicit.
- DocFlow to local model tools: Ollama, LM Studio, MLX, and model caches are outside
  the Python process but expected to stay on the user's machine in local mode.
- DocFlow to the internet: model downloads, user-triggered webpage import, and cloud
  model backends are intentionally external and must stay opt-in.
- GitHub release workflows: CI builds package and container artifacts; release notes
  must identify which artifact was validated.

## Main Risks And Controls

| Risk | Existing control | Remaining gap |
| --- | --- | --- |
| Unexpected outbound traffic | Offline doctor covers startup, ingest, query, model status, and source preview | Webpage import and cloud model backends are intentionally external and need user review |
| Silent model download | `privacy.allow_model_download` defaults to `false` | Users still need to pre-populate local caches or explicitly allow downloads |
| Cloud model leakage | Cloud backends require explicit backend/key configuration and show UI notice | Secrets in local config are not encrypted by DocFlow |
| Parser crash or bad file input | Parser/eval tests and visible failure paths | Parsers are not isolated in a sandbox process |
| Local database or backup exposure | Runtime data is ignored by git and stays local | SQLite databases and backups are not encrypted at rest |
| Browser API exposure | Defaults target local browser use | No built-in authentication if users bind the service to a broader network |
| Supply-chain drift | Pinned dependencies, pinned workflow actions, pinned Docker/Qdrant images, dependency audit, CodeQL, Dependabot, release-surface check | OpenSSF Scorecard should be reviewed before releases |
| Release artifact tampering | Python package checksums plus Docker SBOM/provenance in release workflows | Release artifacts are not signed yet |

## Maintainer Checklist

- New network access must be documented in `docs/privacy.md` and covered by the
  offline doctor or explicitly marked user-triggered.
- New model paths must update `docs/model-licenses.md` and state whether weights
  are downloaded, locally provided, or served by a cloud provider.
- Parser, backup, restore, migration, and release changes must include tests that
  prove failure is visible instead of silently hidden.
- Public release artifacts must pass CI, package smoke, release-surface checks,
  dependency audit, and OpenSSF Scorecard review before tag notes call them ready.
- Workflow action updates and Docker base-image updates must keep pinned commit
  hashes or image digests in the same pull request.
