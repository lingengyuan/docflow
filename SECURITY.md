# Security Policy

DocFlow is built for local private data. Security reports should focus on protecting local documents, local indexes, model prompts, generated notes, and user configuration.

## Supported Versions

Security fixes target the current `main` branch until formal releases are established.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the repository owner before opening a public issue. Include:

- A short summary of the issue.
- Steps to reproduce.
- Affected files, commands, or UI flows.
- Whether local documents, prompts, paths, model outputs, or network traffic are exposed.
- Any suggested mitigation.

If private contact is not available, open a GitHub issue with minimal public detail and avoid posting private files, secrets, or exploit payloads.

## Security Boundaries

DocFlow should not:

- Send telemetry, analytics, or automatic error reports.
- Upload documents for product analytics.
- Contact external model or cloud services unless the user explicitly configures that behavior.
- Hide reduced answer quality or missing source evidence behind silent fallback behavior.

Expected local services include the browser app, SQLite, Qdrant on localhost, and optionally Ollama on localhost.

The detailed maintainer threat model lives in [docs/threat-model.md](docs/threat-model.md).

## Maintainer Checklist

For security-related changes:

- Run the test suite.
- Run the offline network check.
- Review new network access paths.
- Confirm generated files and local runtime data are ignored by git.
- Keep GitHub Actions, Docker base images, and service container images pinned to reviewed commits or image digests. Public application tags such as `edge` may remain moving convenience tags and must not be described as signed or immutable.
- Confirm release artifacts have checksums, and do not claim signed releases until signing is actually enabled.
- Update `docs/privacy.md` when behavior changes.
