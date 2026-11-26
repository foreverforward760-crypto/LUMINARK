# LUMINARK

**An AI-assisted software research portfolio built around Richard Stanfield's 0-9-0 framework / Stanfield's Axiom of Perpetuity (SAP).**

## Project status

LUMINARK is an experimental collection of software prototypes, demonstrations, and research components. The repository includes stage-classification logic, rule-based safety examples, neural-network experiments, dashboards, biofeedback demonstrations, and supporting utilities.

**This repository is not established as a production-grade AI framework.** Some demonstrations use mock or synthetic data. Automated tests exercise selected programmed scenarios; they do not establish real-world accuracy, reliable safety performance, consciousness, clinical efficacy, or predictive validity.

## Conceptual origin and implementation history

The project owner states that the underlying ideas, concepts, and framework in this portfolio originated with him. AI systems generated the software code and suggested technical mechanisms to implement and operationalize those concepts. Supporting constructs introduced during that development process—including NSDT and the Container Rule—are treated as implementation mechanisms within the owner's framework, not as independent origins of the concepts.

This statement records the project owner's account. It is not a legal determination of copyright ownership, patent inventorship, or exclusive rights. Code provenance, third-party dependencies, applicable licenses, and AI-service terms require separate review before commercial transfer or licensing.

## What is in this repository?

The repository contains several distinct experiments rather than one unified, validated product:

- **SAP / 0-9-0 stage model:** code representing a ten-stage framework and selected transition or inversion rules.
- **Ma'at and Yunus modules:** rule-based text-pattern checks and response/containment concepts. These are heuristic examples, not comprehensive AI safety controls.
- **Dashboard:** a Flask and Flask-SocketIO interface for displaying and interacting with selected stage-model outputs.
- **Neural-network experiments:** PyTorch-oriented and other ML examples; the presence of model classes does not establish trained or useful model performance.
- **Biofeedback demonstration:** sample monitoring interfaces and data structures. No verified wearable integration or physiological validation is established here.
- **Logistics experiments:** prototype interfaces and scoring concepts. No validated freight-risk prediction or verified live commercial data integration is claimed.

The repository has accumulated multiple generations of experiments and documentation. Individual components may have different dependencies and maturity levels.

## Local dashboard setup

The dashboard requires Python and the dependencies in requirements.txt.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python web_dashboard/app.py
```

The dashboard binds to 127.0.0.1 by default, uses a process-generated secret unless LUMINARK_SECRET_KEY is supplied, and does not enable debug mode by default. For a separately hosted frontend, configure LUMINARK_CORS_ALLOWED_ORIGINS with an explicit comma-separated origin allowlist. Do not expose this prototype publicly without authentication, deployment hardening, and a security review.

Optional environment settings:
- LUMINARK_SECRET_KEY: stable secret for deployments that need sessions to persist across restarts.
- LUMINARK_HOST: defaults to 127.0.0.1.
- LUMINARK_PORT: defaults to 5000.
- LUMINARK_DEBUG: debug mode is enabled only when explicitly set to 1, true, or yes.
- LUMINARK_CORS_ALLOWED_ORIGINS: explicit comma-separated origin allowlist.

## Tests

The repository includes selected scenario-based tests. They are useful for checking that specific code paths produce expected outputs for the supplied examples. The test scenarios are hand-authored and small; reported percentages are scenario pass rates, not estimates of general-world accuracy or safety effectiveness.

Run the safety scenario tests with:

```bash
python test_safety_protocols.py
```

A passing test suite does not establish production readiness or independent validation.

## Known limitations requiring further work

- Repository-wide dependencies and installation paths have not yet been fully reconciled.
- Some modules are optional, experimental, or dependent on packages not required by the dashboard.
- The framework implementation contains legacy stage labels that need reconciliation with the current canonical 0-9-0 terminology.
- No independent benchmarks, prospective studies, clinical evaluations, or production customer results are established by this repository.
- Authentication, authorization, data retention, privacy, monitoring, and deployment controls require further engineering before networked use.
- Third-party dependency licenses and code provenance have not yet been fully inventoried.

## Development principles

1. Preserve the owner's conceptual framework and terminology.
2. Distinguish conceptual authorship from AI-generated implementation.
3. Keep experimental and synthetic results clearly labeled.
4. Do not claim accuracy, safety, efficacy, or production readiness without appropriate evidence.
5. Validate changes with repeatable tests and document unresolved issues.

## License and commercial use

Review the repository's license file and every third-party dependency before reuse, redistribution, licensing, or sale. This README does not grant rights beyond the applicable licenses or terms.