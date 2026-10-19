# Contributing

Welcome to PANTHEON! We are building an open infrastructure for heterogeneous and crowdsourced multi-agent simulation.

## Start here
1. Read [README](README.md), [architecture](docs/ARCHITECTURE.md), [protocol](docs/CROWD_PROTOCOL.md) and [roadmap](docs/ROADMAP.md).
2. Propose a concrete issue describing expected behavior, tests and supported devices.
3. Keep device-specific logic behind backend interfaces; preserve CPU fallback.
4. Include benchmark configuration and reproducibility information for performance claims.
5. Add tests; run `python -m pytest`.

## Focus areas
Distributed state and events, CUDA, Huawei Ascend, compact persona decision models, secure opt-in resource sharing, validation and benchmarks.

## Privacy and security
Do not introduce personally identifiable profiles into public examples or untrusted workers. No arbitrary code execution from unsolicited providers.

## Licensing
A project license and governance policy are still under discussion; do not assume a grant of open-source redistribution rights until a license is adopted.
