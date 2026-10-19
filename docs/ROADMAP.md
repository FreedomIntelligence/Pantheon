# Roadmap

PANTHEON targets billion-scale multi-agent simulation. The following are **milestones, not claims of completed capability**.

## Phase 0: Local reference / spec
- [x] CPU reference simulator and unit tests.
- [x] Initial architecture and draft protocol.
- [ ] Define `AgentState`, `Persona`, `Action`, `WorkerCapabilities` and lease schemas.
- [ ] Reproducible CPU throughput and memory baselines.

## Phase 1: Heterogeneous compute
- [ ] CUDA GPU inference adapter with explicit version matrix.
- [ ] Huawei Ascend NPU adapter (CANN and torch-npu compatibility).
- [ ] Capability-aware worker placement, optimized batch model inference.
- [ ] Cross-backend numerical and semantics tests.

## Phase 2: Distributed simulation
- [ ] Agent partitioning, distributed tick coordinator and event routing.
- [ ] Checkpoint/resume and deterministic replay.
- [ ] Chaos and recovery tests with trusted multi-node clusters.

## Phase 3: Crowdsourced protocol
- [ ] Identity, enrollment, capability discovery and resource policies.
- [ ] Signed artifacts, sandboxed tasks, leases and verification.
- [ ] Usage accounting, reputation, opt-in federated cluster deployments.

## Phase 4: Persona modeling
- [ ] Licensed/consented/synthetic dataset pipeline and action taxonomy.
- [ ] Small trained profile-conditioned policy model.
- [ ] Quality, calibration, subgroup analysis and p50/p95/p99 performance reports.
- [ ] Hardware-specific quantization and inference.

## Phase 5: Scale
- [ ] 1 million, then 10 million, then 100 million *well-defined active agents*.
- [ ] Feasibility study toward one billion; publish decisions/sec, latency, failure rate, energy and costs.
- [ ] Open independent benchmark and reproducibility suite.

## Contributions welcome
Simulation runtime, CUDA, Huawei Ascend, low-latency policy networks, systems security, distributed protocol design and benchmark reproducibility.
