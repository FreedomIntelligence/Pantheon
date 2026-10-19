# Architecture

## Mission
Build a **heterogeneous, crowdsourced multi-agent simulation infrastructure** in which autonomous agents are cheap to step, computation can be distributed across independently operated clusters, and results can be reproducible.

## Control plane
- **Scenario coordinator:** simulation manifest, seeds, agent schemas, timeline and progress.
- **Membership registry:** worker identity, capabilities, device memory, network metrics, declared policy.
- **Placement scheduler:** cost/latency-aware placement across CPU, NVIDIA CUDA GPU and Huawei Ascend NPU.
- **Work leases:** idempotency key, expiration, heartbeat, retries and workload constraints.
- **Verification and accounting:** signed task assignments, audit trail, accepted compute accounting.

## Data plane
- **Agent shards:** hash/range/social-graph placement with elastic rebalancing.
- **World-state partitioning:** state store for scenario, shard-local agent state, ghost caches for remote neighbors.
- **Event transport:** sequence numbers, replayable events, bounded queues, watermarking.
- **Persona decision service:** profile-conditioned inference; optional batching.
- **Checkpointing:** immutable snapshots, deterministic RNG streams and model version pinning.

## Execution backends

| Backend | Primary runtime | Current maturity |
| --- | --- | --- |
| x86 / ARM CPU | Python reference; future native/vectorized kernels | CPU-only toy simulator |
| NVIDIA GPU | CUDA (PyTorch / compiled kernels) | Planned |
| Huawei Ascend | CANN / torch-npu | Planned |
| Other accelerators | Backend interface / capabilities | Planned |

The scheduler should never assume that kernels and model formats are portable across vendors. Each backend advertises supported operators, model artifacts, available memory, and safety constraints.

## Simulation clock
Support synchronous ticks for repeatability and event-driven execution for sparse activity. Remote stragglers require deadline and determinism policies. Define *active agent* separately from *resident agent* and *decision throughput* separately from concurrent task counts.

## Evaluation
Report agent count, decisions per second, simulation-time / wall-clock ratio, cross-shard event rate, checkpoint recovery duration, dollars per million agent-decisions, energy use and per-device latency percentiles. Never imply billion-scale capability without reproducible results.

## Security boundaries
Crowd devices are not trusted infrastructure. Do not provide secrets or sensitive profiles to untrusted workers. Require isolated execution, authorized datasets, signed model artifacts, egress restrictions, bounded resources, and result verification.

## Implementation stages
1. Portable schemas and CPU baseline.
2. Accelerator adapters and batching.
3. Multiple trusted machines with checkpoint/replay.
4. Opt-in federation and auditable crowdsourcing.
5. Scale tests with explicit throughput and cost definitions.
