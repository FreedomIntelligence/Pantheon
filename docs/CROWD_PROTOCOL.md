# Crowdsourced Compute Protocol (design draft)

**Status:** proposal, not a deployed protocol.

## Goals
Combine spare compute from independent providers and clusters, including CPUs, NVIDIA GPUs and Huawei Ascend NPUs, without assuming uniform accelerator software stacks. Protect resource providers, data owners and experiment integrity.

## Actors
- **Coordinator:** advertises simulation workload and issues short-lived leases.
- **Worker:** opt-in node agent declaring capabilities, capacity, trust status and policy constraints.
- **Verifier:** independently validates reported work.
- **Artifact store:** signed model and scenario artifacts, immutable input/output hashes.
- **Accountant:** records accepted work and resource use (not a payment implementation).

## Proposed task lifecycle
1. Worker enrolls, authenticates, accepts participation policy.
2. Worker publishes `DeviceCapabilities`: architecture, accelerator family, software stack, memory, available slots and data locality.
3. Coordinator submits `WorkOffer`: scenario ID, shard/time window, content hashes, model revision, estimated resources and data classification.
4. Placement matches capability, region, permissions, trust tier, price/energy budget and deadline.
5. Coordinator grants `Lease` with unique idempotency key and timeout.
6. Worker downloads **verified/signed** permitted artifacts into sandbox and executes.
7. Worker reports checkpoints, metrics and signed `ResultReceipt`.
8. Verifier accepts, audits or rejects; accountant records accepted work.
9. Coordinator applies events at deterministic boundaries; expired leases are retried safely.

## Example conceptual contract
```json
{
  "protocol_version": "0.1-draft",
  "work_id": "scenario-42:shard-7:tick-100",
  "model_artifact_sha256": "<digest>",
  "input_artifact_sha256": "<digest>",
  "required_backend": "cpu|cuda|ascend",
  "max_memory_bytes": 2147483648,
  "lease_seconds": 30,
  "data_classification": "public_synthetic"
}
```

## Non-negotiable controls
- Signed artifacts; no arbitrary remote code from unverified publishers.
- Sandboxed, resource-limited jobs; deny filesystem and network privileges by default.
- Opt-in provider registration and revocation.
- Sensitive real-person profiles must not be sent to untrusted volunteer machines.
- Duplicate execution and random audits; tolerance for nondeterministic floating-point output.
- Heartbeat, renewal, retries, deduplication and replay safety.
- Fraud-resistant measurement if incentives are introduced later.

## Open questions
Define threat model, identity and trust tiers, verification policy, data locality and licensing, incentive governance, interoperable operator sets and cross-vendor reproducibility.
