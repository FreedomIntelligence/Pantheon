# Persona Decision Engine

**Goal:** learned, fast, uncertainty-aware decisions conditioned on an agent's synthetic or permissioned profile and evolving context.

## Interface
```
predict(identity, personality, messages, environment, internal_state, task) -> action_distribution, next_state
```

Formally: `P(action_t | identity, personality, messages_t, environment_t, state_t, task)`.

## Model direction
- Small transformer / MLP / recurrent policy with persona embeddings.
- Teacher-student distillation from stronger policies or validated behavioral data.
- Factorized output heads for actions and uncertainty.
- Sparse decisions and batching to avoid one full LLM call per agent per tick.
- Quantization and per-hardware compilation where supported.
- Dynamic per-agent state separate from immutable persona features.

The decision model is the **JEV model**: a compact persona-conditioned policy whose inputs are identity + personality, interaction/communication context (peer decisions, aggregated environment information) and the current task, and whose output is an action together with its probability. Keep the model adapter architecture independent of a particular named paper or model.

## Latency and quality targets
**Aspirational target: p95 < 50 ms** for a clearly specified batch size, action space, context length, model revision and device. This is *not* a measured SLA and not the latency for advancing an entire billion-agent simulation tick.

Metrics:
- p50/p95/p99 decision latency and batch throughput across x86, ARM, CUDA and Ascend.
- Calibration (ECE/Brier score), action likelihood and trajectory fidelity.
- Long-term population-level dynamics and distributional agreement.
- Model memory usage, power, total simulation cost and fairness across population subgroups.

## Data practices
Prefer synthetic profiles or transparently consented/de-identified sources. Do not claim to predict specific identifiable humans accurately. Evaluate for representativeness, privacy, systematic error and misuse risks. Keep private profile records on trusted infrastructure only.

## Milestones
1. Synthetic-data baseline with deterministic policy.
2. Curated training and evaluation schemas.
3. Teacher/student model training and calibrated benchmarking.
4. CUDA and Ascend optimized inference.
5. Batched integration into distributed simulation.
