# Persona Decision Engine

**Goal:** learned, fast, uncertainty-aware decisions conditioned on an agent's synthetic or permissioned profile and evolving context.

## Interface
```
predict(persona, observation, internal_state, history) -> action_distribution, next_state
```

Formally: `P(action_t | profile, observation_t, state_t, history_t)`.

## Model direction
- Small transformer / MLP / recurrent policy with persona embeddings.
- Teacher-student distillation from stronger policies or validated behavioral data.
- Factorized output heads for actions and uncertainty.
- Sparse decisions and batching to avoid one full LLM call per agent per tick.
- Quantization and per-hardware compilation where supported.
- Dynamic per-agent state separate from immutable persona features.

The user's JEB-like decision-model reference is **to be confirmed**; keep the model adapter architecture independent of a particular named paper or model.

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
