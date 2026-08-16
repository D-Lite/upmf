# UPMF Prototype: Reduced-Scope Implementation Guide

This is the canonical build brief for the UPMF prototype supporting the dissertation *Design and Development of a Framework for Managing Unreliable Participation in Decentralized Machine Learning Systems*. The prototype is deliberately laptop-scale, deterministic, resumable, and suitable for unattended execution.

## Quick start

Python 3.12 is the tested version. The simulator selects CUDA, Apple MPS, or CPU automatically.

```bash
git clone <repository-url>
cd <repository-directory>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Run one experiment:

```bash
python scripts/run_single.py \
  --strategy fedavg_sync \
  --systems low \
  --stats low \
  --seed 1
```

Run or resume the complete 24-run matrix:

```bash
python scripts/run_matrix.py --config configs/experiment_matrix.yaml
python scripts/make_plots.py --results results
```

Use `--fresh` only when intentionally discarding an existing run's generated state. Checkpoints are written after every round, so rerunning the same command resumes incomplete work and skips validated completions.

## Reproducing the submitted evidence

The submitted Apple M2 run completed all 24 experiments in approximately 12 minutes. Runtime depends on the selected device and host.

- [`RESULTS.md`](RESULTS.md) gives the plain-language finding.
- [`results/summary.csv`](results/summary.csv) contains the 24 final run summaries.
- [`results/plots/`](results/plots/) contains the four submitted figures.

Reference outputs are versioned for review. Running the matrix may replace them, so use a separate branch or copy if the submitted evidence must remain unchanged.

## Research goal

Compare three coordination strategies under interacting systems and statistical heterogeneity:

1. **UPMF**: adaptive scheduling plus staleness- and divergence-aware aggregation.
2. **Synchronous FedAvg**: waits for sampled clients and drops timed-out work.
3. **Async-VC**: accepts updates within a virtual-time budget and weights them equally.

All strategies use the same MNIST partitions, model, client trainer, optimizer settings, seeded client profiles, and accounting rules. The central question is whether UPMF's advantage over the better baseline grows toward the high-systems/high-statistical-severity condition. Negative or mixed evidence must be reported honestly.

## Scope

- Python 3.10+, PyTorch, torchvision, NumPy, pandas, matplotlib, PyYAML, pytest.
- In-process simulation only; no real networking, queues, containers, or deliberate sleeping.
- MNIST, 20 clients, standard test split.
- Dirichlet partitions: alpha `100` (near-IID) and `0.2` (severe non-IID).
- Systems severity: `low` and `high`, configured in YAML.
- Exactly 24 production runs: 3 strategies × 2 systems levels × 2 statistical levels × seeds `[1, 2]`.
- Two core built components: Adaptive Round Scheduler and Staleness-Tolerant Aggregator.
- Participant profiling is a small EWMA state helper; top-k compression is a short standard utility.

## Repository structure

```text
configs/
  base.yaml
  experiment_matrix.yaml
upmf/
  __init__.py
  client.py
  coordinator_base.py
  data.py
  experiment_matrix.py
  heterogeneity.py
  metrics.py
  model.py
  runner.py
  strategies/
    __init__.py
    async_vc.py
    fedavg_sync.py
    upmf.py
  upmf_components/
    __init__.py
    aggregator.py
    compressor.py
    profiler.py
    scheduler.py
scripts/
  make_plots.py
  run_matrix.py
  run_single.py
tests/
results/
  checkpoints/
```

Downloaded data, checkpoints, per-run artifacts, raw logs, and private dissertation evidence are excluded from Git. The final combined summary and four reference plots are versioned for review.

## Shared learning path

`partition_dirichlet(dataset, num_clients, alpha, seed) -> list[Subset]` partitions labels reproducibly. Tests must show that alpha `0.2` yields more client-level class skew than alpha `100`.

The model is a small CNN with 16- and 32-filter 3×3 convolutions, ReLU and max-pooling, followed by one linear classifier. A single client-training function returns a model delta and completed-step metadata; every strategy must call it unchanged.

## Virtual systems heterogeneity

Each client has a seeded fixed profile:

- `compute_speed_multiplier`
- per-participation availability probability
- network-delay distribution

Low severity uses a light log-normal speed spread, low dropout, and short delay. High severity uses a heavy-tailed spread, higher dropout, and longer delay. Exact values live in `configs/base.yaml`. These are plausible simulation presets, not estimates fitted to Nigerian grid or broadband microdata.

Simulation time is event-driven. It must never call `sleep`; real runtime is recorded separately only as an operational diagnostic.

## Strategies

### Synchronous FedAvg

Sample a configured client fraction, train available clients, and aggregate updates arriving before the timeout by local sample count. Completed work from updates dropped after the timeout is wasted compute. Unavailable clients perform no work and are not wasted compute.

### Async-VC

Process client updates by virtual arrival time within each configured round budget. Updates are equally weighted and receive no staleness or divergence correction. Pending arrivals may cross budget boundaries and remain eligible until the baseline's configured cutoff.

### UPMF

The profiler stores EWMA latency, completed steps, availability, and cosine divergence.

The Scheduler biases invitations toward reliable clients while preserving a configurable exploration floor so slow or unreliable clients are never permanently excluded. Assigned local steps scale with profiled compute capacity and remain within configured bounds. The exact selection score and allocation formula must be documented in code.

The Aggregator weights accepted update `i` as:

```text
raw_weight_i = sample_weight_i
             * exp(-staleness_rate * staleness_i)
             / (1 + divergence_rate * max(0, divergence_i))
```

Weights are normalized across accepted updates. Updates beyond the hard staleness cutoff are dropped and their completed steps count as wasted. Formula parameters live in YAML.

The compressor performs per-tensor top-k magnitude sparsification. Communication accounting includes sparse values and indices. UPMF compresses uplink deltas only; all strategies count dense model downlink bytes per invited client.

## Metrics

Record every round and evaluate accuracy every two rounds:

- held-out test accuracy;
- total and wasted completed local steps;
- wasted-compute fraction;
- cumulative uplink and downlink bytes;
- simulated time and real execution time;
- first round and simulated time reaching the provisional 90% target.

Each run writes an evaluation CSV and final summary JSON containing the full resolved configuration and provenance.

## Checkpointing and unattended execution

After every completed round, atomically write:

- `results/checkpoints/<run_id>.pt`: model, strategy/profiler state, pending events, virtual clock, RNG states, accumulated metrics, and next round.
- `results/checkpoints/<run_id>.json`: human-readable status, configuration hash, timestamps, last completed round, and failure/completion details.

Write temporary files and rename only after a successful flush. Resume only when the checkpoint configuration hash matches. `--fresh` intentionally removes that run's prior generated state.

`nohup` keeps a process alive after terminal disconnection, but cannot keep a local process running while the laptop is asleep or powered off. Checkpointing enables safe restart after wake, power recovery, notebook reconnection, or provider resubmission; it cannot prevent cloud providers from reclaiming a runtime.

The matrix runner executes sequentially by default, skips validated completions, resumes incomplete runs, isolates per-run failures, and rebuilds `results/summary.csv` deterministically. It logs timestamps and run IDs. A failed run does not erase other progress, but the command returns a nonzero final status if any matrix entry remains failed.

## Production matrix

```yaml
strategies: [upmf, fedavg_sync, async_vc]
systems_severities: [low, high]
statistical_severities:
  low: 100.0
  high: 0.2
seeds: [1, 2]
```

## Required outputs

`scripts/make_plots.py` generates:

1. Accuracy versus round at high/high severity.
2. A 2×2 grid of UPMF final-accuracy advantage over the better baseline.
3. Wasted compute by strategy at high systems severity.
4. Communication cost by strategy.

`RESULTS.md` states whether the evidence supports, weakens, or qualifies the interaction claim and notes limitations from two seeds and synthetic systems presets.

## Build and validation order

1. Foundation and deterministic contracts.
2. Data, model, and shared client trainer.
3. Virtual heterogeneity.
4. Metrics, event state, and atomic checkpoints.
5. FedAvg and Async-VC.
6. A learning FedAvg MNIST pilot plus interrupted/resumed equivalence.
7. Profiler, Scheduler, Aggregator, and compressor.
8. UPMF composition and fairness checks.
9. Resilient 24-run orchestration.
10. Plots and `RESULTS.md`.

The prototype is complete when tests pass, all 24 unique runs complete or resume safely, summary and plots exist, and the reported conclusion follows the observed data rather than the expected thesis claim.
