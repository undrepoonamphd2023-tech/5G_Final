# HERO: Hybrid Efficient Resource Offloading for 5G Edge–Fog–Cloud Networks

**LSH-GRL-HPFOO: Locality-Sensitive Hashing + Graph Reinforcement Learning + Hybrid Pufferfish–Osprey Optimization**

Final implementation (V5): results, ablations, robustness studies and reproducibility notes.

- **Research Scholar:** Poonam Arjun Undre (PRN 23079001137)
- **Research Guide:** Dr. Kanhaiya Sharma
- **Institution:** Symbiosis Institute of Technology, Symbiosis International (Deemed University), Pune

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Problem Setting](#problem-setting)
4. [How HERO Works](#how-hero-works)
5. [Experimental Setup](#experimental-setup)
6. [Detailed Results](#detailed-results)
7. [Known Limitations and Honest Findings](#known-limitations-and-honest-findings)
8. [Validity and Reproducibility](#validity-and-reproducibility)
9. [Repository Contents](#repository-contents)
10. [Hyperparameters and Environment](#hyperparameters-and-environment)
11. [Future Work](#future-work)
12. [Citation](#citation)

---

## Overview

HERO is an online task-offloading decision system for 5G heterogeneous networks. Each time a device produces a computing job, HERO decides **where it should run** among 19 machines:

| Tier | Count | Characteristics |
|------|-------|-----------------|
| Device (local) | 1 | No transmission or cost, but slow and drains battery |
| Edge servers | 10 | Close and cheap, individually weak, easily overloaded |
| Fog servers | 5 | More powerful, slightly farther, moderately priced |
| Cloud servers | 3 | Very powerful, 40–70 ms round trip, most expensive |

Decisions are made one job at a time, without knowing future jobs, and must balance four goals: **meeting the deadline, low latency, device battery, and infrastructure cost**.

Every decision is scored by a bounded reward in [0, 1]:

```
r = 0.35·1[ℓ ≤ D] + 0.15·(1 − ℓ/D)+ + 0.15·(1 − E/E_local)+ + 0.35·(1 − C/C_cloud)+
```

where ℓ is latency, D the deadline, E the device energy and C the infrastructure cost.

### What V5 adds

V5 is the final version prepared for the major revision. It adds fairness measurement, a max-min fair baseline, GA/PSO/GWO refiners and a GAT-PPO baseline, a unified 10-seed run with Holm-corrected statistics, LSH timing at realistic memory size, multi-seed robustness tests, an unseen-workload-source test, a scalability study (19–145 machines), and full cost reporting. It also fixes a defect where V4 passed the seed-0 network graph to every seed's agent (effect on seeds 1–9: 0.006 reward on average; conclusions unchanged).

---

## Key Results

All numbers are from the final run (`results_V5`, Colab GPU, PyTorch 2.11), ten independent seeds, evaluated on 2,000 held-out test jobs.

| Result | Value |
|--------|-------|
| Mean reward (10 seeds) | **0.8736 ± 0.0045**, highest of all practical methods |
| Significance | All 10 head-to-head comparisons significant after Holm correction |
| Gap closed (greedy → oracle) | **85.6%** |
| Gap closed (max-min → oracle) | **77.4%** |
| vs. GAT-PPO (graph-attention PPO) | **+4.7%**, 10/10 seeds |
| vs. max-min fair | **+3.3%**, 10/10 seeds |
| vs. latency-greedy | **+5.9%**, 10/10 seeds |
| Infrastructure cost ratio | **0.145** vs 0.316 for greedy (54% lower) |
| Edge-server load balance (Jain) | **0.81** vs 0.18 (greedy) and 0.46 (max-min) |
| Deadline-miss rate | 6.8% |
| Search-space reduction | 3.48 of 19 machines searched (81.7% fewer) |
| Decision time | ≈ 2.3 ms, constant from 19 to 145 machines |
| Quality at scale | 86–88% gap closed from 19 to 145 machines |
| Learning speed | Final quality reached after ~1,500 training decisions |

### Ten-seed comparison (HERO vs. each method, Holm-corrected)

| Method | Mean reward | s.d. | HERO ahead by | Holm p | Cohen's dz | Wins |
|--------|-------------|------|---------------|--------|-----------|------|
| **HERO (proposed)** | **0.8736** | 0.0045 | — | — | — | — |
| LSH-GRL (no HPFOO) | 0.8699 | 0.0050 | +0.42% | 0.0076 | 1.22 | 8/10 |
| GRL only | 0.8694 | 0.0044 | +0.48% | 0.0040 | 1.45 | 9/10 |
| No-graph scoring | 0.8692 | 0.0070 | +0.50% | 0.0382 | 0.77 | 8/10 |
| Max-min fair | 0.8454 | 0.0083 | +3.33% | <0.0001 | 3.47 | 10/10 |
| GAT-PPO | 0.8343 | 0.0120 | +4.71% | <0.0001 | 3.67 | 10/10 |
| Greedy (min-latency) | 0.8248 | 0.0089 | +5.92% | <0.0001 | 5.84 | 10/10 |
| Standard DQN | 0.7824 | 0.0513 | +11.66% | 0.0014 | 1.77 | 10/10 |
| Random | 0.7216 | 0.0095 | +21.06% | <0.0001 | 16.91 | 10/10 |
| Local-only | 0.6365 | 0.0000 | +37.24% | <0.0001 | 52.74 | 10/10 |
| All-to-cloud | 0.5145 | 0.0135 | +69.80% | <0.0001 | 28.67 | 10/10 |
| Myopic oracle (upper bound) | 0.8818 | 0.0039 | −0.93% | — | — | 0/10 |

---

## Problem Setting

No single tier is best for every job: tiny jobs run fastest on the device, heavy computations on fog servers, and data-heavy jobs with little computation are limited by the radio. Servers also have queues, so a choice that looks best in isolation can hurt later jobs. The reward is deadline-seeking: it changes smoothly with latency below the deadline but drops by 0.35 the moment a deadline is missed. HERO is therefore a **cost-aware, deadline-seeking multi-objective** offloader, not a latency minimiser.

---

## How HERO Works

```
Job + live network state
        │
        ▼
 GRL (judge)  ──►  Top-3 shortlist  ──►  HPFOO (tie-breaker)  ──►  Chosen machine
 scores all 19            ▲               re-ranks ≈3.5 candidates
                          │
                  LSH (memory) adds ≤ 1 candidate
                  from a similar past job

 Reward (Eq. 1) trains GRL and fits HPFOO's utility
```

**GRL (Graph Reinforcement Learning).** A two-layer graph-attention encoder (4 heads) feeds a Double DQN that scores each machine separately. It reads machine capacity, queue, distance and tier together with job attributes, and learns purely from the reward.

**LSH (Locality-Sensitive Hashing memory).** Jobs are fingerprinted so similar jobs share a bucket. A retrieved suggestion is used only if verified (cosine similarity ≥ 0.99, at least 2 matches), and only above-average-reward decisions are stored. The memory guides exploration in training and may add one machine to the shortlist at decision time.

**HPFOO (Hybrid Pufferfish–Osprey Optimization).** A population of 14 candidate preference profiles (six weights covering queueing delay, processing time, distance, price, deadline pressure and staying local). The *osprey* move shifts each candidate toward a better-performing peer (exploration); the *pufferfish* move applies a shrinking random perturbation (refinement). A change is kept only if it improves the candidate. HPFOO has no access to simulator formulas and is fitted only from rewards received over the last 2,000 decisions. Its utility is blended with the GRL score using weight α, selected on validation data per seed.

---

## Experimental Setup

### Data

Four real measurement traces (2,000 jobs each, 8,000 total) plus a real cell-site location list:

| Trace | Origin | Contributes |
|-------|--------|-------------|
| Task-offloading (cellular) | Cellular traffic records | Traffic volume, radio type, session duration |
| VEC edge data | Vehicular edge computing | Execution time, mobility status |
| Azure LLM inference | Azure LLM inference logs | Token counts (computational demand) |
| Borg traces | Google Borg cluster | Job duration, memory, priority |
| `site.csv` | Real cell-site locations | Geographic spread of edge servers |

A fixed, documented mapping converts trace fields into CPU cycles and deadlines. About 30% of jobs are latency-critical (URLLC-like), 70% relaxed (eMBB-like). Arrivals are Poisson with a rate giving ≈ 90% edge utilisation. The notebook stops rather than substituting synthetic data if any file fails to load.

> **Note:** Exact dataset names, versions, links and licences should be listed here before publishing, e.g. `TODO: add dataset links and licences`.

### Fair-testing practice

- **Three-way split:** 60% train / 15% validation / 25% test (2,000 unseen jobs). All tuning uses validation data only.
- **Frozen evaluation:** learning off, no exploration.
- **Ten paired seeds:** each seed changes server capacities, distances, links, network initialisation and exploration; all methods share each seed.
- **Statistics:** paired t-tests, 95% CIs, Cohen's dz, seeds won, Holm correction within each comparison family.
- **Two gates before any learning:** Gate 1 (seven physical sanity checks, all passed); Gate 2 (≥ 5% headroom between best classic rule and oracle, measured 7.52%).

### Compared methods

HERO, LSH-GRL (no HPFOO), GRL only, per-node scoring without graph, Standard DQN, GAT-PPO (GAPO family, re-implemented), LSH-GRL with GA / PSO / GWO refiners (equal 182-evaluation budget), Greedy (min-latency), Max-min fair, Random, Local-only, All-to-cloud, and the Myopic oracle (upper bound, not a real system).

---

## Detailed Results

### Component contributions (ablation, 10 seeds)

| Comparison | Change | 95% CI of diff. | Holm p | Wins | Reading |
|------------|--------|-----------------|--------|------|---------|
| Seeing network state (no-graph vs DQN) | +11.10% | [0.052, 0.121] | 0.0012 | 10/10 | Largest single effect; misses fall 9.95 points |
| HPFOO refinement (HERO vs LSH-GRL) | +0.42% | [0.0015, 0.0058] | 0.011 | 8/10 | Small, consistent; misses fall 0.71 points |
| LSH memory (LSH-GRL vs GRL) | +0.06% | [−0.0009, 0.0020] | 0.82 | 7/10 | No measurable reward effect |
| Graph layer (GRL vs no-graph) | +0.02% | [−0.0045, 0.0048] | 0.94 | 4/10 | No measurable reward effect |
| Learning vs best classic rule (GRL vs greedy) | +5.41% | [0.039, 0.051] | <0.0001 | 10/10 | Learning pays off |

### Fairness and equity (10-seed means)

| Measure | HERO | Greedy | Max-min | GAT-PPO | Oracle |
|---------|------|--------|---------|---------|--------|
| Jain index, tenant success | 0.9975 | 0.9986 | 0.9986 | 0.9892 | 0.9986 |
| Worst-tenant miss rate | 12.6% | 9.9% | 9.8% | 23.9% | 9.6% |
| Latency 95th percentile (ms) | 372 | 130 | 256 | 347 | 370 |
| Latency 99th percentile (ms) | 649 | 480 | 514 | 653 | 578 |
| Latency-critical (URLLC) miss rate | 21.7% | 18.1% | 17.7% | 37.7% | 17.5% |
| High-priority miss rate | 3.2% | 2.2% | 2.2% | 6.2% | 2.0% |
| Jain index, all-server load | 0.567 | 0.240 | 0.361 | 0.474 | 0.502 |
| Jain index, edge-server load | 0.814 | 0.177 | 0.461 | 0.979 | 0.810 |
| Cost ratio | 0.145 | 0.316 | 0.252 | 0.183 | 0.128 |

HERO balances **infrastructure load** far better than the rule-based baselines, at under half of greedy's cost. For **users**, the rule-based policies are slightly more protective of latency-critical and high-priority jobs. Fairness is therefore reported as a measured property with one strength (load balance) and one limitation (urgent-job protection).

### Refiner comparison: HPFOO vs GA, PSO, GWO (equal 182-evaluation budget)

| Refiner | Reward | s.d. | Miss rate | Cost ratio | ms per call | Holm p vs HPFOO |
|---------|--------|------|-----------|-----------|-------------|-----------------|
| HPFOO (proposed) | 0.8736 | 0.0045 | 6.76% | 0.145 | 19.2 | — |
| GA | 0.8735 | 0.0042 | 6.89% | 0.143 | 23.8 | 1.00 |
| PSO | 0.8728 | 0.0051 | 7.03% | 0.145 | 16.7 | 1.00 |
| GWO | 0.8727 | 0.0058 | 6.83% | 0.148 | 17.0 | 1.00 |
| No refiner (LSH-GRL) | 0.8699 | 0.0050 | 7.47% | 0.148 | — | 0.015 |

HPFOO ranks first numerically, but differences among the four refiners are within seed-to-seed noise. The gain comes from having a refinement stage.

### Robustness

- **Mobility (4 levels):** HERO beats greedy at every level (by 0.037–0.049, Holm p < 0.001) and beats GRL-only at every level (by 0.004–0.005, Holm p ≤ 0.039).
- **Perturbed conditions (15):** HERO beats greedy in all 15 (Holm p < 0.01) and stays ahead of max-min. Conditions: deadline multiplier 0.6–1.6, arrival-gap multiplier 0.4–2.0, radio offset −8 to +4 dB.
- **Heavy load:** at 2× arrival rate HERO scores above the oracle on 9/10 seeds (0.8544 vs 0.8415). This is a consistent trend but **not significant after Holm correction** (p = 0.051), so it should not be claimed as established.
- **Arrival patterns (Poisson, periodic, bursty, heavy-tailed Pareto):** reward varies by < 0.002 (0.8717–0.8741); HERO beats greedy on all 10 seeds in every pattern and closes 85–88% of the gap.

### Generalisation to an unseen workload source (train on 3 traces, test on the 4th; 5 seeds)

| Held-out source | HERO | GRL | GAT-PPO | DQN | Greedy | Max-min | Oracle | Headroom |
|-----------------|------|-----|---------|-----|--------|---------|--------|----------|
| Cellular | 0.8329 | 0.8295 | 0.8149 | 0.7422 | 0.8386 | 0.8510 | 0.8601 | 2.6% |
| Vehicular | 0.8241 | 0.8148 | 0.8001 | 0.6613 | 0.6886 | 0.8220 | 0.8460 | 22.9% |
| Azure LLM | 0.9409 | 0.9361 | 0.8858 | 0.8932 | 0.9458 | 0.9407 | 0.9475 | 0.2% |
| Borg | 0.6766 | 0.6704 | 0.6196 | 0.3386 | 0.5834 | 0.7174 | 0.7352 | 26.0% |

HERO is the best *learned* method on every unseen source. Where there is real headroom (vehicular, Borg), it beats greedy on all five seeds. Where greedy is already near-optimal (cellular, Azure), HERO finishes marginally behind, and max-min is strongest on unseen cellular and Borg traces.

### Scalability (mean over 3 seeds)

| Machines | Decision (ms) | 95th pct (ms) | Training (s) | GPU memory (MB) | Reward | Gap closed |
|----------|---------------|---------------|--------------|-----------------|--------|------------|
| 19 | 2.30 | 2.55 | 50.7 | 33 | 0.8711 | 86.8% |
| 37 | 2.31 | 2.56 | 51.3 | 57 | 0.8780 | 87.0% |
| 73 | 2.32 | 2.59 | 52.6 | 141 | 0.8791 | 86.2% |
| 145 | 2.31 | 2.58 | 99.2 | 448 | 0.8801 | 88.3% |

The model has 24,321 parameters regardless of network size. Throughput is ≈ 410 jobs/s; training time scales linearly with training decisions (fitted exponent 1.03).

### LSH memory

| Measure | Value |
|---------|-------|
| Recall@10 (controlled scaling test) | 94.8% while examining 98.3% fewer stored jobs |
| Speed-up vs exhaustive search | 2.6× at 140,000 entries (break-even ≈ 20,000 entries) |
| Memory entries in the running pipeline | 3,469 |
| LSH lookup vs full scan (in pipeline) | 0.64 ms vs 0.34 ms |
| Decision time with / without LSH candidate | 2.31 / 1.59 ms |
| Reward with / without LSH candidate | 0.8736 / 0.8730 (p = 0.12) |

> Entries beyond the 4,800 real training jobs in the scaling test are perturbed copies of real jobs; this test measures query-time scaling, not data volume.

### Design choices and sensitivity

- **Shortlist size:** 3 machines performs as well as all 19, with 81.7% less optimiser search.
- **Blend weight α** (validated per seed): 0.25 on six seeds, 0.5 on two, 0 on one, 1 on one.
- **Reward weightings:** HERO ranks first under 5 of 7 weightings; the two reversals occur only where cost weight ≤ 0.10. Entropy-derived weights were numerically unstable on this data and were not adopted.
- **Pufferfish step-size schedule:** not a sensitive setting (rewards 0.8727, 0.8724, 0.8715).

---

## Known Limitations and Honest Findings

The following should be stated plainly in any paper or derived work.

- HERO is **not** faster or more energy-efficient than latency-greedy. Its advantage is the multi-objective balance, mainly through cost.
- HERO misses slightly more deadlines than greedy/max-min for latency-critical and high-priority jobs and for some tenants. The framework is therefore **not** claimed to be equity-aware or fairer to users than the baselines.
- HPFOO performs on par with GA, PSO and GWO; no significant difference was found.
- The LSH memory and the graph-attention layer show no measurable reward gain on their own at the studied 19-machine scale.
- On unseen workloads, max-min fair is stronger on two of four sources.
- Simulation only; no physical testbed.
- Single-cell radio model (no inter-cell interference, handover or multi-user contention).
- Jobs are independent; task graphs are not modelled.
- Trace fields are mapped to cycles and deadlines by stated assumptions; arrivals and deadlines are generated and calibrated.
- The learned policy offloads nearly every job, so on-device execution is rarely exercised.
- No security or privacy layer.

---

## Validity and Reproducibility

Built-in checks of the final run:

| Check | Result |
|-------|--------|
| Gate 1: seven physical sanity checks | Pass (7/7) |
| Gate 2: headroom above greedy | Pass (7.52%) |
| Deterministic evaluation (replay reproduces test reward) | Pass (difference 0) |
| Local-only identical on every seed | Pass |
| Rewards in [0, 1]; Jain's indices in (0, 1] | Pass |
| Network size restored after scalability study | Pass |
| Equal evaluation budget for all refiners | Pass (182 per call) |
| All unseen-source runs complete | Pass (20/20) |
| Oracle ahead of HERO at nominal load | Pass (10/10 seeds) |
| Agreement with V4 | 51/68 (expected, see below) |

All 50 values that do not involve neural-network training (greedy, random, local-only, all-to-cloud, oracle on every seed) match V4 exactly, confirming the data and simulated network are identical. Trained methods differ slightly because V5 ran on a newer PyTorch build (2.11, CUDA 13.0) and different GPU hardware. Results are reproducible on the same software and hardware; across platforms, individual-seed values vary within normal seed noise.

**Run time (Colab GPU):** ten-seed study 63 min, refiner study 35 min, unseen-source study 54 min, scalability study 17 min.

---

## Repository Contents

> Adjust this section to match your actual repository layout.

```
.
├── <notebook>.ipynb                      # Full implementation: all 37 cells, single end-to-end run
├── results_V5/
│   ├── results_all.json                  # All results plus the "config" block (hyperparameters, versions, seeds, hardware)
│   ├── table_main_comparison.csv         # Seed-0 breakdown with cost ratio
│   ├── table4_multiseed_tests_holm.csv   # Ten-seed tests with Holm p, CI, dz, wins
│   ├── table_component_contributions.csv # Ablation with CIs
│   ├── table_fairness*.csv               # Fairness measures
│   ├── table_refiners.csv                # HPFOO vs GA / PSO / GWO
│   ├── table_loso_generalisation.csv     # Unseen-source results
│   ├── table_scalability_*.csv           # Network size and task-count scaling
│   ├── table_arrival_patterns.csv        # Arrival-pattern robustness
│   ├── table_lsh_scalability.csv         # LSH retrieval scaling
│   ├── table_lsh_inference.csv           # LSH in-pipeline cost
│   ├── table_weight_sensitivity.csv      # Reward-weighting sensitivity
│   ├── table_pufferfish_schedule.csv     # Step-size schedules
│   ├── table_param_sensitivity_ofat.csv  # Perturbed conditions
│   ├── table_per_seed_all_methods.csv    # Per-seed results for every method
│   └── Figure_*.png / output*.png        # Publication and diagnostic figures
└── data/                                 # Workload traces and site.csv (see Data section)
```

**Figure style note:** Figures 3, 9 and 12–21 follow the publication style (serif font, bold axis titles, legend below, no title, 1000 DPI). Figures 4–8, 10 and 11 come from the notebook's general plotting cell and carry an in-image title line; crop it if needed.

### Running

1. Open the notebook in Google Colab with a GPU runtime.
2. Place the four workload traces and `site.csv` where the notebook expects them (the run stops if any file fails to load).
3. Run all cells. Finished seeds are cached to Google Drive, so a disconnection does not lose completed work.
4. Outputs are written to `results_V5/`.

---

## Hyperparameters and Environment

| Item | Value |
|------|-------|
| Network | 1 device, 10 edge, 5 fog, 3 cloud (19 machines); 100 MHz uplink; SNR 25/18/11/5 dB by mobility |
| Workload | 8,000 jobs (4 × 2,000); split 4,800 / 1,200 / 2,000; mean gap 11.9 ms; deadline scale 2.565 |
| Reward weights | 0.35 deadline, 0.15 latency, 0.15 energy, 0.35 cost |
| GRL | 2 graph-attention layers, 4 heads, 32-dim embedding, 128-unit MLP; Double DQN, γ = 0.95, lr 3×10⁻⁴, batch 128, replay 20,000, target update 200 |
| Training | 25 episodes × 250 decisions; exploration 1.0 → 0.05 over 60%; validation checkpoint every 5 episodes |
| LSH | 4 tables, 12 bits, cosine ≥ 0.99, ≥ 2 matches, bucket cap 40, 4 load buckets |
| HPFOO | population 14, 6 iterations (182 evaluations per call), bounds [−3, 3], every 250 decisions, 2,000-decision fitness window |
| Shortlist / α | top 3 (validated); α ∈ {0, 0.25, 0.5, 0.75, 1} validated per seed |
| GAT-PPO | same encoder; GAE γ = λ = 0.95, clip 0.2, 4 epochs, minibatch 64, entropy 0.01 |
| Seeds | 10 (main, refiners); 5 (unseen source, step size); 3 (scalability) |
| Platform | Google Colab GPU, PyTorch 2.11.0 (CUDA 13.0), Python 3.13, SciPy 1.16.3 |

---

## Future Work

- Fairness-constrained reward (penalising urgent-job misses or the tenant miss gap)
- Interference-aware multi-cell radio model with handover
- Dependency-aware offloading for task graphs
- Online learning under workload drift
- Larger deployments where the LSH memory reaches its efficient range
- Prototype testbed with real telemetry

---

## Citation

If you use this work, please cite it as:

```bibtex
@misc{undre2026hero,
  author = {Undre, Poonam Arjun and Sharma, Kanhaiya},
  title  = {HERO: A Hybrid Efficient Resource Offloading Framework for 5G Edge--Fog--Cloud Networks Using Locality-Sensitive Hashing, Graph Reinforcement Learning and Hybrid Pufferfish--Osprey Optimization},
  year   = {2026},
  note   = {Symbiosis Institute of Technology, Symbiosis International (Deemed University), Pune}
}
```

*Replace with the final publication reference once available.*
