# Mi3\_UCM\_AV2\_SM: Fault‑Tolerant, Spatially‑Aware Scenario Mining for **Argoverse 2**

> **Technical Report — Argoverse 2 Scenario Mining Challenge, 2025**

---

## Abstract

Scenario mining from large autonomous‑driving datasets is critical for validating self‑driving systems. **This repository is a direct derivative of the open‑source [RefAV](https://github.com/CainanD/RefAV) framework and constitutes our official submission to the Argoverse 2 Scenario Mining Challenge.** **RefAV** pioneered the use of Large Language Models (LLMs) to turn natural‑language queries into executable code but still suffers from runtime failures and semantic misinterpretation of multi‑object spatial relations. We introduce two enhancements:

1. **Fault‑Tolerant Iterative Code Generation (FT‑ICG)** — automatically re‑prompts the LLM with error traces until syntactically and logically correct code is produced.
2. **Enhanced Prompting for Spatial Relational Functions (EP‑SRF)** — clarifies argument semantics for functions that encode relative direction, heading, or interaction.

Evaluated on the Argoverse 2 validation set with **Qwen‑2.5‑VL‑7B**, **Gemini 2.5 Flash**, and **Gemini 2.5 Pro**, the combined system delivers consistent gains, culminating in a **52.37 HOTA‑Temporal** on the hidden test set (Gemini 2.5 Pro). These results establish a new state‑of‑the‑art for reliable, high‑precision scenario mining.


## 1. Introduction

The **Argoverse 2 Scenario Mining Challenge** provides 10 k natural‑language queries that describe safety‑critical or uncommon traffic situations. **RefAV** tackles this with a two‑phase pipeline: (a) an LLM converts each query into *atomic function* calls; (b) the functions execute on sensor logs to retrieve matching time windows.

While flexible, RefAV exhibits two pain points:

* **Runtime errors** — LLM‑generated code often fails due to missing imports, wrong argument counts, or logic bugs.
* **Spatial‑relation confusion** — functions like `has_objects_in_relative_direction()` or `facing_toward()` require correct assignment of *track* vs *related* candidates; mis‑ordering silently yields wrong results.

The proposed **Mi3\_UCM\_AV2\_SM** repository **follows RefAV’s architecture and environment settings** while extending it to overcome both issues without altering its configuration interface.

---

## 2. Method

### 2.1 Fault‑Tolerant Iterative Code Generation (FT‑ICG)

FT‑ICG treats code generation as an iterative debugging loop (Fig. 1). On any exception, the stack trace is injected back into the prompt, instructing the LLM to *"fix the bug and avoid similar errors."* The process repeats up to `K` iterations (default = 4) or until execution succeeds.

*This simple scaffolding recovers from >90 % of failures in practice.*

### 2.2 Enhanced Prompting for Spatial Relational Functions (EP‑SRF)

Before generation, we prepend explicit guidelines:

> *If you use `has_objects_in_relative_direction()` / `being_crossed_by()` / `heading_in_relative_direction_to()`, the **direction** is relative to **track\_candidates**; for `facing_toward()` and `heading_toward()` the **track\_candidates** must face **related\_candidates**.*

These clarifications reduce argument swaps and eliminate a major source of silent semantic error.

---

## 3. Experiments

### 3.1 Implementation Details

* **Dataset:** Argoverse 2 sensor suite with Le3DE2D tracks.
* **Metrics:** HOTA‑Temporal (primary), HOTA, Timestamp F1, Log F1.
* **LLMs:** Qwen‑2.5‑VL‑7B (local RTX 4090), Gemini 2.5 Flash/Pro (remote API).
* **Tracking:** LT3D official public track.
* Exceeded retry budgets are *rare* (<1 %) and fixed manually.

### 3.2 Ablation Results (Validation Set)

| LLM                  | Method            |    HOTA‑T |      HOTA |     TS‑F1 |    Log‑F1 |
| -------------------- | ----------------- | --------: | --------: | --------: | --------: |
| **Qwen‑2.5‑VL‑7B**   | Baseline RefAV    |     33.27 |     36.72 |     61.94 |     58.12 |
|                      | + FT‑ICG          |     34.71 |     39.32 |     62.77 |     58.09 |
|                      | + FT‑ICG + EP‑SRF | **37.55** | **42.48** | **65.03** | **60.90** |
| **Gemini 2.5 Flash** | Baseline RefAV    |     42.73 |     44.27 |     69.84 |     60.13 |
|                      | + FT‑ICG          |     44.13 |     45.07 |     70.44 |     60.66 |
|                      | + FT‑ICG + EP‑SRF | **44.58** | **45.12** | **71.54** | **60.79** |
| **Gemini 2.5 Pro**   | Baseline RefAV    |     43.34 |     45.57 |     69.84 |     59.13 |
|                      | + FT‑ICG          |     45.53 |     46.07 |     71.34 |     59.66 |
|                      | + FT‑ICG + EP‑SRF | **46.71** | **45.93** | **72.30** | **61.36** |

**Official Test Set (Gemini 2.5 Pro + FT‑ICG + EP‑SRF)** → **52.37 HOTA‑Temporal** · 51.53 HOTA · 77.48 TS‑F1 · 65.82 Log‑F1.

---

## 4. Repository Structure & Quick Start

```
Mi3_UCM_AV2_SM/
├── src/
│   ├── atomic_fns/               # (Auto‑)generated code snippets
│   ├── configs/
│   │   ├── experiments.yml       # LLM settings (see below)
│   │   └── paths.py              # Local paths
│   └── ...
├── tolerance_manual_test_gen_code/  # Pre‑generated predictions
└── run_experiment.py              # Entry point (unchanged)
```

### Setup

> **Environment note:** Please modify paths.py to the correct path, add the configuration for gemini-2.5-pro-preview-05-06 to experiments.yml, and run run_experiment.py to generate the combined atomic function according to the description and run it.

### Run

```bash
python run_experiment.py \
    --experiment {your-exp} \
    --split {split}
```
### Inference

Please modify LLM_PRED_DIR in paths.py to the **tolerance_manual_test_gen_code** folder and use the code that has already been generated.

Integrated CLIP methods will come in the future.
