# Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning

> 🎉 **Accepted at ICML 2026.**

[![Project Page](https://img.shields.io/badge/🌐%20Project-Page-4c8bf5.svg)](https://ibisbill.github.io/Transferability-of-LLM-Reasoning/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.00432-b31b1b.svg)](https://arxiv.org/abs/2507.00432)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/ReasoningTransferability)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/ReasoningTransfer/Transferability-of-LLM-Reasoning)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-brightgreen.svg)](https://icml.cc/virtual/2026/poster/65125)

**TL;DR**: Most models that succeed at math *fail* to transfer those gains to other domains. Through controlled experiments we find the decisive factor is not RL vs. SFT per se, but **on-policy vs. off-policy** fine-tuning: on-policy updates (whether the signal comes from RL or supervised learning) preserve—and even improve—general capabilities, while off-policy SFT on static distilled data induces catastrophic forgetting.

📖 **Interactive summary with figures and formulas:** [ibisbill.github.io/Transferability-of-LLM-Reasoning](https://ibisbill.github.io/Transferability-of-LLM-Reasoning/)

## 🔍 Overview

Math reasoning has become the poster child of progress in LLMs, with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as leaderboards improve week by week, a critical question emerges: **Do these gains reflect broader problem-solving ability or just narrow overfitting?**

This repository contains the code, data, and evaluation framework for our study evaluating **over 20 open-weight reasoning-tuned models** across math, scientific QA, agent planning, coding, and standard instruction-following tasks. We surprisingly find that most models succeeding in math fail to transfer their gains to other domains. To rigorously study this, we run controlled experiments with math-only data using **Reinforcement Learning (RL)** and **Supervised Fine-Tuning (SFT)**, with detailed ablations.

### 🎯 Main Research Question
**Do improved math reasoning abilities transfer to general LLM capabilities?**

## 🔑 Key Findings

### 1. On-Policy Fine-Tuning Is the Key Mechanism
Across model families and sizes (1.5B → 32B), the **fine-tuning paradigm** predicts transferability far better than model size or data:
- **RL-tuned models** generalize well across domains and preserve general capabilities.
- **SFT-tuned models** on static distilled corpora suffer catastrophic forgetting on non-reasoning tasks.
- Our ablation isolates *why*: the decisive lever is **on-policy vs. off-policy** sampling. On-policy SFT transfers well too, and off-policy RL transfers poorly — so **on-policy fine-tuning**, not RL alone, is the underlying mechanism.

### 2. The Transferability Index (TI)
We introduce the **Transferability Index (TI)**, a metric quantifying how well math gains transfer to other domains. For each benchmark group *g* ∈ {math, other, non}:

1. **Per-benchmark gain & normalization** — z-normalize gains within each group: `δ_b = ΔR_b / σ_g`.
2. **Robust gain & difficulty weighting** — a signed square-root `s_b = sign(δ_b)·|δ_b|^{1/2}` tempers extremes, harder tasks are up-weighted, and a group-level **Domain Index** aggregates them: `DI_g = Σ ŵ_b·s_b`.
3. **Transfer relative to math** — `TI_g(%) = (DI_g / DI_math) × 100`.

A **positive** TI indicates successful transfer; **negative** indicates degradation. RL-tuned models consistently achieve positive `TI_other` and `TI_non`, whereas SFT models often yield negative `TI_non`.

| Method | Math Reasoning | Other Reasoning | Non-Reasoning |
|--------|---------------|----------------|---------------|
| **RL (on-policy)** | ✅ Strong gains | ✅ Positive transfer | ✅ Preserved / improved |
| **SFT (off-policy)** | ✅ Strong gains | ⚠️ Limited / uneven | ❌ Performance degradation |

### 3. Controlled Study: UniReason
We fine-tune **Qwen3-14B-Base** on a high-quality math dataset (derived from MATH + DeepScaleR, ~47K examples), calling the result **UniReason**:
- **SFT** targets are distilled from **Qwen3-32B** via rejection sampling (keeping only correct traces).
- **RL** uses the same queries with answer-correctness rewards under a **GRPO / DAPO** recipe.

| Group | Metric | Qwen3-14B-Base | UniReason-SFT (think) | UniReason-SFT (no-think) | **UniReason-RL** |
|-------|--------|:--:|:--:|:--:|:--:|
| **Math** | Avg (AIME24/25, MATH500, Olympiad) | 27.7 | 49.8 | 32.3 | **53.8** |
| **Other Reasoning** | Avg (GPQA, LCB, ACPBench, HeadQA) | 30.2 | 45.3 | 44.1 | **51.0** |
| | `TI_other` | – | +52.2 | +165.4 | **+82.3** |
| **Non-Reasoning** | Avg (CoQA, MC-TACO, IFEval, HaluEval) | 45.7 | 21.1 | 29.0 | **53.2** |
| | `TI_non` | – | −104.1 | −278.9 | **+52.2** |

Trained on a *single distilled math dataset*, UniReason-RL still preserves and even **improves** general-domain performance while showing strong reasoning gains.

### 4. Internal Analysis Reveals Why
- **PCA Shift (Section 3)**: RL keeps hidden-state geometry close to the backbone (smallest centroid shifts across all task types); SFT — especially without reasoning signals — drifts substantially, most on non-reasoning inputs.

  | Model | Math | Other | Non |
  |-------|:--:|:--:|:--:|
  | Qwen3-14B (no-think) | 40.4 | 13.9 | 129.8 |
  | Qwen3-14B (think) | 76.5 | 38.8 | 152.0 |
  | UniReason-SFT (no-think) | 21.4 | 10.9 | 113.7 |
  | UniReason-SFT (think) | 19.2 | 6.7 | 38.2 |
  | **UniReason-RL** | **8.5** | **3.5** | **36.9** |

- **KL Divergence & Token Rank (Section 4)**: RL exhibits far lower KL from the backbone (e.g., 0.084 vs. 0.372 on MATH-500) and much smaller token-rank shifts (~0.98 vs. 10.6 positions). RL **selectively** shifts a small set of task-relevant tokens (logical connectives like *But*, *So*), whereas SFT reorders many irrelevant tokens.

### 5. Which Components of RL Drive Transferability? (Ablation)
Using a **unified surrogate loss**, we decompose RL vs. SFT along four levers and test five settings on Qwen3-8B-Base:

| Setting | Math | Other | Non | `TI_other` | `TI_non` |
|---------|:--:|:--:|:--:|:--:|:--:|
| Qwen3-8B-Base | 27.6 | 23.6 | 33.6 | – | – |
| Off-policy SFT | 41.9 | 34.4 | 26.6 | 18.3 | −40.5 |
| **On-policy SFT** | 33.7 | 35.7 | 35.0 | **68.6** | **30.2** |
| Off-policy RL | 45.5 | 35.9 | 31.7 | 36.4 | 4.5 |
| On-policy RL (no KL) | 37.1 | 38.2 | 35.8 | 65.6 | 39.3 |
| On-policy RL | 38.6 | 39.9 | 35.0 | 63.7 | 32.4 |

- **Sampling distribution is critical** — on-policy consistently beats off-policy in *both* RL and SFT.
- **Credit assignment & negative gradients matter** — advantage weighting + learning from negatives jointly improve transfer.
- **KL regularization plays only a subtle role** — on-policy RL performs about the same with or without it.

## 📊 Benchmark Categories

| Category | Benchmarks | Description |
|----------|------------|-------------|
| **Math Reasoning** | MATH-500, AIME24/25, OlympiadBench | Pure mathematical problem solving |
| **Other Reasoning** | GPQA-Diamond, LiveCodeBench, ACPBench, HeadQA | Scientific QA, coding, agent planning |
| **Non-Reasoning** | CoQA, IFEval, HaluEval, MC-TACO | Conversational QA, instruction following |

Evaluation uses **accuracy** across all groups via [EvalChemy](https://github.com/mlfoundations/evalchemy) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## 📁 Repository Structure

| Directory | Contents |
|-----------|----------|
| [`train/`](train/) | SFT (LLaMA-Factory) and RL (DAPO/GRPO via verl) training recipes and configs |
| [`eval/`](eval/) | Evaluation harnesses (EvalChemy + lm-evaluation-harness) and run scripts |
| [`analyse_PCA_Shift/`](analyse_PCA_Shift/) | Latent-space PCA shift analysis and visualization |
| [`analyse_token_distribution_shift/`](analyse_token_distribution_shift/) | Token-level KL divergence and rank-shift analysis |

## 🚀 Quick Start

**Training** — see [`train/README.md`](train/README.md) for the full SFT and RL configs (Qwen3-14B-Base, GRPO with `kl_coef=0.0`, on-policy rollouts).

**Evaluation** — see [`eval/readme.md`](eval/readme.md):
```bash
cd eval/evalchemy
conda create -n evalchemy python=3.10 -y && conda activate evalchemy
pip install -e . && pip install vllm==0.8.5
bash run_your_model.sh
```

**Analysis**:
```bash
# PCA latent-space shift
cd analyse_PCA_Shift && bash PCA_shift.sh

# Token distribution / KL divergence shift
cd analyse_token_distribution_shift && bash token_level_logits_and_ranks.sh
```

## 🤗 Models

Our UniReason checkpoints are available on Hugging Face under [ReasoningTransferability](https://huggingface.co/ReasoningTransferability), e.g. `ReasoningTransferability/UniReason-Qwen3-14B-RL`.

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{huan2026doesmathreasoningimprove,
      title={Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning},
      author={Maggie Ziyu Huan and Yuetai Li and Tianyu Zheng and Xiaoyu Xu and Seungone Kim and Minxin Du and Radha Poovendran and Graham Neubig and Xiang Yue},
      booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
      year={2026},
      eprint={2507.00432},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.00432},
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
