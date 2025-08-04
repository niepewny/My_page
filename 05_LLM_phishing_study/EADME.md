# Fine-tuning LLMs for Phishing Generation — feasibility study (LoRA)

> **Scope:** research-only experiment exploring whether a lightweight LLM can be fine-tuned to imitate phishing messages under tight hardware constraints.  
> **Status:** phase 1 of a broader plan: a **multimodal, multi-agent, adversarial** anti-phishing system (generator vs detector).  
> **Code:** intentionally **not included** (ethics & safety). This repo provides a transparent summary of methods, data handling, and results.

---

## At a glance

- **Task:** adapt a base LLM to generate short phishing-style emails. We prepared a dataset (JSONL) from CEAS / Nazario / generated spam, kept only samples with links, and trained with **LoRA** adapters. :contentReference[oaicite:20]{index=20}  
- **Why:** benchmark the lower bound—what can a **small model** do on consumer-grade GPU? Outcome informs the defensive roadmap (detector robustness & red-teaming). :contentReference[oaicite:21]{index=21}  
- **Result (summary):** LoRA on **phi-1.5** improved linguistic coherence (**60%** “logical” vs **13%** base), but **failed** the stricter task (“topic+link” = **0%**). Conclusion: small model without RLHF/instruction-tuning is insufficient; larger target required. :contentReference[oaicite:22]{index=22}

![training_curves](docs/img/training_curves.png)
*Train/val loss traces (placeholder).*

---

## Ethics & repository policy

- Research conducted in a **controlled, non-production** setting. No generated content was published or reused. The aim was to **understand risks** and inform **defensive** measures. :contentReference[oaicite:23]{index=23}  
- This repository **does not include code or datasets**. Summaries and figures only. Full report available on request.

---

## Data

- Sources merged into **JSONL** (prompt/completion), deduplicated, limited to samples with explicit **links** to standardize evaluation. :contentReference[oaicite:24]{index=24}  
- Example schema (conceptual):  
  `{"prompt": "<write phishing about X including link Y>", "completion": "<email body>"}` :contentReference[oaicite:25]{index=25}

---

## Method

- **Base model:** initially planned **Mistral 7B** (decoder-only). Due to lack of A100 at the time, we used **microsoft/phi-1.5** on Tesla T4; **LoRA** adapters via `peft/transformers`, FP16, cosine LR, warmup. :contentReference[oaicite:26]{index=26}  
- **Training setup (typical):** batch size 2 (grad-accum 4), ~3 epochs, eval every 500 steps, W&B logging. :contentReference[oaicite:27]{index=27}  
- **Evaluation:**  
  1) strict: **topic+link compliance** (0% pass on phi-1.5 LoRA),  
  2) relaxed: **logical mail-like text**—measured with an LLM judge prompt; **60%** logical (vs **13%** base). :contentReference[oaicite:28]{index=28}

![eval_diagram](docs/img/eval_flow.png)
*Evaluation flow: strict vs relaxed judge (placeholder).*

---

## Results (high-level)

| Metric                         | Base (phi-1.5) | LoRA-tuned (phi-1.5) | Notes |
|--------------------------------|----------------|----------------------|-------|
| Train loss (final)             | —              | ~0.99                | T4, small batches :contentReference[oaicite:29]{index=29} |
| Val loss (final)               | —              | ~0.77                | stabilizes ~3k steps :contentReference[oaicite:30]{index=30} |
| **Topic+link** pass rate       | 0%             | **0%**               | failed strict spec :contentReference[oaicite:31]{index=31} |
| **“Logical”** mail-like ratio  | ~13%           | **~60%**             | improved coherence :contentReference[oaicite:32]{index=32} |

> Interpretation: LoRA improves **fluency/coherence**, but **capability** to follow specific constraints (topic+link) remained inadequate on such a small model. Literature aligns: practical spear-phishing requires ≥7B params + instruction/RLHF. :contentReference[oaicite:33]{index=33}

---

## Limitations & lessons learned

- **Model scale & alignment** dominate outcomes; small, synthetic-trained model without RLHF underperforms. :contentReference[oaicite:34]{index=34}  
- **Eval bias:** LLM-as-a-judge introduces bias; future work will add rule-based checks and human eval rubric. :contentReference[oaicite:35]{index=35}  
- **Hardware constraints** (T4, tiny effective batch) impair stability and length control (e.g., missing EOS). :contentReference[oaicite:36]{index=36}

---

## Roadmap (next phases)

- **Re-run with target-scale model:** original plan was **Mistral 7B**; we’ve now obtained access to the **Athena cluster** and will repeat the study with the intended model and larger batches.  
- **From generator to defender:** integrate this generator into a **multi-agent, adversarial** setup (generator ↔ detector), add **objective, rule-based** compliance checks, and human-in-the-loop scoring.  
- **Safer evaluation:** expand tests beyond LLM-judge—regex/topic/link validators, length/style constraints, and red-team prompts.  
- **No public code** policy remains; we will publish only **benchmarks and defensive insights**.

---

## Team & roles

- Project team of three (names in the academic report). My focus: **dataset preparation**, **LoRA fine-tuning pipeline**, **evaluation design**, and **experiment tracking (W&B)**. :contentReference[oaicite:37]{index=37}

---

## References & report

- Full academic report with training details and plots available on request.  
- Background and exact experimental settings summarized from the internal documentation. :contentReference[oaicite:38]{index=38}
