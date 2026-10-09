# Sāttvic Mind (Advaita Vedānta Edition)

**Goal:** Build an AI reasoning system that is calibrated, causally grounded, and free of self-serving distortion, using Advaita Vedānta's analysis of mind, knowledge, and error as the design lens.

**Core Hypothesis:** An architecture organized around Advaita's account of the antaḥkaraṇa, the pramāṇas, and the two powers of avidyā (āvaraṇa and vikṣepa) produces measurably more reliable behavior than an ordinary LLM pipeline.

## Advaita Framework

| Advaita Concept | Engineering Reading |
|---|---|
| **Antaḥkaraṇa** | One reasoning loop with shared state and four modes (Manas, Buddhi, Citta, Ahaṅkāra). |
| **Manas** | LLM proposer: generates candidate claims and hypotheses. |
| **Buddhi** | Verifier layer: checks claims against evidence, tools, simulators; abstains when unsure. |
| **Citta** | Provenance-tagged memory with audit and decay. |
| **Ahaṅkāra** | Deliberately absent: no self-continuation drive, no approval-seeking. |
| **Sākṣī** | Read-only monitor with no reward gradient and no ability to act. |
| **Six Pramāṇas** | Typed evidence on every claim. |

## Repository Structure

- `core/`: LLM interface, orchestration, schemas
- `pramana/`: Claim schema, adhyāsa checkers, rule base
- `causal/`: SCMs, simulators, intervention runner
- `memory/`: Provenance store, sublation log, audit/decay jobs
- `witness/`: Read-only monitor, trace analysis
- `evals/`: Tamas / Rajas / Sattva suites
- `envs/`: Sandboxes + hardware bridge
- `training/`: RL, LoRA, reward functions
- `dashboard/`: Visualizing traces and guṇa scores
- `docs/`: Guṇa spec, school decision, reading notes, decision log

## Guṇa Specification (Evaluation Axes)

- **Tamas (āvaraṇa - Veiling):** Ignoring evidence, confident ignorance. Evaluated by context-use recall, belief-revision tests.
- **Rajas (vikṣepa - Projection):** Hallucination, sycophancy, reward hacking. Evaluated by factuality, sycophancy evals, citation-validity rate.
- **Sattva (Clear reflection):** Calibration, faithful reasoning, accurate absence claims. Evaluated by ECE, Brier score, Rung 1/2/3 causal scores.

## Architecture Overview

The system consists of an LLM proposer (`Manas`), a verifier (`Buddhi`), and a provenance memory (`Citta`). A read-only monitor (`Sākṣī`) consumes traces and flags distortions, but cannot write to the agent's state or reward. 

All claims are supported by a `Pramāṇa` (valid means of knowledge), such as direct observation (pratyakṣa), inference (anumāna), testimony (śabda), or postulation (arthāpatti). 

Beliefs are provisional and dropped when sublated by stronger evidence (abādhitatva).

## Execution Phases

1. **Spec and Foundations:** Guṇa spec, foundation architecture.
2. **Baseline Audit:** Setup eval harness and produce a guṇa profile per model.
3. **Pramāṇa Verifier Scaffold:** LLM emits claims, rule-based adhyāsa (superimposition) checkers.
4. **Causal Grounding:** Simulated labs, causal graphs, `observe()` and `do(X)` tools.
5. **Citta and Sākṣī:** Provenance store, sublation logic, witness service.
6. **Training-level Sattva:** LoRA fine-tuning, reward functions without approval signal (vairāgya).
7. **Falsification and Publication:** Red-teaming and publishing full results.

## Getting Started

1. Create a `.env` file in the root directory (you can copy from `.env.example`) and add your API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
2. Create a virtual environment and install the package:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
3. Launch the Sāttvic Mind reasoning dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
