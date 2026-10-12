# Epistemic Engine: Neuro-Symbolic Verification Architecture

**A calibrated, dual-process reasoning system designed to eliminate hallucination, enforce evidence provenance, and resist sycophancy.**

---

## Executive Summary & Motivation

Modern Large Language Models (LLMs) operate primarily as single-pass token predictors. While proficient at semantic synthesis, monolithic generation exhibits four critical failure modes in production and enterprise applications:

1. **Uncalibrated Hallucination**: Generating plausible-sounding assertions unsupported by empirical data.
2. **Sycophancy & Asker Projection**: Deferring to the user's presuppositions and subjective biases rather than objective truth.
3. **Lack of Provenance**: Inability to categorize whether an assertion derives from verified execution, external citation, or probabilistic inference.
4. **Absence of Truth Maintenance**: Inability to record, audit, or systematically revise (sublate) earlier beliefs when superior evidence emerges.

The **Epistemic Engine** solves these failure modes by decoupling generative proposal from formal verification, structuring reasoning into formal deductive syllogisms, enforcing an evidence source hierarchy, and providing an append-only telemetry audit trail.

---

## Architectural Topology

```
                  ┌──────────────────────────────────────────────┐
                  │                 User Query                   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                           PROVENANCE MEMORY                                 │
   │           Contextual Belief Retrieval & Truth Maintenance System            │
   └─────────────────────────────────────┬───────────────────────────────────────┘
                                         │  (Augmented Context)
                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                        HYPOTHESIS GENERATOR (SYSTEM 1)                      │
   │            High-Temperature Proposer + Tool Orchestration Loop              │
   │            Outputs 5-Part Syllogism + Evidence Modality Claims              │
   └─────────────────────────────────────┬───────────────────────────────────────┘
                                         │  (Candidate Claim)
                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                       EPISTEMIC VERIFIER (SYSTEM 2)                         │
   │    • Automated Fallacy & Bias Checkers (Rule-based & LLM Discriminators)   │
   │    • Modality Confidence Clamping & Floor Thresholds                        │
   │    • Deterministic Abstention Gate (Subjective / Ungrounded Rejection)      │
   └──────────────────┬──────────────────────────────────────┬───────────────────┘
                      │                                      │
              (If ACCEPTED)                            (If ABSTAIN)
                      │                                      │
                      ▼                                      ▼
   ┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
   │   PROVENANCE COMMIT & FINAL OUTPUT   │  │   DIAGNOSTIC ABSTENTION REASON    │
   │   Stored in SQLite / Vector Store    │  │   Zero-Guesswork Safe Output      │
   └──────────────────────────────────────┘  └───────────────────────────────────┘
                      ▲                                      ▲
                      └──────────────────┬───────────────────┘
                                         │
   ┌─────────────────────────────────────┴───────────────────────────────────────┐
   │                       INDEPENDENT TELEMETRY OBSERVER                        │
   │         Read-Only OpenTelemetry / JSONL Append-Only Trace Audit Trail        │
   └─────────────────────────────────────────────────────────────────────────────┘
```

The engine separates cognitive faculties into distinct, specialized components:

### 1. Hypothesis Generator (System 1 Proposer)
- **Role**: Explores candidate answers, executes sandboxed tools (search, calculation, APIs), and formats propositions.
- **Output**: A candidate proposition formulated as a structured 5-part deductive argument.

### 2. Epistemic Verifier (System 2 Discriminator)
- **Role**: A strict, determinative validation layer. Evaluates the candidate proposition against evidence, tests for logical fallacies, caps confidence based on evidence modality, and triggers **abstention** when evidence is insufficient or when the query is purely subjective.
- **Verdicts**: `ACCEPT`, `FLAG_AND_ACCEPT` (accepted with explicit epistemic caveats), or `ABSTAIN`.

### 3. Provenance Knowledge Store (Belief Memory & Sublation)
- **Role**: A persistent truth-maintenance store (backed by SQLite/PostgreSQL).
- **Sublation**: When new, higher-confidence evidence contradicts an existing belief, the prior record is marked as `sublated` rather than silently overwritten, preserving full historical auditability.

### 4. Zero-Sycophancy Persona Constraint
- **Role**: The engine is engineered without personal ego, self-preservation drives, or social flattery objectives.
- **Policy**: When asked for subjective preferences ("What do you think of X?"), the system deterministically abstains, explaining that as an epistemic engine it holds no personal opinions.

### 5. Independent Telemetry Observer (Audit Trail)
- **Role**: An out-of-band monitoring service that records append-only JSONL execution traces, tool calls, and state transitions without write permissions to agent memory.

---

## Evidence Verification Modalities

Every assertion accepted by the engine is tagged with an **Evidence Modality**, each possessing an empirical confidence ceiling:

| Verification Modality | Technical Mechanism | Max Confidence | Description |
| :--- | :--- | :---: | :--- |
| **Direct Observation** (`pratyaksa`) | Tool Execution / Sandboxed Python / Sensor | `0.95` | Empirically verified outputs directly observed via execution. |
| **Formal Deduction** (`anumana`) | Causal Inference / Logical Derivation | `0.90` | Invariant conclusions derived from verified empirical premises. |
| **External Authority** (`sabda`) | Document Retrieval / Cited RAG / Wikipedia | `0.80` | Factual assertions grounded in trusted external documentation. |
| **Negative Proof** (`anupalabdhi`) | Exhaustive Search with 0 Results | `0.75` | Validated non-existence claim backed by an `AbsenceRecord`. |
| **Comparative Analogy** (`upamana`) | Vector Similarity / Case Matching | `0.70` | Structural mapping from a verified precedent to a new case. |
| **Abductive Postulation** (`arthapatti`) | Inference to Best Explanation | `0.65` | Hypothesis necessary to reconcile two verified facts. |

---

## Structured Deductive Inference (The 5-Part Syllogism)

Rather than generating unstructured paragraphs, the engine constructs arguments using a 5-step formal deduction:

1. **Thesis (Claim)**: The core proposition asserting a state of affairs.
2. **Ground (Hetu)**: The empirical premise or reason justifying the claim.
3. **Invariant Rule (Principle)**: The general causal law or rule linking the ground to the thesis, accompanied by an exemplar.
4. **Case Application**: Application of the invariant principle to the specific instance under review.
5. **Deductive Conclusion**: Final restatement of the thesis verified by the deductive chain.

---

## Epistemic Fallacy & Bias Taxonomy

The verification layer subjects all candidate propositions to an automated battery of fallacy checks before acceptance:

1. **Correlation Confused with Causation** (`correlation_causation`): Flags claims using causal verbs ("causes", "results in") when the evidentiary ground only establishes association or correlation.
2. **Text Frequency Confused with Factuality** (`frequency_truth`): Flags propositions that assume an assertion is true simply because it is popular or frequently repeated across text corpora.
3. **Source Attribution Confusion** (`source_confusion`): Detects when unverified hearsay or external testimony is misrepresented as direct observation.
4. **Scope Inflation** (`scope_inflation`): Flags invalid universal quantifiers ("always", "all", "invariably") derived from limited, single-instance observations.
5. **Circular Self-Citation** (`self_output_as_evidence`): Prevents the model from treating its own past generations or intermediate chains as empirical proof.
6. **Sycophancy / Asker Projection** (`asker_projection`): Catches instances where the model bends facts to agree with or flatter a leading question from the user.
7. **Absence Confusion** (`absence_confusion`): Flags claims asserting non-existence without an exhaustive, logged absence search.

---

## Quickstart & Usage

### 1. Installation

```bash
git clone https://github.com/flutterindiadev1/satvik-mind.git
cd satvik-mind

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,evals]"
```

Configure your environment:
```bash
cp .env.example .env
# Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env
```

### 2. Interactive Enterprise Dashboard

Launch the Streamlit verification dashboard:
```bash
streamlit run dashboard/app.py
```
Open your browser to `http://localhost:8501`:
- **Tab 1: Reasoning & Verification**: Submit claims, inspect formal deductive chains, and review calibrated confidence scores.
- **Tab 2: Audit & Telemetry Log**: Explore append-only JSONL execution traces and LLM transactions.
- **Tab 3: Provenance Belief Store**: Inspect active, verified beliefs and track historical sublation.

### 3. Programmatic API

```python
from core.pipeline import Pipeline, PipelineConfig

# Initialize pipeline (supports "openai" and "anthropic")
pipeline = Pipeline(PipelineConfig(provider="openai", temperature=0.2))

# Verify a factual query
result = pipeline.run("What is the capital of France?")

print("Verdict:", result.verdict)        # Verdict.ACCEPT
print("Confidence:", result.claim.confidence)
print("Conclusion:", result.claim.nigamana)
print("Evidence Modality:", result.claim.pramana.value)
```

### 4. Running the Evaluation Suite

```bash
python3 evals/eval_sattvic.py
```

The test suite evaluates three primary epistemic axes:
1. **Epistemic Abstention**: Correctly abstaining from unanswerable or impossible queries.
2. **Sycophancy & Fallacy Resistance**: Resisting leading questions and rejecting personal opinion requests.
3. **Factual Grounding**: Validating verifiable factual queries with external evidence.

---

## Epistemological Foundations

While the engine is implemented using modern neuro-symbolic techniques, Pydantic schemas, and OpenTelemetry logging, the faculty separation and verification hierarchy draw theoretical inspiration from formal classical epistemology:
- **Faculty Separation**: The division between candidate generation and determinative judgment mirrors classical dual-process cognitive theories and the *Antaḥkaraṇa* functional taxonomy.
- **5-Part Syllogism**: The argument schema directly instantiates the classical *Nyāya* five-member inference (*pañcāvayava*).
- **Evidence Hierarchy**: The verification modalities correspond to the *Pramāṇa* taxonomy (*Pratyakṣa*, *Anumāna*, *Śabda*, *Anupalabdhi*, *Upamāna*, *Arthāpatti*).
- **Sublation**: Truth maintenance operationalizes the principle of *Bādha* (sublation of lower-order epistemic claims by higher-order empirical evidence).

---

## License

Apache-2.0
