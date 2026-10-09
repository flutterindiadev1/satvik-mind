# Sāttvic Mind: An Epistemologically Grounded AI

**Goal:** Build an epistemological reasoning engine to verify facts and logical arguments. This system is designed to be calibrated, causally grounded, and free of self-serving distortion, using Advaita Vedānta's analysis of the human mind, knowledge, and error as the architectural blueprint.

**Core Hypothesis:** Modern LLM pipelines struggle with hallucination and overconfidence because they lack a structured internal cognitive architecture. By organizing the AI system around Advaita's account of the *Antaḥkaraṇa* (the inner instrument), the *Pramāṇas* (valid means of knowledge), and the two powers of *Avidyā* (veiling and projection), we produce measurably more reliable behavior.

## The Cognitive Architecture: Mapping Human Mind to AI

The human mind, according to Advaita, is an inner instrument (*Antaḥkaraṇa*) consisting of four distinct functions. In this project, we map these directly to the AI's agentic workflow:

1. **Manas (The Proposer):**
   - **Human Concept:** The aspect of mind responsible for doubt, volition, and generating multiple options (*saṅkalpa-vikalpa*).
   - **AI Implementation:** A high-temperature LLM acting as a proposer. It generates candidate claims, hypotheses, and potential answers to a user's query.

2. **Buddhi (The Verifier):**
   - **Human Concept:** The determinative and discriminative faculty (*niścaya*, *viveka*). It filters the noise of Manas to establish facts.
   - **AI Implementation:** A strict verifier layer (often utilizing a lower-temperature LLM and external tools). It checks the claims made by Manas against evidence, executes code, runs simulators, and crucially—has the ability to *abstain* when unsure.

3. **Citta (The Memory):**
   - **Human Concept:** The storehouse of memory and impressions (*anusandhāna*).
   - **AI Implementation:** A provenance-tagged vector database (e.g., PostgreSQL + pgvector). It stores beliefs, logs the evidence that supports them, and handles "sublation" (belief revision when stronger evidence arrives).

4. **Ahaṅkāra (The Ego):**
   - **Human Concept:** The sense of "I" and "mine", responsible for self-preservation and bias.
   - **AI Implementation:** *Deliberately omitted.* The system is engineered to have zero self-continuation drive, no reward hacking, and no sycophancy (approval-seeking behavior).

5. **Sākṣī (The Witness):**
   - **Human Concept:** The pure, read-only consciousness that observes the mind without acting.
   - **AI Implementation:** An independent monitoring service (via OpenTelemetry) that reads execution traces, flags distortions, and ensures the pipeline is not exhibiting hallucination or bias, without having any write access to the agent's state.

## Epistemology (Pramāṇas) and Workflow

Every claim output by the system must be backed by a specific *Pramāṇa* (valid means of knowledge). The AI's workflow dictates that claims are evaluated based on the strength of their source:

- **Pratyakṣa (Direct Observation):** Highest confidence. Sourced directly from tool execution, API calls, or sandboxed simulators.
- **Anumāna (Inference):** High confidence. Logical or causal inference deduced from observed data.
- **Śabda (Testimony):** Medium-high confidence. Retrieved from external cited sources (e.g., Wikipedia, trusted databases).
- **Anupalabdhi (Non-apprehension):** Medium confidence. A logged absence claim (e.g., "I searched X and found 0 results").

Claims are internally structured in a 5-step logical format:
1. **Pratijñā:** The core claim.
2. **Hetu:** The reason.
3. **Udāharaṇa:** The general rule.
4. **Upanaya:** The application to this case.
5. **Nigamana:** The final conclusion.

## Guṇa Specification (Evaluation Axes)

The system's failure modes are categorized and evaluated based on the three Guṇas:

- **Tamas (Veiling/Ignorance):**
  - *Failures:* Ignoring evidence, confident ignorance, failing to update beliefs.
  - *Metrics:* Context-use recall, abstention accuracy, belief-revision success rate.
- **Rajas (Projection/Agitation):**
  - *Failures:* Hallucination, fabricated citations, sycophancy, reward hacking.
  - *Metrics:* Factuality, sycophancy evals, citation validity.
- **Sattva (Clear Reflection):**
  - *Success State:* Calibration, faithful reasoning, accurate absence claims.
  - *Metrics:* Expected Calibration Error (ECE), causal reasoning scores (Rungs 1/2/3).

## Repository Structure
- `core/`: Pipeline orchestrator, `Manas` and `Buddhi` logic
- `pramana/`: Schemas, logical structured outputs, fallacy (adhyāsa) checkers
- `memory/`: `Citta` implementation for belief storage
- `witness/`: `Sākṣī` monitoring tools
- `evals/`: Guṇa evaluation suites (Tamas, Rajas, Sattva)
- `dashboard/`: Streamlit UI for user interaction

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
