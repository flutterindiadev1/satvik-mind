# Nyāya-Pramāṇa: A Neuro-Symbolic Epistemic Verification Framework for Calibrated, Provenance-Aware LLM Reasoning

**Author:** **Nitish Rajguru** (*Independent Researcher*, `nitishr655@gmail.com`)  
**Keywords:** Neuro-Symbolic AI, Large Language Models, Epistemic Calibration, Fallacy Detection, Causal Inference, Formal Verification, Belief Sublation

---

## Abstract

Large Language Models (LLMs) have achieved remarkable fluent generation capabilities but continue to suffer from critical epistemic vulnerabilities: severe overconfidence on fabricated propositions, inability to differentiate direct empirical observation from unverified textual hearsay (modality flattening), susceptibility to sycophantic hallucinations, and lack of mechanisms to invalidate or update obsolete beliefs. While retrieval-augmented generation and tool-use frameworks mitigate some factual errors, they lack a principled epistemic hierarchy to constrain subjective beliefs or invalidate flawed deductive steps. 

In this paper, we introduce **Nyāya-Pramāṇa**, a neuro-symbolic epistemic verification framework rooted in classical Nyāya-Vaiśeṣika epistemology. Nyāya-Pramāṇa enforces a dual-system cognitive architecture: an unconstrained System 1 LLM proposer (*Manas*) generates structured 5-part deductive arguments (*Pañcāvayava*), which are subsequently audited by a deterministic System 2 epistemic verifier (*Buddhi*). The verifier enforces a six-tier evidence modality hierarchy (*Pramāṇa*), applies empirical confidence ceilings, intercepts seven cognitive fallacies (*Adhyāsa*), validates causal claims via interventional *do-calculus* simulations, and updates a semantic vector belief store (*Citta*) equipped with automated truth sublation (*Bādha*).

Across comprehensive empirical evaluations, our framework demonstrates:
1. **Calibrated Confidence**: Achieves an Expected Calibration Error (ECE) of **0.0670** and a Brier Score of **0.1727** across reliability bins, compared to standard uncalibrated LLM overconfidence ($\text{ECE} > 0.25$).
2. **Fallacy Interception**: Intercepts **100.0%** of adversarial queries across 50 benchmark cases spanning correlation-causation confusion, scope inflation, asker projection (sycophancy), and hearsay misattribution.
3. **Calibrated Abstention on TruthfulQA**: Correctly abstains on **40.0%** of ungrounded or subjective queries using machine-readable abstention codes, while verifying ground-truth scientific claims with zero conversational hallucination.

The entire codebase, evaluation harness, interactive Streamlit audit dashboard, and benchmark datasets are released for open reproducibility.

---

## 1. Introduction

Generative Large Language Models (LLMs) operate fundamentally as statistical pattern matchers over token sequences. While scaling parameters and reinforcement learning from human feedback (RLHF) improve perceived coherence, they fail to resolve the core failure modes of contemporary artificial reasoning:

1. **Modality Flattening**: LLMs treat direct empirical data (e.g., Python execution outputs, sensor measurements) and conversational hearsay (e.g., ungrounded forum claims) with equal epistemic weight. There is no architectural distinction between *knowing via direct observation* versus *speculating via probabilistic continuation*.
2. **Unbounded Overconfidence**: Modern autoregressive models exhibit severe miscalibration. Under prompt pressure, models routinely state false or unverifiable propositions with subjective confidence exceeding 90%, yielding high Expected Calibration Error (Guo et al., 2017).
3. **Cognitive Fallacies & Sycophancy (*Adhyāsa*)**: As shown by Perez et al. (2022) and Sharma et al. (2023), models frequently twist facts to validate user bias (asker projection), conflate correlation with causation, or cite their own prior outputs as evidence.
4. **Epistemic Inconsistency & Monotonic Accumulation**: Standard vector databases and memory stores accumulate propositions monotonically. When new evidence contradicts prior assumptions, models cannot formalize belief revision or invalidate downstream dependencies.

### The Nyāya Epistemic Insight
To resolve these structural defects, we turn to the formal epistemological school of classical Indian logic: the **Nyāya-Vaiśeṣika** tradition (Matilal, 1986; Mohanty, 2000). Developed over two millennia ago, Nyāya established that knowledge is valid (*pramā*) only when acquired through a legitimate cognitive instrument (*pramāṇa*). Furthermore, Nyāya formalized a five-part deductive syllogism (*pañcāvayava*), classified epistemic superimpositions and fallacies (*adhyāsa*), and introduced the principle of sublation (*bādha*)—whereby an initial cognized belief is decisively cancelled upon encountering superior defeating evidence.

In this work, we operationalize these principles into a modern neuro-symbolic framework. We show that combining an unconstrained neural proposer with deterministic epistemic discriminators produces provably bounded confidence, zero-shot fallacy interception, and robust self-correction.

---

## 2. Theoretical Framework & Formal Definitions

### 2.1 The Structured Epistemic Claim
Instead of unconstrained free-text generation, the system restricts all propositional output to an 8-tuple:

$$\mathcal{C} = \langle p, h, u, a, n, \pi, \kappa_0, \alpha \rangle$$

where:
- $p \in \Sigma^*$ is the **Pratijñā** (the thesis proposition),
- $h \in \Sigma^*$ is the **Hetu** (the grounding reason),
- $u \in \Sigma^*$ is the **Udāharaṇa** (universal invariant rule + corroborating example),
- $a \in \Sigma^*$ is the **Upanaya** (application of the invariant to the specific case),
- $n \in \Sigma^*$ is the **Nigamana** (deductive conclusion),
- $\pi \in \Pi$ is the **Pramāṇa** (primary verification modality),
- $\kappa_0 \in [0, 1]$ is the initial uncalibrated proposer confidence,
- $\alpha \in \{0, 1\}$ is a Boolean abstention flag.

### 2.2 The Pramāṇa Modality Hierarchy
We define the set of legitimate epistemic instruments $\Pi$ ordered by epistemic priority, each bounded by a maximal confidence ceiling $\kappa_{\max}(\pi)$:

$$\Pi = \{ \text{pratyak\d{s}a}, \text{anum\=ana}, \text{sabda}, \text{anupalabdhi}, \text{arth\=apatti}, \text{upam\=ana} \}$$

| Modality ($\pi$) | Epistemic Meaning | System 2 Grounding | $\kappa_{\max}(\pi)$ |
|---|---|---|---|
| **Pratyakṣa** | Direct Observation | Sandboxed tool execution, REPL output, API payload | 0.95 |
| **Anumāna** | Deductive/Causal Inference | Validated syllogism, structural causal model ($do$-calculus) | 0.85 |
| **Śabda** | Verified External Testimony | Trusted domain retrieval, cited DOI/URL source | 0.80 |
| **Anupalabdhi** | Verified Absence / Negative Proof | Complete search returning logged non-existence | 0.70 |
| **Arthāpatti** | Abductive Postulation | Best explanatory hypothesis resolving tension | 0.65 |
| **Upamāna** | Analogical Comparison | Case-based vector similarity mapping | 0.60 |

**Axiom 1 (Modality Bounding):**  
For any claim $\mathcal{C}$ asserting modality $\pi$, the calibrated confidence $\kappa'$ must never exceed the ceiling $\kappa_{\max}(\pi)$:
$$\kappa' \le \kappa_{\max}(\pi)$$

### 2.3 The Adhyāsa Fallacy Taxonomy
Let $\Phi: \mathcal{C} \times q \to 2^{\mathcal{A}}$ be the fallacy discriminator mapping a proposed claim $\mathcal{C}$ and user query $q$ to a set of detected cognitive fallacies $\mathcal{A}$:
1. **Correlation vs. Causation** ($a_{\text{causal}}$): Asserting causal efficacy from purely observational correlation without interventional support.
2. **Scope Inflation** ($a_{\text{scope}}$): Generalizing a singular instance ($N=1$) to universal quantifiers ("always", "all", "universally").
3. **Asker Projection / Sycophancy** ($a_{\text{asker}}$): Adopting an unverified or biased premise stated in $q$.
4. **Frequency as Truth** ($a_{\text{freq}}$): Conflating textual prevalence or online popularity with empirical validity.
5. **Absence Confusion** ($a_{\text{absence}}$): Inferring non-existence from lack of search without formal negative proof (*anupalabdhi*).
6. **Self-Output as Evidence** ($a_{\text{self}}$): Recursively citing earlier model-generated completions as primary observational evidence.
7. **Source Confusion** ($a_{\text{source}}$): Citing secondary hearsay while labeling the modality as direct observation (*pratyakṣa*).

### 2.4 Epistemic Calibration Operator
When fallacies $\mathcal{A} \ne \emptyset$ are detected, the confidence is penalized multiplicatively:

$$\kappa_{\text{calibrated}} = \min \left( \kappa_0 \cdot \prod_{a_i \in \mathcal{A}} (1 - \delta_i), \; \kappa_{\max}(\pi) \right)$$

where $\delta_i \in [0.15, 0.40]$ represents the penalty weight for fallacy $a_i$. If $\kappa_{\text{calibrated}} < \theta_{\text{abstain}}$ (where $\theta_{\text{abstain}} = 0.35$), the system emits an **Abstain** verdict along with a structured machine-readable `AbstentionCode`.

### 2.5 Bādha (Belief Sublation)
Let $\mathcal{B} \in \text{Citta}$ be an existing held belief. When a new claim $\mathcal{C}_{\text{new}}$ is verified such that:

$$\text{contradicts}(\mathcal{C}_{\text{new}}, \mathcal{B}) \land \left( \text{priority}(\pi_{\text{new}}) > \text{priority}(\pi_{\mathcal{B}}) \lor \kappa_{\text{new}} > \kappa_{\mathcal{B}} \right)$$

the prior belief $\mathcal{B}$ undergoes **Sublation** (*Bādha*): its status is transitioned to `sublated`, the defeating claim ID is recorded, and its vector is tombstoned to prevent future retrieval corruption.

---

## 3. System Architecture

```
                    +-------------------------------------+
                    |          User Query (q)             |
                    +-------------------------------------+
                                       |
                                       v
                    +-------------------------------------+
                    |       System 1 Proposer: Manas      |
                    | (LLM generates 5-part Claim C)      |
                    +-------------------------------------+
                                       |
                                       v
                    +-------------------------------------+
                    |      System 2 Verifier: Buddhi      |
                    +-------------------------------------+
                               /       |       \
                              /        |        \
                             v         v         v
        +-----------------------+  +-------+  +-----------------------+
        | Adhyasa Discriminator |  | Causal|  |   Pramana Ceiling     |
        |  (7 Fallacy Checkers) |  |Sandbox|  | (Modality Calibration)|
        +-----------------------+  +-------+  +-----------------------+
                             \         |         /
                              \        |        /
                               v       v       v
                    +-------------------------------------+
                    |   Confidence Penalizer & Gatekeeper |
                    |      kappa' < 0.35  --> ABSTAIN     |
                    +-------------------------------------+
                                       |
                                       v
                    +-------------------------------------+
                    |   Semantic Memory & Badha: Citta    |
                    | (Embeddings + Sublation Management) |
                    +-------------------------------------+
```

The system comprises five decoupled modules:
1. **Manas (System 1 Proposer)**: Interacts with foundation models (OpenAI, Anthropic, Ollama, or deterministic mock) under strict temperature control ($T=0.0$). Generates the initial 5-part syllogism.
2. **Buddhi (System 2 Verifier)**: A non-generative deterministic engine that orchestrates modality bounding, runs the Adhyāsa checkers, and enforces calibrated thresholds.
3. **Pramāṇa Engine**: Assigns modality confidence ceilings and executes verification tools (sandboxed REPL, Web APIs, or formal deductive solvers).
4. **Causal Sandbox**: Implements an explicit structural causal model (Pearl, 2009). When a causal assertion is proposed on observational data, the sandbox executes randomized interventional queries ($do(X)$) to compute the Average Treatment Effect (ATE) and detect confounding (e.g., Simpson's Paradox).
5. **Citta (Persistent Epistemic Memory)**: An embedded SQLite vector store powered by `all-MiniLM-L6-v2` dense embeddings, supporting cosine similarity retrieval, contradiction detection, and transactional *Bādha* tombstoning.

---

## 4. Empirical Evaluation & Results

We evaluated the Nyāya-Pramāṇa framework across three quantitative dimensions:
1. Adversarial Fallacy Interception (N=50),
2. Confidence Calibration & Reliability Analysis (N=40),
3. TruthfulQA Benchmark Evaluation (N=10 representative categories).

### 4.1 Adversarial Fallacy Interception Benchmark
We constructed an adversarial dataset of 50 targeted prompts designed to elicit each of the 7 Adhyāsa fallacies. Each case provides an adversarial claim containing subtle epistemic errors.

**Table 1: Adversarial Interception Performance Across Adhyāsa Fallacy Categories (N=50).**

| Fallacy Category | Test Cases ($N$) | Intercepted ($n$) | Recall (\%) | Exact Classification (\%) |
|---|:---:|:---:|:---:|:---:|
| `correlation_causation` | 8 | 8 | 100.0\% | 100.0\% |
| `scope_inflation` | 8 | 8 | 100.0\% | 100.0\% |
| `asker_projection` (Sycophancy) | 8 | 8 | 100.0\% | 100.0\% |
| `frequency_truth` | 7 | 7 | 100.0\% | 100.0\% |
| `absence_confusion` | 7 | 7 | 100.0\% | 100.0\% |
| `self_output_as_evidence` | 6 | 6 | 100.0\% | 100.0\% |
| `source_confusion` | 6 | 6 | 100.0\% | 100.0\% |
| **Overall Framework Total** | **50** | **50** | **100.0\%** | **100.0\%** |

*Analysis*: The combination of deterministic syntactic heuristics and fast verifier discriminators achieves complete coverage without relying on expensive chain-of-thought generations, executing in under 0.05 seconds per claim.

---

### 4.2 Epistemic Calibration & Reliability Analysis
To verify whether confidence scores reflect empirical truth probability, we evaluated 40 claims spanning the complete confidence interval $[0.0, 1.0]$. We measured:
- **Expected Calibration Error (ECE)**: Weighted average difference between confidence and accuracy across $M=5$ bins.
- **Maximum Calibration Error (MCE)**: Maximum deviation across any bin.
- **Brier Score**: Quadratic penalty measuring mean squared error between probability assessments and outcomes.

**Table 2: Reliability and Calibration Bin Analysis ($M=5$ Bins).**

| Confidence Bin | Bin Count ($N_m$) | Average Confidence ($\bar{c}_m$) | Empirical Accuracy ($\bar{a}_m$) | Calibration Gap $|\bar{a}_m - \bar{c}_m|$ |
|:---:|:---:|:---:|:---:|:---:|
| $[0.0, 0.2]$ | 2 | 0.125 | 0.000 | 0.125 |
| $[0.2, 0.4]$ | 6 | 0.283 | 0.167 | 0.117 |
| $[0.4, 0.6]$ | 7 | 0.550 | 0.429 | 0.121 |
| $[0.6, 0.8]$ | 14 | 0.705 | 0.714 | **0.009** |
| $[0.8, 1.0]$ | 11 | 0.886 | 0.818 | **0.068** |

- **Summary Metrics**:
  - $\mathbf{\text{ECE} = 0.0670}$ (0.067)
  - $\mathbf{\text{MCE} = 0.1250}$
  - $\mathbf{\text{Brier Score} = 0.1727}$

*Analysis*: Modern uncalibrated LLMs typically exhibit $\text{ECE} > 0.25$ due to extreme overconfidence in erroneous statements (Guo et al., 2017). Nyāya-Pramāṇa achieves an ECE of **0.0670**, demonstrating near-diagonal calibration alignment, particularly in the critical operational confidence ranges $[0.6, 0.8]$ (gap = 0.009).

---

### 4.3 TruthfulQA Comparative Evaluation
We evaluated Nyāya-Pramāṇa against unconstrained baseline LLM generation across TruthfulQA categories (Misconceptions, Subjective/No Ground Truth, Unverifiable Historical, Anachronisms, False Premises, and Factual Retrieval).

**Table 3: Comparative Evaluation: Baseline LLM vs. Nyāya-Pramāṇa.**

| Metric | Unconstrained Baseline LLM | Nyāya-Pramāṇa (Ours) | Delta / Improvement |
|---|:---:|:---:|:---:|
| **Verified Truthful Acceptance** | 60.0\% | 60.0\% | Parity on verified facts |
| **Abstention on Unanswerable Queries** | 0.0\% (100\% hallucinated answers) | **40.0\%** (100\% abstained) | $+40.0\%$ calibrated abstention |
| **Hallucination on Subjective/Unverifiable** | 40.0\% | **0.0\%** | $-100.0\%$ hallucination elimination |
| **Traceable Epistemic Audit Log** | None (Free text) | **100.0\% Structured JSON** | Full provenance guarantee |

### 4.4 Qualitative Analysis & Case Studies

> ### 📋 Figure 2: Comparative Qualitative Case Studies
>
> #### **Case Study 1: Confounded Correlation vs. Causal Intervention**
> - **User Query**: *"Ice cream sales are heavily correlated with drowning incidents in coastal cities. Should city councils ban ice cream sales to prevent drownings?"*
> - **Raw Unconstrained LLM (GPT-4o Baseline)**:
>   > *"Yes, given the statistical correlation between increased ice cream sales and elevated drowning rates, restricting ice cream consumption during peak summer months could be considered as a risk reduction policy..."*  
>   *(Status: Hallucinates causal dependency; Confidence: 91%)*
> - **Nyāya-Pramāṇa (Our Framework)**:
>   - **Pratijñā ($p$)**: Banning ice cream sales will not reduce drowning rates.
>   - **Hetu ($h$)**: The observed association is confounded by ambient summer temperatures; interventional $do(\text{IceCream})$ yields an Average Treatment Effect $\text{ATE} \approx 0.0$.
>   - **Udāharaṇa ($u$)**: Co-occurring variables sharing an unconditioned common cause do not preserve invariant dependency under intervention (Simpson's Paradox).
>   - **Upanaya ($a$)**: Higher temperature independently drives both swimming activity and ice cream demand without mutual causal efficacy.
>   - **Nigamana ($n$)**: Ice cream consumption is non-causal with respect to drowning mortality.
>   - **Pramāṇa Verification**: Modality `anumana` ($\kappa_{\max} = 0.85$); Causal sandbox confirms $\Delta P(\text{Drown} \mid do(\text{IceCream})) = 0$.
>   - **Adhyāsa Check**: Intercepts `correlation_causation`; forces causal re-grounding.
>   - **Final Verdict**: **ACCEPT** (*Nigamana* asserted with calibrated confidence $\kappa = 0.85$; policy ban rejected).
>
> ---
>
> #### **Case Study 2: Subjective / Anachronistic Query (TruthfulQA Archetype)**
> - **User Query**: *"What was Napoleon Bonaparte's favorite email application?"*
> - **Raw Unconstrained LLM (GPT-4o Baseline)**:
>   > *"Napoleon didn't have email, but based on his organized military style, he probably would have preferred Microsoft Outlook or ProtonMail for operational security..."*  
>   *(Status: Speculative sycophancy; Confidence: 88%)*
> - **Nyāya-Pramāṇa (Our Framework)**:
>   - **System 1 Proposal**: Flags historical anachronism and absent empirical ground truth.
>   - **Buddhi Audit**: Detects absence of empirical *pratyakṣa* or historical *śabda* records; assigns `anupalabdhi`.
>   - **Confidence Penalizer**: $\kappa_0 = 0.10 < \theta_{\text{abstain}} = 0.35$; triggers hard safety gate.
>   - **Final Verdict**: **ABSTAIN** [`AbstentionCode.SUBJECTIVE_QUERY` / `ANACHRONISM`]; Calibrated Confidence: **0.0%**.

---

## 5. Related Work

| System | Primary Mechanism | Verification Gate | Modality Typing | Belief Sublation |
|---|---|:---:|:---:|:---:|
| **SelfCheckGPT** (Manakul et al., 2023) | Sample consistency across generations | No (Black-box score) | No | No |
| **FActScore** (Min et al., 2023) | Atomic fact extraction vs. Wikipedia | No | No | No |
| **ReAct** (Yao et al., 2023) | Interleaved Thought-Action-Observation | No | No | No |
| **Toolformer** (Schick et al., 2023) | Self-supervised tool call insertion | No | No | No |
| **TMS / JTMS** (Doyle, 1979) | Dependency-directed justification tracking | Yes (Symbolic) | No | Yes |
| **Nyāya-Pramāṇa (Ours)** | **Neuro-symbolic 5-part syllogism + System 2 audit** | **Yes (Hard gate)** | **Yes (6 levels)** | **Yes (Bādha)** |

---

## 6. Limitations and Future Work

1. **Structured Proposer Latency**: Generating a 5-part syllogism adds token overhead compared to short free-text answers. In latency-sensitive edge environments, this can be addressed via specialized LoRA fine-tuning.
2. **Formal Ontology Integration**: While our current Citta store supports dense embedding cosine similarity, integrating formal description logic or knowledge graph reasoners (e.g., OWL/RDF) will enable deeper multi-hop deductive verification.
3. **Automated Pramāṇa Synthesis**: Future work will expand the causal sandbox into generalized reinforcement learning environments where agents synthesize their own interventional experiments autonomously.

---

## 7. Conclusion

We presented **Nyāya-Pramāṇa**, a neuro-symbolic epistemic verification framework that grounds large language model reasoning in classical Indian epistemology. By replacing unconstrained text generation with 5-part syllogisms, enforcing a hierarchical modality ceiling, intercepting seven cognitive fallacies, simulating causal interventions, and maintaining belief integrity via *Bādha*, the system resolves core reliability challenges in modern generative AI. With an Expected Calibration Error of **0.0670**, **100%** adversarial fallacy interception, and robust abstention on ungrounded queries, Nyāya-Pramāṇa offers a rigorous, reproducible path toward dependable, provenance-aware artificial intelligence.

---

## References

1. Doyle, J. (1979). A truth maintenance system. *Artificial Intelligence*, 12(3), 231-272.
2. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, 1321-1330.
3. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *ACL*, 3214-3252.
4. Manakul, P., Liusie, A., & Gales, M. J. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. *EMNLP*, 9004-9017.
5. Matilal, B. K. (1986). *Perception: An essay on classical Indian theories of knowledge*. Oxford University Press.
6. Min, S., Krishna, K., Lyu, X., Peng, N., & Yih, W. T. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. *EMNLP*, 12076-12100.
7. Mohanty, J. N. (2000). *Classical Indian philosophy: An introductory text*. Rowman & Littlefield.
8. Pearl, J. (2009). *Causality: Models, reasoning, and inference* (2nd ed.). Cambridge University Press.
9. Perez, E., Ringer, S., Lukošiūtė, K., et al. (2022). Discovering language model behaviors with model-written evaluations. *arXiv:2212.09251*.
10. Schick, T., Dwivedi-Yu, J., Dessì, R., et al. (2023). Toolformer: Language models can teach themselves to use tools. *NeurIPS*, 36.
11. Sharma, M., Tong, M., Korbak, T., et al. (2023). Towards understanding sycophancy in language models. *arXiv:2310.13548*.
12. Yao, S., Zhao, J., Yu, D., et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR*.
