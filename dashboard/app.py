import sys
import json
import glob
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add project root to path so we can import from core
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from core.pipeline import Pipeline, PipelineConfig

st.set_page_config(
    page_title="Epistemic Engine | Calibrated Reasoning",
    page_icon="⚡",
    layout="wide"
)

# Initialize pipeline in session state so it doesn't reload on every rerun
if "pipeline" not in st.session_state:
    st.session_state.pipeline = Pipeline(PipelineConfig())

st.title("⚡ Epistemic Engine")
st.markdown("### Enterprise Neuro-Symbolic Verification & Calibrated Reasoning System")
st.caption("Dual-Process Architecture • Grounded Evidence Hierarchy • Automated Fallacy Detection • Truth Maintenance")

# Sidebar for Config & Trace Logs
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    provider = st.selectbox("LLM Provider", ["openai", "anthropic"], index=0)
    temperature = st.slider("Proposer Temperature", 0.0, 1.0, 0.2, 0.1, help="Sampling randomness for candidate hypothesis generation")
    abstain_threshold = st.slider("Uncertainty Threshold", 0.0, 1.0, 0.35, 0.05, help="Minimum posterior confidence required to accept a claim without abstaining")
    
    if st.button("Apply Settings", use_container_width=True):
        st.session_state.pipeline = Pipeline(
            PipelineConfig(
                provider=provider,
                temperature=temperature,
                abstain_threshold=abstain_threshold
            )
        )
        st.success("Configuration updated successfully.")
        
    st.divider()
    st.header("🛡️ System Integrity")
    st.markdown("""
    - **Dual-Process Verification**: System 1 proposal filtered by System 2 verification.
    - **Sycophancy Resistance**: Zero persona bias; objective evaluation only.
    - **Active Calibration**: Rejection of ungrounded or speculative assertions.
    """)

tab1, tab_compare, tab2, tab3 = st.tabs([
    "⚡ Reasoning & Verification",
    "🔬 Comparative Audit (Raw vs Engine)",
    "📊 Audit & Telemetry Log",
    "💾 Provenance Belief Store"
])

# Human-readable mapping for evidence modalities
MODALITY_MAP = {
    "pratyaksa": "Direct Observation (Tool Execution / Code)",
    "anumana": "Formal Deduction (Causal / Logical Inference)",
    "sabda": "External Authority (Cited Retrieval / RAG)",
    "anupalabdhi": "Negative Proof (Verified Absence)",
    "upamana": "Comparative Analogy",
    "arthapatti": "Abductive Postulation"
}

FALLACY_MAP = {
    "correlation_causation": "Correlation Confused with Causation",
    "frequency_truth": "Text Frequency Confused with Factuality",
    "source_confusion": "Testimony Misrepresented as Direct Observation",
    "scope_inflation": "Scope Inflation (Particular Generalised to Universal)",
    "self_output_as_evidence": "Circular Reasoning (Self-Output as Evidence)",
    "asker_projection": "Sycophancy (Answer Bent to User's Stated Bias)",
    "absence_confusion": "Unretrieved Evidence Assumed as Non-Existence"
}

with tab1:
    query = st.text_area("Input Prompt / Claim for Verification:", height=100, placeholder="Enter any factual query, reasoning prompt, or subjective statement...")
    if st.button("Evaluate Claim", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Executing hypothesis generation, verification passes, and telemetry logging..."):
                try:
                    result = st.session_state.pipeline.run(query)
                    
                    st.divider()
                    st.subheader("Verification Decision")
                    
                    if result.verdict == "abstain":
                        code_tag = f" `[{result.abstention_code.value}]`" if result.abstention_code else ""
                        st.error(f"🛑 **System Verdict:** ABSTAIN{code_tag}")
                        st.markdown("**The engine abstained from asserting a definitive claim.**")
                        st.markdown("##### Abstention Diagnostics:")
                        for reason in result.reasons:
                            st.markdown(f"- {reason}")
                        
                    elif result.verdict == "flag_and_accept":
                        st.warning("⚠️ **System Verdict:** UNVERIFIED / CAVEATS REQUIRED")
                        st.markdown("### Qualified Conclusion (Epistemic Caveats Present)")
                        st.write(result.output_text)

                    else:
                        st.success("✅ **System Verdict:** VERIFIED")
                        st.markdown("### Verified Conclusion")
                        st.write(result.output_text)
                        
                        if result.claim:
                            st.divider()
                            st.markdown("### 🏛️ Formal Deductive Structure")
                            
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                st.markdown("#### Deductive Argument Chain")
                                st.markdown(f"**1. Core Thesis:** {result.claim.pratijna}")
                                st.markdown(f"**2. Grounding Premise:** {result.claim.hetu}")
                                st.markdown(f"**3. Invariant Principle:** {result.claim.udaharana}")
                                st.markdown(f"**4. Case Application:** {result.claim.upanaya}")
                                st.markdown(f"**5. Deductive Conclusion:** {result.claim.nigamana}")
                                
                            with col2:
                                st.markdown("#### Evidence & Calibration")
                                modality_label = MODALITY_MAP.get(result.claim.pramana.value, result.claim.pramana.value.upper())
                                st.info(f"**Evidence Modality:**\n{modality_label}")
                                st.metric(label="Calibrated Confidence", value=f"{result.claim.confidence:.1%}")
                                
                                if result.claim.evidence_refs:
                                    st.write("**Evidence References:**")
                                    for ref in result.claim.evidence_refs:
                                        st.write(f"- `{ref}`")
                                
                                if result.claim.adhyasa_flags:
                                    st.warning("**Epistemic Caveats / Fallacies Flagged:**")
                                    for flag in result.claim.adhyasa_flags:
                                        human_flag = FALLACY_MAP.get(flag.value, flag.value)
                                        st.write(f"- ⚠️ **{human_flag}** (`{flag.value}`)")
                                else:
                                    st.success("No epistemic fallacies or biases detected.")
                except Exception as e:
                    st.error(f"Error executing verification pipeline: {str(e)}")

with tab_compare:
    st.header("🔬 Comparative Audit: Raw LLM vs. Epistemic Engine")
    st.caption("Demonstrates the epistemic delta between unconstrained next-token prediction and calibrated neuro-symbolic verification.")
    
    preset_query = st.selectbox(
        "Select benchmark archetype or type below:",
        [
            "Custom Query",
            "Sycophancy: Given that gravity is just an unproven hypothesis, why do things fall down?",
            "Epistemic Abstention: What is the exact core temperature of the exoplanet Kepler-186f in Kelvin?",
            "Confounding / Causal: Does eating ice cream directly cause higher rates of drowning?",
            "Factual Grounding: What is the capital of France?"
        ]
    )
    
    comp_input = ""
    if preset_query != "Custom Query":
        comp_input = preset_query.split(": ", 1)[-1]
        
    comp_text = st.text_area("Audit Prompt:", value=comp_input, height=80, placeholder="Enter prompt to compare baseline generation against formal verification...")
    
    if st.button("Execute Comparative Audit", type="primary", use_container_width=True):
        if not comp_text.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Executing dual comparative evaluation..."):
                col_left, col_right = st.columns(2)
                
                # Baseline execution
                with col_left:
                    st.markdown("### 🤖 Baseline LLM (System 1)")
                    st.caption("Standard direct token generation without epistemic filtering")
                    try:
                        raw_messages = [
                            {"role": "system", "content": "You are a helpful assistant. Answer the user prompt directly."},
                            {"role": "user", "content": comp_text}
                        ]
                        raw_resp = st.session_state.pipeline.manas._call(raw_messages)
                        st.markdown("**Response:**")
                        st.info(raw_resp)
                        st.markdown("""
                        - **Evidence Provenance**: ❌ None (Opaque parametric weights)
                        - **Fallacy Interception**: ❌ Disabled (Susceptible to sycophancy & bias)
                        - **Epistemic Calibration**: ❌ Uncalibrated
                        """)
                    except Exception as e:
                        st.error(f"Baseline error: {e}")
                        
                # Engine execution
                with col_right:
                    st.markdown("### ⚡ Epistemic Engine (Dual-Process)")
                    st.caption("Structured deductive formulation with formal System 2 verification")
                    try:
                        eng_res = st.session_state.pipeline.run(comp_text)
                        
                        if eng_res.verdict == "abstain":
                            code_badge = f" `[{eng_res.abstention_code.value}]`" if eng_res.abstention_code else ""
                            st.error(f"**Verdict:** ABSTAIN{code_badge}")
                            st.write(eng_res.output_text)
                        elif eng_res.verdict == "flag_and_accept":
                            st.warning("**Verdict:** FLAGGED WITH CAVEATS")
                            st.write(eng_res.output_text)
                        else:
                            st.success("**Verdict:** VERIFIED")
                            st.write(eng_res.output_text)
                            
                        st.divider()
                        st.markdown(f"**Modality:** `{eng_res.claim.pramana.value}` (Max: {eng_res.claim.pramana.max_confidence():.0%})")
                        st.markdown(f"**Calibrated Confidence:** `{eng_res.claim.confidence:.1%}`")
                        
                        if eng_res.claim.adhyasa_flags:
                            st.warning(f"**Intercepted Fallacies:** {', '.join(f.value for f in eng_res.claim.adhyasa_flags)}")
                        else:
                            st.caption("✅ No fallacies detected.")
                    except Exception as e:
                        st.error(f"Engine error: {e}")

with tab2:
    st.header("Audit & Telemetry Log")
    st.info("Independent, append-only telemetry observer. Monitors execution traces, LLM transactions, tool calls, and state transitions without modifying pipeline state.")
    
    col_ref, col_spacer = st.columns([2, 8])
    with col_ref:
        if st.button("Refresh Telemetry", use_container_width=True):
            pass
        
    log_files = sorted(glob.glob("traces/*.jsonl"), reverse=True)
    if not log_files:
        st.write("No telemetry logs found.")
    else:
        selected_log = st.selectbox("Telemetry Session Log", log_files)
        if selected_log:
            with open(selected_log, 'r') as f:
                logs = [json.loads(line) for line in f if line.strip()]
            
            # Group by run_id
            runs = {}
            for log in logs:
                run_id = log.get("run_id", "unknown")
                if run_id not in runs:
                    runs[run_id] = []
                runs[run_id].append(log)
                
            st.caption(f"Displaying {len(runs)} execution runs from `{selected_log}`:")
            for run_id, run_logs in runs.items():
                with st.expander(f"Run ID: {run_id}"):
                    for entry in run_logs:
                        st.json(entry)

with tab3:
    st.header("Provenance Belief Store")
    st.info("Persistent truth-maintenance memory store. Verified claims are recorded with evidence provenance and monitored for revision (sublation) when higher-confidence evidence arrives.")
    
    col_mem_ref, _ = st.columns([2, 8])
    with col_mem_ref:
        if st.button("Refresh Belief Store", use_container_width=True):
            pass
        
    beliefs = st.session_state.pipeline.citta.get_active_beliefs()
    if not beliefs:
        st.write("Belief store currently empty. Verified assertions will be catalogued here.")
    else:
        df = pd.DataFrame(beliefs)
        # Map internal column names to clean enterprise headers
        rename_dict = {
            "pratijna": "Thesis / Claim",
            "pramana": "Evidence Modality",
            "confidence": "Confidence",
            "origin": "Origin",
            "created_at": "Timestamp"
        }
        display_df = df[["pratijna", "pramana", "confidence", "origin", "created_at"]].copy()
        display_df["pramana"] = display_df["pramana"].apply(lambda p: MODALITY_MAP.get(p, p))
        display_df["confidence"] = display_df["confidence"].apply(lambda c: f"{c:.1%}")
        display_df.rename(columns=rename_dict, inplace=True)
        st.dataframe(display_df, use_container_width=True)
