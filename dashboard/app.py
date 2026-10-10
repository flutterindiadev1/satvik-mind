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
    page_title="Sāttvic Mind",
    page_icon="🧘",
    layout="wide"
)

# Initialize pipeline in session state so it doesn't reload on every rerun
if "pipeline" not in st.session_state:
    st.session_state.pipeline = Pipeline(PipelineConfig())

st.title("Sāttvic Mind")
st.markdown("### 🧘 An epistemological reasoning engine to verify facts and logical arguments")

# Sidebar for Config & Trace Logs
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("AI Model Provider", ["openai", "anthropic"], index=0)
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.2, 0.1)
    abstain_threshold = st.slider("Uncertainty Threshold", 0.0, 1.0, 0.35, 0.05)
    
    if st.button("Update Settings"):
        st.session_state.pipeline = Pipeline(
            PipelineConfig(
                provider=provider,
                temperature=temperature,
                abstain_threshold=abstain_threshold
            )
        )
        st.success("Settings updated!")
        
    st.divider()
    st.header("🔍 System State")
    st.info("The system continuously monitors itself for hallucinations and confident ignorance.")

tab1, tab2, tab3 = st.tabs(["🗣️ Reasoning Engine", "👁️ Sākṣī (Witness Logs)", "🧠 Citta (Memory)"])

with tab1:
    query = st.text_area("Ask a question:", height=100)
    if st.button("Submit", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Analyzing and verifying..."):
                try:
                    # Run the pipeline
                    result = st.session_state.pipeline.run(query)
                    
                    # Show results based on verdict
                    st.divider()
                    st.subheader("Results")
                    
                    if result.verdict == "abstain":
                        st.error("⚠️ **System Verdict:** ABSTAIN")
                        st.write("I do not have enough evidence to answer this question with high confidence.")
                        st.write("**Reasons:**")
                        for reason in result.reasons:
                            st.write(f"- {reason}")
                        
                    else:
                        st.success("✅ **System Verdict:** VERIFIED")
                        st.markdown("### Answer")
                        st.write(result.output_text)
                        
                        if result.claim:
                            st.divider()
                            st.markdown("### 🏛️ How I Reached This Conclusion")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### The 5-Step Logic")
                                st.markdown(f"**1. Claim:** {result.claim.pratijna}")
                                st.markdown(f"**2. Reason:** {result.claim.hetu}")
                                st.markdown(f"**3. General Rule:** {result.claim.udaharana}")
                                st.markdown(f"**4. Application:** {result.claim.upanaya}")
                                st.markdown(f"**5. Final Conclusion:** {result.claim.nigamana}")
                                
                            with col2:
                                st.markdown("#### Evidence & Validity")
                                st.info(f"**Source Type:** {result.claim.pramana.value.upper()}")
                                st.metric(label="Confidence Level", value=f"{result.claim.confidence:.2f}")
                                
                                if result.claim.evidence_refs:
                                    st.write("**References:**")
                                    for ref in result.claim.evidence_refs:
                                        st.write(f"- {ref}")
                                
                                if result.claim.adhyasa_flags:
                                    st.warning("**Logical Fallacies Detected:**")
                                    for flag in result.claim.adhyasa_flags:
                                        st.write(f"- 🚩 {flag}")
                                else:
                                    st.success("No logical fallacies detected.")
                except Exception as e:
                    st.error(f"Error executing pipeline: {str(e)}")

with tab2:
    st.header("Sākṣī (The Witness)")
    st.info("Sākṣī is the read-only observer. It monitors traces, LLM calls, and internal states without interfering.")
    
    if st.button("Refresh Logs"):
        pass # Streamlit reruns on button click
        
    log_files = sorted(glob.glob("traces/*.jsonl"), reverse=True)
    if not log_files:
        st.write("No trace logs found.")
    else:
        selected_log = st.selectbox("Select Log File", log_files)
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
                
            for run_id, run_logs in runs.items():
                with st.expander(f"Run ID: {run_id[:8]}..."):
                    for entry in run_logs:
                        st.json(entry)

with tab3:
    st.header("Citta (The Memory Store)")
    st.info("Citta stores verified beliefs. If a claim is ACCEPTED by Buddhi, it is deposited here and can be used as context for future reasoning.")
    
    if st.button("Refresh Memory"):
        pass
        
    beliefs = st.session_state.pipeline.citta.get_active_beliefs()
    if not beliefs:
        st.write("Citta is currently empty. Verified beliefs will appear here.")
    else:
        df = pd.DataFrame(beliefs)
        display_cols = ["pratijna", "pramana", "confidence", "origin", "created_at"]
        st.dataframe(df[display_cols], use_container_width=True)
