import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add project root to path so we can import from core
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from core.pipeline import Pipeline, PipelineConfig

st.set_page_config(
    page_title="Sāttvic Mind Dashboard",
    page_icon="🧘",
    layout="wide"
)

# Initialize pipeline in session state so it doesn't reload on every rerun
if "pipeline" not in st.session_state:
    st.session_state.pipeline = Pipeline(PipelineConfig())

st.title("Sāttvic Mind (Advaita Vedānta Edition)")
st.markdown("### 🧘 AI Reasoning System Grounded in Epistemology")

# Sidebar for Config & Trace Logs
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("LLM Provider", ["openai", "anthropic"], index=0)
    temperature = st.slider("Temperature (Manas)", 0.0, 1.0, 0.2, 0.1)
    abstain_threshold = st.slider("Abstain Threshold", 0.0, 1.0, 0.35, 0.05)
    
    if st.button("Update Configuration"):
        st.session_state.pipeline = Pipeline(
            PipelineConfig(
                provider=provider,
                temperature=temperature,
                abstain_threshold=abstain_threshold
            )
        )
        st.success("Configuration updated!")
        
    st.divider()
    st.header("🔍 System State")
    st.info("The **Sākṣī** (Witness) is monitoring for Tamas and Rajas behaviors.")

# Main Input
query = st.text_area("Ask a question to Manas (Proposer):", height=100)
if st.button("Query Sāttvic Mind", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing through the Antaḥkaraṇa loop (Manas ➔ Buddhi ➔ Citta)..."):
            try:
                # Run the pipeline
                result = st.session_state.pipeline.run(query)
                
                # Show results based on verdict
                st.divider()
                st.subheader("Results")
                
                if result.verdict == "abstain":
                    st.error("⚠️ **Buddhi Verdict:** ABSTAIN")
                    st.write(f"**Reason:** {result.reasoning}")
                    st.write(result.output_text)
                    
                else:
                    st.success("✅ **Buddhi Verdict:** ACCEPT")
                    st.markdown("### Answer")
                    st.write(result.output_text)
                    
                    if result.claim:
                        st.divider()
                        st.markdown("### 🏛️ Epistemological Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### The 5-Step Argument (Nyāya)")
                            st.markdown(f"**1. Pratijñā (Claim):** {result.claim.pratijna}")
                            st.markdown(f"**2. Hetu (Reason):** {result.claim.hetu}")
                            st.markdown(f"**3. Udāharaṇa (Example/Rule):** {result.claim.udaharana}")
                            st.markdown(f"**4. Upanaya (Application):** {result.claim.upanaya}")
                            st.markdown(f"**5. Nigamana (Conclusion):** {result.claim.nigamana}")
                            
                        with col2:
                            st.markdown("#### Evidence & Validity")
                            st.info(f"**Pramāṇa (Means of Knowledge):** {result.claim.pramana.value.upper()}")
                            st.metric(label="Confidence", value=f"{result.claim.confidence:.2f}")
                            
                            if result.claim.evidence_refs:
                                st.write("**Evidence References:**")
                                for ref in result.claim.evidence_refs:
                                    st.write(f"- {ref}")
                            
                            if result.claim.adhyasa_flags:
                                st.warning("**Adhyāsa (Superimposition) Flags Detected:**")
                                for flag in result.claim.adhyasa_flags:
                                    st.write(f"- 🚩 {flag}")
                            else:
                                st.success("No Adhyāsa (Fallacies) detected.")
            except Exception as e:
                st.error(f"Error executing pipeline: {str(e)}")
