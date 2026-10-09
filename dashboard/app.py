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
    page_title="Sāttvic Mind",
    page_icon="🧘",
    layout="wide"
)

# Initialize pipeline in session state so it doesn't reload on every rerun
if "pipeline" not in st.session_state:
    st.session_state.pipeline = Pipeline(PipelineConfig())

st.title("Sāttvic Mind")
st.markdown("### 🧘 An AI that Thinks Before It Speaks")

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

# Main Input
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
                    st.write(f"**Reasons:** {'; '.join(result.reasons)}")
                    st.write(result.output_text)
                    
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
