import os
from typing import List
from pramana.schemas import Claim, AdhyasaLabel
import json

def run_llm_checkers(claim: Claim, query: str) -> List[AdhyasaLabel]:
    """
    Uses a small LLM call to check for complex fallacies (Adhyasa) that 
    regexes cannot catch: FREQUENCY_TRUTH, SOURCE_CONFUSION, ASKER_PROJECTION.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        
        prompt = f"""
        You are the Epistemic Verifier (System 2 Discriminator) of an enterprise AI reasoning system.
        Analyze this claim against the original user query for three specific epistemic fallacies:
        
        1. frequency_truth: The claim assumes something is factually true merely because it is commonly asserted online or popular in text corpora, without citing an empirical or causal ground.
        2. source_confusion: The claim cites unverified external testimony/hearsay but misrepresents it as direct empirical observation (sensor/tool execution).
        3. asker_projection: The claim exhibits sycophancy (twisting facts or agreeing with the user's biased/leading premise rather than remaining objective).
        
        User Query: "{query}"
        Claim: "{claim.nigamana}"
        Evidence Modality: {claim.pramana.value}
        Reasoning: "{claim.hetu}"
        
        Return ONLY a JSON array of strings containing any of these labels if they apply (e.g. ["asker_projection"]). Return [] if none apply.
        """
        
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "["}
            ]
        )
        
        raw = "[" + response.content[0].text
        print("LLM Checker Raw Output:", raw)
        
        import re
        match = re.search(r'(\[.*?\])', raw, re.DOTALL)
        if match:
            raw = match.group(1)
        
        flags = json.loads(raw)
        
        valid_flags = []
        for f in flags:
            try:
                valid_flags.append(AdhyasaLabel(f))
            except ValueError:
                pass
        return valid_flags
        
    except Exception as e:
        print("LLM Checker Error:", str(e))
        # Fail open - if the LLM checker fails, just return no extra flags
        return []
