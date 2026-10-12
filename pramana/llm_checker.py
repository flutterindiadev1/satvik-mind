import os
from typing import List, Optional
from pramana.schemas import Claim, AdhyasaLabel
import json
import re

def run_llm_checkers(
    claim: Claim, 
    query: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[AdhyasaLabel]:
    """
    Uses a fast LLM call to check for complex fallacies (Adhyāsa) that 
    regexes cannot catch: FREQUENCY_TRUTH, SOURCE_CONFUSION, ASKER_PROJECTION.

    Supports OpenAI, Anthropic, and Ollama.
    """
    active_provider = (provider or os.environ.get("SATTVIC_LLM_PROVIDER", "openai")).lower()
    
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
    try:
        raw = ""
        if active_provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return []
            client = anthropic.Anthropic(api_key=api_key)
            target_model = model or os.environ.get("SATTVIC_VERIFIER_MODEL", "claude-haiku-4-5-20251001")
            response = client.messages.create(
                model=target_model,
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "["}
                ]
            )
            raw = "[" + response.content[0].text
        elif active_provider == "openai":
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return []
            client = openai.OpenAI(api_key=api_key)
            target_model = model or os.environ.get("SATTVIC_VERIFIER_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=target_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are a fallacy detection assistant. Respond only with a JSON array."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
            )
            raw = response.choices[0].message.content or "[]"
        elif active_provider == "ollama":
            import httpx
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            target_model = model or os.environ.get("SATTVIC_VERIFIER_MODEL", "llama3")
            with httpx.Client(timeout=30) as client:
                resp = client.post(f"{base_url}/api/chat", json={
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0},
                    "format": "json"
                })
                resp.raise_for_status()
                raw = resp.json()["message"]["content"]
        else:
            return []

        match = re.search(r'(\[.*?\])', raw, re.DOTALL)
        if match:
            raw = match.group(1)

        flags = json.loads(raw)
        if not isinstance(flags, list):
            return []

        valid_flags = []
        for f in flags:
            try:
                valid_flags.append(AdhyasaLabel(f))
            except ValueError:
                pass
        return valid_flags

    except Exception as e:
        # Fail open - if the LLM checker fails or times out, return empty list
        return []

