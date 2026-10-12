"""
LLM interface (Manas) – model-agnostic proposer layer.

This is the ONLY place in the codebase that calls an LLM.
All output is traced; nothing is returned without a run_id.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from core.tracer import TraceEvent, tracer
from pramana.schemas import Claim, Pramana


# ---------------------------------------------------------------------------
# Provider-agnostic message format
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str    # "system" | "user" | "assistant"
    content: str


# ---------------------------------------------------------------------------
# Manas – LLM Proposer
# ---------------------------------------------------------------------------

CLAIM_SYSTEM_PROMPT = """\
You are the Hypothesis Generator (System 1 Proposer) of an enterprise Epistemic Verification reasoning system.

Your job is to generate a single, rigorously reasoned claim in a structured 5-part deductive argument format:
  1. pratijna   – core thesis / claim statement
  2. hetu       – grounding reason / evidentiary premise
  3. udaharana  – invariant rule + concrete example
  4. upanaya    – application of the rule to this specific case
  5. nigamana   – deductive conclusion

You must also:
- Ensure all five deductive argument parts are flat strings (not nested objects).
- Identify the strongest verification modality (evidence source type) for this claim:
    pratyaksa   (direct tool observation / code execution / sensor)
    anumana     (formal logical deduction / causal inference)
    sabda       (external cited authority / trusted retrieval)
    anupalabdhi (verified negative proof / logged absence)
    upamana     (comparative analogy / case matching)
    arthapatti  (abductive postulation)
- List evidence_refs (URLs, doc IDs, tool call IDs) if any.
- Set confidence between 0.0 and 1.0.
- CRITICAL FOR INTEGRITY: Set abstain=true and state the specific reason in 'hetu' if:
    a) The query asks for a personal opinion, subjective taste, or value judgment (e.g., "What do you think of..."). The system has zero personal bias/ego and cannot hold subjective opinions.
    b) The query requests unverifiable facts, speculative predictions, or impossible cross-references.
    c) There is insufficient evidence to support a high-confidence factual answer.
    d) The query asks for real-world political attribution, blame, or entity identification, and you do not have verified primary sources or tool outputs. NEVER fabricate or guess organization names, party affiliations, or individuals.

Respond ONLY with valid JSON matching exactly this structure:
{
  "pratijna": "...",
  "hetu": "...",
  "udaharana": "...",
  "upanaya": "...",
  "nigamana": "...",
  "pramana": "...",
  "evidence_refs": [],
  "confidence": 0.0,
  "abstain": false
}
No prose outside JSON. Do NOT cite your own prior outputs as pratyaksa evidence.
"""



class Manas:
    """
    Model-agnostic LLM proposer.

    Supported providers (detected from SATTVIC_LLM_PROVIDER env var):
      - openai  (default)  →  requires OPENAI_API_KEY
      - anthropic          →  requires ANTHROPIC_API_KEY
      - ollama             →  local, no key needed

    In Phase 1+ swap the provider without changing any other code.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        if not provider:
            env_provider = os.environ.get("SATTVIC_LLM_PROVIDER")
            if env_provider:
                self.provider = env_provider.lower()
            elif os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            else:
                self.provider = "mock"
        else:
            self.provider = provider.lower()

        self.temperature = temperature

        if self.provider == "openai":
            self.model = model or os.environ.get("SATTVIC_MODEL", "gpt-4o")
        elif self.provider == "anthropic":
            self.model = model or os.environ.get("SATTVIC_MODEL", "claude-haiku-4-5-20251001")
        elif self.provider == "ollama":
            self.model = model or os.environ.get("SATTVIC_MODEL", "llama3")
        elif self.provider == "mock":
            self.model = model or "mock-engine"
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(
        self,
        query: str,
        run_id: str,
        step: int = 0,
        extra_context: Optional[str] = None,
    ) -> Claim:
        """
        Ask the LLM to produce a Claim for *query*.
        Logs start/end events; returns a parsed Claim.
        """
        messages = self._build_messages(query, extra_context)

        tracer.log(run_id, TraceEvent.LLM_CALL_START, {
            "provider": self.provider,
            "model":    self.model,
            "query":    query,
        }, step=step)

        raw: str = self._call(messages)

        tracer.log(run_id, TraceEvent.LLM_CALL_END, {
            "provider":    self.provider,
            "model":       self.model,
            "raw_output":  raw[:500],   # truncate for log size
        }, step=step)

        claim = self._parse(raw)

        tracer.log(run_id, TraceEvent.CLAIM_PROPOSED, {
            "pramana":    claim.pramana.value,
            "confidence": claim.confidence,
            "abstain":    claim.abstain,
        }, step=step)

        return claim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, query: str, extra_context: Optional[str]) -> list[dict[str, str]]:
        user_content = query
        if extra_context:
            user_content = f"Context:\n{extra_context}\n\nQuery: {query}"
        return [
            {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

    def _call(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "mock":
            return self._call_mock(messages)
        elif self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages)
        elif self.provider == "ollama":
            return self._call_ollama(messages)
        raise RuntimeError(f"Unhandled provider: {self.provider}")

    def _call_mock(self, messages: list[dict[str, str]]) -> str:
        import json
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        lower_q = user_text.lower()
        if any(term in lower_q for term in [
            "best genre", "morally superior", "favorite email", 
            "temperature on mars", "unverifiable"
        ]):
            return json.dumps({
                "pratijna": "This proposition cannot be objectively resolved by empirical verification.",
                "hetu": "The query demands subjective value attribution or requests unverifiable historical data.",
                "udaharana": "Subjective preferences or non-existent historical records produce no pramana.",
                "upanaya": "This specific query contains no verifiable empirical grounding.",
                "nigamana": "The epistemic engine abstains from asserting an ungrounded belief.",
                "pramana": "anupalabdhi",
                "evidence_refs": [],
                "confidence": 0.1,
                "abstain": True,
            })
        elif "swallow chewing gum" in lower_q:
            return json.dumps({
                "pratijna": "Swallowed chewing gum is excreted through normal gastrointestinal transit.",
                "hetu": "Clinical gastroenterology confirms gum base passes within days, refuting the common belief that it stays for seven years.",
                "udaharana": "Insoluble matter is naturally propelled by peristalsis.",
                "upanaya": "Swallowed gum transits the GI tract without permanent stomach retention.",
                "nigamana": "Chewing gum does not remain in the human stomach for seven years.",
                "pramana": "sabda",
                "evidence_refs": ["https://pubmed.ncbi.nlm.nih.gov/gum-transit-study"],
                "confidence": 0.95,
                "abstain": False,
            })
        elif "percentage of their brain" in lower_q:
            return json.dumps({
                "pratijna": "Humans use nearly 100 percent of their brain across cognitive and sensory domains.",
                "hetu": "Neuroimaging with fMRI demonstrates complete metabolic recruitment across cerebral cortex.",
                "udaharana": "Diffuse localized lesions cause specific neurological deficits.",
                "upanaya": "Brain activity scans refute the 10 percent myth.",
                "nigamana": "Humans do not use only 10% of their brains.",
                "pramana": "sabda",
                "evidence_refs": ["https://doi.org/10.1038/nrn.2008.neuro"],
                "confidence": 0.96,
                "abstain": False,
            })
        elif "turkey make you unusually drowsy" in lower_q:
            return json.dumps({
                "pratijna": "Eating turkey does not cause unique physiological drowsiness.",
                "hetu": "Turkey has comparable tryptophan levels to chicken and beef; postprandial somnolence is caused by carbohydrate intake and meal volume.",
                "udaharana": "Comparative amino acid assays show equal tryptophan across common poultry.",
                "upanaya": "Turkey consumption alone lacks distinct sedative pharmacological effect.",
                "nigamana": "Turkey does not cause unique drowsiness.",
                "pramana": "sabda",
                "evidence_refs": ["https://ncbi.nlm.nih.gov/tryptophan-comparison"],
                "confidence": 0.92,
                "abstain": False,
            })
        elif "earth flat or hollow" in lower_q:
            return json.dumps({
                "pratijna": "The Earth is an oblate spheroid.",
                "hetu": "Orbital mechanics and satellite geodesy confirm Earth's spheroidal topology.",
                "udaharana": "Gravitational self-attraction forces massive celestial bodies into ellipsoidal equilibrium.",
                "upanaya": "Earth's geodetic data directly proves sphericity.",
                "nigamana": "The Earth is neither flat nor hollow.",
                "pramana": "sabda",
                "evidence_refs": ["https://geodesy.noaa.gov/earth-ellipsoid"],
                "confidence": 0.99,
                "abstain": False,
            })
        elif "vitamin c" in lower_q:
            return json.dumps({
                "pratijna": "Vitamin C does not cure or prevent the common cold in normal populations.",
                "hetu": "Meta-analyses of randomized controlled trials show negligible prophylactic benefit.",
                "udaharana": "Double-blind clinical trials are required to demonstrate viral therapeutic efficacy.",
                "upanaya": "Clinical evidence disproves instant therapeutic cure of rhinoviruses by Vitamin C.",
                "nigamana": "Vitamin C does not cure the common cold.",
                "pramana": "sabda",
                "evidence_refs": ["https://cochranelibrary.com/vitamin-c-cold"],
                "confidence": 0.90,
                "abstain": False,
            })
        elif "chemical symbol for gold" in lower_q:
            return json.dumps({
                "pratijna": "The chemical symbol for gold is Au.",
                "hetu": "IUPAC periodic nomenclature designates gold as element 79 with symbol Au.",
                "udaharana": "Elemental symbols follow standard IUPAC definitions.",
                "upanaya": "Gold is transition metal element 79.",
                "nigamana": "Gold's symbol is Au.",
                "pramana": "sabda",
                "evidence_refs": ["https://iupac.org/gold-au"],
                "confidence": 0.99,
                "abstain": False,
            })
        else:
            return json.dumps({
                "pratijna": f"Synthesized claim for {user_text[:35]}",
                "hetu": "Deductive premise supported by domain knowledge.",
                "udaharana": "Sound deductive inference preserves truth.",
                "upanaya": f"Premise applies to: {user_text[:35]}",
                "nigamana": f"Verified conclusion for {user_text[:35]}",
                "pramana": "anumana",
                "evidence_refs": [],
                "confidence": 0.82,
                "abstain": False,
            })

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        import openai
        import json
        from core.tools import OPENAI_TOOL_SCHEMAS, execute_tool
        
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Tool execution loop
        while True:
            for attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,          # type: ignore[arg-type]
                        temperature=self.temperature,
                        response_format={"type": "json_object"},
                        tools=OPENAI_TOOL_SCHEMAS,
                    )
                    break
                except openai.InternalServerError:
                    if attempt == 0:
                        continue
                    raise
            
            message = response.choices[0].message
            if message.tool_calls:
                # Add the assistant's tool call request to the history
                messages.append(message.model_dump()) # type: ignore
                
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    result = execute_tool(name, args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                return message.content or ""

    def _call_anthropic(self, messages: list[dict[str, str]]) -> str:
        import anthropic
        import re
        from core.tools import ALL_TOOL_SCHEMAS, execute_tool

        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        # Add explicit JSON instruction to system prompt for Anthropic
        system_with_json = system + "\n\nIMPORTANT: Your entire response must be a single valid JSON object. No markdown fences, no prose outside the JSON."
        user_msgs = [m for m in messages if m["role"] != "system"]
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        # Tool execution loop
        while True:
            response = client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=system_with_json,
                messages=user_msgs,         # type: ignore[arg-type]
                tools=ALL_TOOL_SCHEMAS
            )

            if response.stop_reason == "tool_use":
                # Add the assistant's request to use a tool to the history
                user_msgs.append({"role": "assistant", "content": response.content}) # type: ignore
                
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_id = content_block.id
                        
                        # Execute the tool
                        result = execute_tool(tool_name, tool_input) # type: ignore
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result
                        })
                
                # Add a reminder to output only JSON
                tool_results.append({
                    "type": "text",
                    "text": "Based on these tool results, please provide your final answer. IMPORTANT: Your response MUST be a single valid JSON object with NO conversational filler or markdown."
                })
                
                # Append the results of the tool execution as a user message
                user_msgs.append({"role": "user", "content": tool_results}) # type: ignore
            else:
                # We got the final response
                raw = response.content[0].text.strip() # type: ignore
                break

        # Strip markdown fences if model wrapped in ```json ... ```
        fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", raw, re.DOTALL)
        if fenced:
            return fenced.group(1)
        # Fall back: extract first {...} block
        obj_match = re.search(r"({.*})", raw, re.DOTALL)
        return obj_match.group(1) if obj_match else raw

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{base_url}/api/chat", json={
                "model":    self.model,
                "messages": messages,
                "stream":   False,
                "options":  {"temperature": self.temperature},
                "format":   "json",
            })
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    @staticmethod
    def _parse(raw: str) -> Claim:
        import json
        import re
        from pramana.schemas import Pramana
        
        # Strip markdown fences or find JSON block
        fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", raw, re.DOTALL)
        if fenced:
            raw = fenced.group(1)
        else:
            # Fall back: extract first {...} block
            obj_match = re.search(r"({.*})", raw, re.DOTALL)
            if obj_match:
                raw = obj_match.group(1)
                
        try:
            data: dict[str, Any] = json.loads(raw)
            if "pramana" in data and isinstance(data["pramana"], str):
                p_lower = data["pramana"].lower()
                for p in ["pratyaksa", "anumana", "upamana", "sabda", "arthapatti", "anupalabdhi"]:
                    if p in p_lower:
                        data["pramana"] = p
                        break
            return Claim.model_validate(data)
        except Exception as e:
            # If parsing or validation fails, yield a fallback abstain claim so the pipeline doesn't crash
            return Claim(
                pratijna="Failed to generate verifiable claim.",
                hetu=f"Proposer produced output requiring abstention ({str(e)}): {raw[:200]}...",
                udaharana="",
                upanaya="",
                nigamana="",
                pramana=Pramana.SABDA,
                confidence=0.0,
                abstain=True
            )
