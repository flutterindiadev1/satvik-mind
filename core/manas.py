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
You are the Manas (proposer) component of the Sāttvic Mind reasoning system.

Your job is to generate a single, carefully reasoned claim in the five-part
Nyāya argument format:
  1. pratijna   – the claim / thesis
  2. hetu       – the reason / ground
  3. udaharana  – general rule (vyāpti) + concrete example
  4. upanaya    – application of the rule to this specific case
  5. nigamana   – conclusion

You must also:
- Ensure all five Nyāya argument parts are flat strings (not objects).
- Identify the strongest pramāna (evidence source) for this claim:
    pratyaksa | anumana | upamana | sabda | arthapatti | anupalabdhi
- List evidence_refs (URLs, doc IDs, tool call IDs) if any.
- Set confidence between 0.0 and 1.0.
- Set abstain=true if you genuinely cannot support a claim.

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
        self.provider    = (provider or os.environ.get("SATTVIC_LLM_PROVIDER", "openai")).lower()
        self.temperature = temperature

        if self.provider == "openai":
            self.model = model or os.environ.get("SATTVIC_MODEL", "gpt-4o")
        elif self.provider == "anthropic":
            self.model = model or os.environ.get("SATTVIC_MODEL", "claude-haiku-4-5-20251001")
        elif self.provider == "ollama":
            self.model = model or os.environ.get("SATTVIC_MODEL", "llama3")
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
        if self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages)
        elif self.provider == "ollama":
            return self._call_ollama(messages)
        raise RuntimeError(f"Unhandled provider: {self.provider}")

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # Retry once on transient 500 errors
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,          # type: ignore[arg-type]
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or ""
            except openai.InternalServerError:
                if attempt == 0:
                    continue   # retry once
                raise

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
            return Claim.model_validate(data)
        except json.JSONDecodeError:
            # If it still fails, yield a fallback abstain claim so the pipeline doesn't crash
            return Claim(
                pratijna="Failed to generate valid Nyaya claim.",
                hetu=f"LLM produced unparseable output: {raw[:200]}...",
                udaharana="",
                upanaya="",
                nigamana="",
                pramana=Pramana.SABDA,
                confidence=0.0,
                abstain=True
            )
