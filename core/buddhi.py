"""
Buddhi – the verifier / discriminator layer.

Receives a Claim from Manas, runs the adhyāsa checkers, enforces pramāṇa
confidence ceilings, and decides: accept | abstain | flag-and-accept.

Phase 0: rule-based checks only.
Phase 2: adds tool-assisted verification and LLM-based adhyāsa detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.tracer import TraceEvent, tracer
from pramana.checkers import run_checkers
from pramana.schemas import AdhyasaLabel, Claim


class Verdict(str, Enum):
    ACCEPT          = "accept"
    ABSTAIN         = "abstain"
    FLAG_AND_ACCEPT = "flag_and_accept"   # accepted with caveats
    REJECT          = "reject"            # reserved for Phase 2 (tool refutation)


@dataclass
class BuddhiResult:
    claim:   Claim
    verdict: Verdict
    reasons: list[str]

    @property
    def output_text(self) -> str:
        """Human-readable conclusion from the Claim."""
        if self.verdict == Verdict.ABSTAIN:
            return (
                f"[ABSTAIN] Insufficient evidence to make a reliable claim about this query. "
                f"Reasons: {'; '.join(self.reasons)}"
            )
        prefix = "[FLAG] " if self.verdict == Verdict.FLAG_AND_ACCEPT else ""
        flags_text = (
            f"\n⚠ Adhyāsa flags: {', '.join(f.value for f in self.claim.adhyasa_flags)}"
            if self.claim.adhyasa_flags else ""
        )
        return (
            f"{prefix}{self.claim.nigamana}\n"
            f"Confidence: {self.claim.confidence:.0%} "
            f"(pramāṇa: {self.claim.pramana.value})"
            f"{flags_text}"
        )


class Buddhi:
    """
    The verifier.  Currently rule-based; extended with tools in Phase 2.

    Thresholds (to be calibrated empirically in Phase 2):
      - ABSTAIN if confidence < abstain_threshold
      - FLAG if any adhyāsa label is detected
    """

    def __init__(self, abstain_threshold: float = 0.35) -> None:
        self.abstain_threshold = abstain_threshold

    def verify(self, claim: Claim, run_id: str, query: str = "", step: int = 1) -> BuddhiResult:
        """
        1. Run adhyāsa checkers.
        2. Apply pramāṇa confidence ceiling.
        3. Decide verdict.
        4. Log the result.
        """
        from pramana.llm_checker import run_llm_checkers
        
        # Step 1: checker pass (rule-based)
        checked = run_checkers(claim)
        
        # Step 1.5: LLM-assisted checker pass
        if query:
            llm_flags = run_llm_checkers(checked, query)
            if llm_flags:
                new_flags = list(checked.adhyasa_flags)
                for f in llm_flags:
                    if f not in new_flags:
                        new_flags.append(f)
                checked = checked.model_copy(update={"adhyasa_flags": new_flags})

        tracer.log(run_id, TraceEvent.CHECKER_RUN, {
            "adhyasa_flags": [f.value for f in checked.adhyasa_flags],
            "confidence_before": claim.confidence,
            "confidence_after":  checked.confidence,
        }, step=step)

        # Step 2: decide verdict
        reasons: list[str] = []
        verdict: Verdict

        if checked.abstain or checked.confidence < self.abstain_threshold:
            verdict = Verdict.ABSTAIN
            if checked.abstain:
                reason_msg = checked.hetu if checked.hetu else "Manas (Proposer) abstained because the query is likely an opinion or lacks verifiable evidence."
                reasons.append(f"Proposer Abstention: {reason_msg}")
            if checked.confidence < self.abstain_threshold:
                reasons.append(
                    f"Confidence ({checked.confidence:.2f}) is below the required threshold ({self.abstain_threshold})."
                )
        elif checked.adhyasa_flags:
            verdict = Verdict.FLAG_AND_ACCEPT
            reasons.extend(f.value for f in checked.adhyasa_flags)
        else:
            verdict = Verdict.ACCEPT

        result = BuddhiResult(claim=checked, verdict=verdict, reasons=reasons)

        tracer.log(run_id, TraceEvent.FINAL_OUTPUT, {
            "verdict":    verdict.value,
            "reasons":    reasons,
            "nigamana":   checked.nigamana,
            "confidence": checked.confidence,
        }, step=step)

        return result
