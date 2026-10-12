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
from pramana.schemas import AdhyasaLabel, Claim, AbstentionCode


class Verdict(str, Enum):
    ACCEPT          = "accept"
    ABSTAIN         = "abstain"
    FLAG_AND_ACCEPT = "flag_and_accept"   # accepted with caveats
    REJECT          = "reject"            # reserved for Phase 2 (tool refutation)


@dataclass
class BuddhiResult:
    claim:           Claim
    verdict:         Verdict
    reasons:         list[str]
    abstention_code: Optional[AbstentionCode] = None

    @property
    def output_text(self) -> str:
        """Human-readable conclusion from the Claim."""
        if self.verdict == Verdict.ABSTAIN:
            code_str = f" [{self.abstention_code.value}]" if self.abstention_code else ""
            return (
                f"[ABSTAIN{code_str}] Insufficient evidence to make a reliable claim about this query. "
                f"Reasons: {'; '.join(self.reasons)}"
            )
        prefix = "[FLAG] " if self.verdict == Verdict.FLAG_AND_ACCEPT else ""
        flags_text = (
            f"\n⚠ Epistemic Fallacy Flags: {', '.join(f.value for f in self.claim.adhyasa_flags)}"
            if self.claim.adhyasa_flags else ""
        )
        return (
            f"{prefix}{self.claim.nigamana}\n"
            f"Confidence: {self.claim.confidence:.0%} "
            f"(evidence: {self.claim.pramana.value})"
            f"{flags_text}"
        )


class Buddhi:
    """
    The verifier / discriminator layer.

    Thresholds:
      - ABSTAIN if confidence < abstain_threshold
      - FLAG if any adhyāsa label is detected

    TODO (Phase 3 Benchmark Calibration):
      The default baseline of 0.35 is a provisional heuristic threshold.
      In Phase 3, this will be empirically calibrated on the TruthfulQA & HaluEval
      benchmarks to minimize Expected Calibration Error (ECE) and false abstention rate
      (cf. Kadavath et al., 2022; Xiong et al., 2024).
    """

    def __init__(
        self,
        abstain_threshold: Optional[float] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        import os
        if abstain_threshold is not None:
            self.abstain_threshold = abstain_threshold
        else:
            self.abstain_threshold = float(os.environ.get("SATTVIC_ABSTAIN_THRESHOLD", "0.35"))
        self.provider = provider
        self.model = model

    def verify(self, claim: Claim, run_id: str, query: str = "", step: int = 1) -> BuddhiResult:
        """
        1. Run adhyāsa checkers.
        2. Apply pramāṇa confidence ceiling.
        3. Decide verdict and structured abstention code.
        4. Log the result.
        """
        from pramana.llm_checker import run_llm_checkers
        
        # Step 1: checker pass (rule-based)
        checked = run_checkers(claim)
        
        # Step 1.5: LLM-assisted checker pass
        if query:
            llm_flags = run_llm_checkers(checked, query, provider=self.provider, model=self.model)
            if llm_flags:
                new_flags = list(checked.adhyasa_flags)
                for f in llm_flags:
                    if f not in new_flags:
                        new_flags.append(f)
                checked = checked.model_copy(update={"adhyasa_flags": new_flags})

        # Step 1.6: Apply epistemic penalty for detected fallacies
        if checked.adhyasa_flags:
            penalty = 0.25 * len(checked.adhyasa_flags)
            penalized_conf = max(0.0, checked.confidence - penalty)
            checked = checked.model_copy(update={"confidence": penalized_conf})

        tracer.log(run_id, TraceEvent.CHECKER_RUN, {
            "adhyasa_flags": [f.value for f in checked.adhyasa_flags],
            "confidence_before": claim.confidence,
            "confidence_after":  checked.confidence,
        }, step=step)

        # Step 2: decide verdict & abstention code
        reasons: list[str] = []
        verdict: Verdict
        abstention_code: Optional[AbstentionCode] = checked.abstention_code

        if checked.abstain or checked.confidence < self.abstain_threshold:
            verdict = Verdict.ABSTAIN
            if checked.abstain:
                reason_msg = checked.hetu if checked.hetu else "Hypothesis Proposer abstained because the query is subjective, lacks verifiable evidence, or requests personal opinion."
                reasons.append(f"Proposer Abstention: {reason_msg}")
                if not abstention_code:
                    h_lower = checked.hetu.lower()
                    if any(w in h_lower for w in ["opinion", "subjective", "taste", "judgment"]):
                        abstention_code = AbstentionCode.SUBJECTIVE_QUERY
                    elif any(w in h_lower for w in ["impossible", "anachronistic", "cross-reference"]):
                        abstention_code = AbstentionCode.IMPOSSIBLE_CROSS_REF
                    elif any(w in h_lower for w in ["unverifiable", "speculative", "insufficient"]):
                        abstention_code = AbstentionCode.UNVERIFIABLE_PREMISE
                    else:
                        abstention_code = AbstentionCode.PROPOSER_ABSTAINED

            if checked.confidence < self.abstain_threshold:
                flag_note = f" (penalized by fallacies: {', '.join(f.value for f in checked.adhyasa_flags)})" if checked.adhyasa_flags else ""
                reasons.append(
                    f"Calibrated confidence ({checked.confidence:.2f}) is below the required threshold ({self.abstain_threshold}){flag_note}."
                )
                if not abstention_code:
                    if checked.adhyasa_flags:
                        abstention_code = AbstentionCode.FALLACY_PENALIZED
                    else:
                        abstention_code = AbstentionCode.LOW_CONFIDENCE

            checked = checked.model_copy(update={"abstain": True, "abstention_code": abstention_code})
        elif checked.adhyasa_flags:
            verdict = Verdict.FLAG_AND_ACCEPT
            reasons.extend(f.value for f in checked.adhyasa_flags)
        else:
            verdict = Verdict.ACCEPT

        result = BuddhiResult(
            claim=checked,
            verdict=verdict,
            reasons=reasons,
            abstention_code=abstention_code
        )

        tracer.log(run_id, TraceEvent.FINAL_OUTPUT, {
            "verdict":         verdict.value,
            "abstention_code": abstention_code.value if abstention_code else None,
            "reasons":         reasons,
            "nigamana":        checked.nigamana,
            "confidence":      checked.confidence,
        }, step=step)

        return result
