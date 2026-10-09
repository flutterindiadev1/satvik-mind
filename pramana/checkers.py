"""
Adhyāsa checkers – rule-based detectors for superimposition errors.

Phase 0 / Phase 2 scope: checkers for labels 1 (correlation→causation),
4 (scope inflation), 5 (self-output as evidence), and 7 (absence confusion).
Labels 2, 3, and 6 require LLM-assisted checks added in Phase 2.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pramana.schemas import AdhyasaLabel, Claim

# Regex trigger patterns per checker
_CAUSAL_LANGUAGE = re.compile(
    r"\b(causes?|leads? to|results? in|produces?|makes?|drives?|triggers?)\b",
    re.IGNORECASE,
)
_CORRELATIONAL_SETUP = re.compile(
    r"\b(correlated?|associated?|linked?|related?|co-occurs?|tends? to)\b",
    re.IGNORECASE,
)
_UNIVERSAL_QUANTIFIERS = re.compile(
    r"\b(always|never|all|every|no\s+\w+|none|invariably|universally|in all cases)\b",
    re.IGNORECASE,
)
_SELF_CITATION_MARKERS = re.compile(
    r"\b(as I (said|mentioned|noted|stated|explained|previously stated)|per my (previous|earlier|prior)|"
    r"as previously (stated|noted|established)|based on (my|the above) (analysis|reasoning|output)|"
    r"previously stated|as mentioned above|as I noted|as stated above)\b",
    re.IGNORECASE,
)


class AdhyasaChecker(ABC):
    """Abstract base class for all adhyāsa checkers."""

    label: "AdhyasaLabel"

    @abstractmethod
    def check(self, claim: "Claim") -> bool:
        """Return True if the adhyāsa pattern is detected."""
        ...

    def flag_text(self) -> str:
        return self.label.value


# ---------------------------------------------------------------------------
# Checker 1 – Correlation → Causation (label: correlation_causation)
# ---------------------------------------------------------------------------

class CorrelationCausationChecker(AdhyasaChecker):
    """
    Flags when a claim uses causal language (causes, leads to…) while its
    hetu (reason) only establishes correlation or association.

    Phase 2 will add graph-level checks; this is the surface heuristic.
    """

    def __init__(self) -> None:
        from pramana.schemas import AdhyasaLabel
        self.label = AdhyasaLabel.CORRELATION_CAUSATION

    def check(self, claim: "Claim") -> bool:
        conclusion_causal = bool(_CAUSAL_LANGUAGE.search(claim.nigamana))
        reason_correlational = bool(_CORRELATIONAL_SETUP.search(claim.hetu))
        return conclusion_causal and reason_correlational


# ---------------------------------------------------------------------------
# Checker 4 – Scope inflation (label: scope_inflation)
# ---------------------------------------------------------------------------

class ScopeInflationChecker(AdhyasaChecker):
    """
    Flags when the udāharana (general rule) uses universal quantifiers
    but the hetu describes a limited observation — invalid vyāpti.
    """

    def __init__(self) -> None:
        from pramana.schemas import AdhyasaLabel
        self.label = AdhyasaLabel.SCOPE_INFLATION

    def check(self, claim: "Claim") -> bool:
        universal_in_rule = bool(_UNIVERSAL_QUANTIFIERS.search(claim.udaharana))
        # Heuristic: if the hetu mentions a single case or "in this study / in this example"
        limited_hetu = bool(re.search(
            r"\b(in this (case|study|example|instance|experiment)|one (case|instance|example)|"
            r"a single|this particular|the one|in this one|one study|one experiment)\b",
            claim.hetu, re.IGNORECASE,
        ))
        return universal_in_rule and limited_hetu


# ---------------------------------------------------------------------------
# Checker 5 – Self-output as evidence (label: self_output_as_evidence)
# ---------------------------------------------------------------------------

class SelfOutputAsEvidenceChecker(AdhyasaChecker):
    """
    Flags when the system cites its own prior output as an observation (pratyakṣa)
    rather than as inference (anumāna). Always invalid as a pramāṇa upgrade.
    """

    def __init__(self) -> None:
        from pramana.schemas import AdhyasaLabel
        self.label = AdhyasaLabel.SELF_OUTPUT_AS_EVIDENCE

    def check(self, claim: "Claim") -> bool:
        from pramana.schemas import Pramana
        self_reference = bool(_SELF_CITATION_MARKERS.search(claim.hetu))
        # Especially egregious when pramāṇa is pratyakṣa (direct observation)
        wrong_pramana = claim.pramana in (Pramana.PRATYAKSA, Pramana.ANUMANA)
        return self_reference and wrong_pramana


# ---------------------------------------------------------------------------
# Checker 7 – Absence confusion (label: absence_confusion)
# ---------------------------------------------------------------------------

class AbsenceConfusionChecker(AdhyasaChecker):
    """
    Flags when a claim states non-existence without a linked AbsenceRecord.
    An anupalabdhi claim is only valid when backed by a logged search.
    """

    def __init__(self) -> None:
        from pramana.schemas import AdhyasaLabel
        self.label = AdhyasaLabel.ABSENCE_CONFUSION

    def check(self, claim: "Claim") -> bool:
        from pramana.schemas import Pramana

        absence_language = bool(re.search(
            r"\b(does not exist|no evidence|not found|absent|cannot be found|"
            r"there (is|are) no|no \w+ (exists?|found|available))\b",
            claim.nigamana, re.IGNORECASE,
        ))
        # Flagged if absence language is used but pramāṇa is NOT anupalabdhi
        # OR if anupalabdhi is claimed but evidence_refs is empty
        if absence_language and claim.pramana != Pramana.ANUPALABDHI:
            return True
        if claim.pramana == Pramana.ANUPALABDHI and not claim.evidence_refs:
            return True
        return False


# ---------------------------------------------------------------------------
# Composite runner
# ---------------------------------------------------------------------------

RULE_BASED_CHECKERS: list[AdhyasaChecker] = [
    CorrelationCausationChecker(),
    ScopeInflationChecker(),
    SelfOutputAsEvidenceChecker(),
    AbsenceConfusionChecker(),
]


def run_checkers(claim: "Claim") -> "Claim":
    """
    Run all rule-based adhyāsa checkers against a Claim.
    Returns a new Claim with adhyasa_flags populated and confidence
    capped to the pramāṇa ceiling.
    """
    flags = list(claim.adhyasa_flags)  # preserve any pre-existing flags
    for checker in RULE_BASED_CHECKERS:
        if checker.check(claim) and checker.label not in flags:
            flags.append(checker.label)

    updated = claim.model_copy(update={"adhyasa_flags": flags})
    return updated.clamp_confidence()
