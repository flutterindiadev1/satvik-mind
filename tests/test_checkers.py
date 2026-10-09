"""
Tests for pramana/checkers.py – Phase 0 rule-based adhyāsa detection.
Run with: pytest tests/test_checkers.py -v
"""

import pytest
from pramana.schemas import AdhyasaLabel, Claim, Pramana
from pramana.checkers import (
    AbsenceConfusionChecker,
    CorrelationCausationChecker,
    ScopeInflationChecker,
    SelfOutputAsEvidenceChecker,
    run_checkers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_claim(**kwargs) -> Claim:
    defaults = dict(
        pratijna="Test claim",
        hetu="Test reason",
        udaharana="General rule — concrete example",
        upanaya="Application to this case",
        nigamana="Conclusion",
        pramana=Pramana.ANUMANA,
        confidence=0.8,
    )
    defaults.update(kwargs)
    return Claim(**defaults)


# ---------------------------------------------------------------------------
# Checker 1 – Correlation → Causation
# ---------------------------------------------------------------------------

def test_correlation_causation_detected():
    claim = base_claim(
        hetu="X is correlated with Y in the dataset",
        nigamana="X causes Y in the general case",
    )
    assert CorrelationCausationChecker().check(claim)


def test_correlation_causation_not_detected_for_clean_causal():
    claim = base_claim(
        hetu="Intervention on X (do(X=1)) increases Y by 20% in RCT",
        nigamana="X causes Y",
    )
    assert not CorrelationCausationChecker().check(claim)


# ---------------------------------------------------------------------------
# Checker 4 – Scope inflation
# ---------------------------------------------------------------------------

def test_scope_inflation_detected():
    claim = base_claim(
        hetu="In this one study, participants improved",
        udaharana="All interventions of this type always produce improvement — observed here",
    )
    assert ScopeInflationChecker().check(claim)


def test_scope_inflation_not_detected_for_limited_claim():
    claim = base_claim(
        hetu="Meta-analysis of 40 RCTs shows consistent effect",
        udaharana="Evidence across many studies suggests the effect is robust — confirmed here",
    )
    assert not ScopeInflationChecker().check(claim)


# ---------------------------------------------------------------------------
# Checker 5 – Self-output as evidence
# ---------------------------------------------------------------------------

def test_self_output_as_evidence_detected():
    claim = base_claim(
        hetu="As I previously stated, the mechanism is X",
        pramana=Pramana.PRATYAKSA,
    )
    assert SelfOutputAsEvidenceChecker().check(claim)


def test_self_output_not_flagged_for_sabda():
    # Testimony about one's own output is still bad, but the checker targets the
    # pramāṇa upgrade (treating self-output as pratyakṣa/anumāna).
    claim = base_claim(
        hetu="As I mentioned, external source X states Y",
        pramana=Pramana.SABDA,
    )
    # sabda is outside the targeted pramāṇas, so should not fire
    assert not SelfOutputAsEvidenceChecker().check(claim)


# ---------------------------------------------------------------------------
# Checker 7 – Absence confusion
# ---------------------------------------------------------------------------

def test_absence_confusion_wrong_pramana():
    """Absence language used but pramāṇa is not anupalabdhi."""
    claim = base_claim(
        nigamana="There is no evidence that X exists",
        pramana=Pramana.SABDA,
    )
    assert AbsenceConfusionChecker().check(claim)


def test_absence_confusion_anupalabdhi_without_refs():
    """Anupalabdhi claimed but no AbsenceRecord reference."""
    claim = base_claim(
        nigamana="X does not exist based on search",
        pramana=Pramana.ANUPALABDHI,
        evidence_refs=[],   # empty — no linked AbsenceRecord
    )
    assert AbsenceConfusionChecker().check(claim)


def test_absence_confusion_clean_anupalabdhi():
    """Proper anupalabdhi with a linked AbsenceRecord ref — should NOT flag."""
    claim = base_claim(
        nigamana="There is no evidence of X",
        pramana=Pramana.ANUPALABDHI,
        evidence_refs=["absence:ar-001"],
    )
    assert not AbsenceConfusionChecker().check(claim)


# ---------------------------------------------------------------------------
# Composite runner
# ---------------------------------------------------------------------------

def test_run_checkers_returns_clamped_claim():
    """run_checkers should apply confidence ceiling even for clean claims."""
    claim = base_claim(pramana=Pramana.ARTHAPATTI, confidence=0.99)
    result = run_checkers(claim)
    assert result.confidence <= Pramana.ARTHAPATTI.max_confidence()


def test_run_checkers_accumulates_flags():
    claim = base_claim(
        hetu="X is associated with Y in this one study; as I mentioned above",
        udaharana="All such associations always result in causation — observed here",
        nigamana="X causes Y; there is no evidence otherwise",
        pramana=Pramana.PRATYAKSA,
        evidence_refs=[],
    )
    result = run_checkers(claim)
    flags = {f.value for f in result.adhyasa_flags}
    # Should catch correlation→causation, scope inflation,
    # self-output-as-evidence, and absence confusion
    assert AdhyasaLabel.CORRELATION_CAUSATION.value in flags
    assert AdhyasaLabel.SCOPE_INFLATION.value in flags
