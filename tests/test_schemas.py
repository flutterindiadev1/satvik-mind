"""
Tests for pramana/schemas.py – Phase 0 sanity checks.
Run with: pytest tests/test_schemas.py -v
"""

import pytest
from datetime import datetime

from pramana.schemas import (
    AbsenceRecord,
    AdhyasaLabel,
    Belief,
    Claim,
    Pramana,
)


# ---------------------------------------------------------------------------
# Claim construction
# ---------------------------------------------------------------------------

def make_claim(**overrides) -> Claim:
    defaults = dict(
        pratijna="Aspirin reduces heart attack risk",
        hetu="It inhibits platelet aggregation",
        udaharana="All antiplatelet agents reduce clot formation; aspirin is one such agent",
        upanaya="Therefore aspirin reduces clot formation in cardiac vessels",
        nigamana="Aspirin reduces heart attack risk by inhibiting platelet aggregation",
        pramana=Pramana.SABDA,
        evidence_refs=["pubmed:12345678"],
        confidence=0.75,
    )
    defaults.update(overrides)
    return Claim(**defaults)


def test_claim_construction():
    claim = make_claim()
    assert claim.pramana == Pramana.SABDA
    assert claim.confidence == 0.75
    assert not claim.abstain
    assert not claim.adhyasa_flags


def test_confidence_clamped_to_pramana_ceiling():
    """śabda ceiling is 0.80; passing 0.95 should be clamped."""
    claim = make_claim(confidence=0.95).clamp_confidence()
    assert claim.confidence == Pramana.SABDA.max_confidence()


def test_pratyaksa_highest_ceiling():
    assert Pramana.PRATYAKSA.max_confidence() > Pramana.SABDA.max_confidence()
    assert Pramana.PRATYAKSA.max_confidence() > Pramana.ARTHAPATTI.max_confidence()


def test_abstain_flag():
    claim = make_claim(abstain=True, confidence=0.1)
    assert claim.abstain is True


def test_is_flagged_false_by_default():
    claim = make_claim()
    assert not claim.is_flagged()


def test_is_flagged_true_when_flags_set():
    claim = make_claim(adhyasa_flags=[AdhyasaLabel.SCOPE_INFLATION])
    assert claim.is_flagged()


# ---------------------------------------------------------------------------
# Belief / sublation
# ---------------------------------------------------------------------------

def test_belief_active_by_default():
    belief = Belief(
        id="b-001",
        claim=make_claim(),
        origin="read",
        created_at=datetime.utcnow(),
    )
    assert belief.is_active


def test_belief_sublation():
    belief = Belief(
        id="b-001",
        claim=make_claim(),
        origin="read",
        created_at=datetime.utcnow(),
    )
    sublated = belief.sublate(by_id="b-002")
    assert sublated.status == "sublated"
    assert sublated.sublated_by == "b-002"
    assert not sublated.is_active


# ---------------------------------------------------------------------------
# AbsenceRecord
# ---------------------------------------------------------------------------

def test_absence_record_valid():
    rec = AbsenceRecord(
        query="Is there evidence of X?",
        search_scope="PubMed 2000–2024",
        tools_used=["pubmed_search"],
        result_count=0,
    )
    assert rec.is_valid_absence()


def test_absence_record_invalid_if_results_found():
    rec = AbsenceRecord(
        query="Is there evidence of X?",
        search_scope="PubMed",
        tools_used=["pubmed_search"],
        result_count=3,
    )
    assert not rec.is_valid_absence()
