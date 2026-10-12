"""
Tests for Phase 1 upgrades:
- Citta contradiction detection & epistemic strength comparison
- Auto-sublation in Citta and Pipeline
- LLM checker multi-provider graceful handling & flag parsing
- Reproducible causal sandbox seeding
"""

import os
import json
import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from core.citta import Citta
from core.pipeline import Pipeline, PipelineConfig
from core.buddhi import Buddhi, Verdict
from causal.sandbox import MedicalSandbox
from pramana.schemas import Claim, Pramana, AdhyasaLabel
from pramana.llm_checker import run_llm_checkers


@pytest.fixture
def temp_citta(tmp_path):
    db_file = tmp_path / "test_citta.db"
    return Citta(db_path=str(db_file))


def test_citta_contradiction_detection(temp_citta):
    # Store initial belief
    claim_1 = {
        "pratijna": "Drug X increases patient recovery rates",
        "hetu": "Observed in preliminary trial",
        "udaharana": "Effective drugs increase recovery",
        "upanaya": "Drug X was given to patients",
        "nigamana": "Drug X increases patient recovery rates",
        "pramana": "sabda",
        "confidence": 0.70,
    }
    b1_id = temp_citta.store_belief(claim_1, origin="read")
    assert len(temp_citta.get_active_beliefs()) == 1

    # Contradictory claim
    contradictory_claim = {
        "pratijna": "Drug X decreases patient recovery rates",
        "hetu": "Large randomized interventional trial",
        "udaharana": "Ineffective drugs decrease recovery",
        "upanaya": "Drug X caused adverse outcomes",
        "nigamana": "Drug X decreases patient recovery rates",
        "pramana": "pratyaksa",
        "confidence": 0.95,
    }

    contradictions = temp_citta.find_contradictions(contradictory_claim)
    assert len(contradictions) == 1
    assert contradictions[0]["id"] == b1_id


def test_epistemic_strength_hierarchy(temp_citta):
    # Higher pramana tier wins
    weak_belief = {"pramana": "sabda", "confidence": 0.80}
    strong_claim = {"pramana": "pratyaksa", "confidence": 0.75}
    assert temp_citta.is_epistemically_stronger(strong_claim, weak_belief) is True

    # Same pramana tier: higher confidence wins
    equal_pramana_lower_conf = {"pramana": "anumana", "confidence": 0.70}
    equal_pramana_higher_conf = {"pramana": "anumana", "confidence": 0.85}
    assert temp_citta.is_epistemically_stronger(equal_pramana_higher_conf, equal_pramana_lower_conf) is True
    assert temp_citta.is_epistemically_stronger(equal_pramana_lower_conf, equal_pramana_higher_conf) is False


def test_auto_sublation_execution(temp_citta):
    # Store old weaker belief
    old_claim = {
        "pratijna": "Red meat consumption increases heart disease risk",
        "hetu": "Correlational study cohort",
        "udaharana": "Correlations observed in text",
        "upanaya": "Observed cohort",
        "nigamana": "Red meat consumption increases heart disease risk",
        "pramana": "sabda",
        "confidence": 0.65,
    }
    old_id = temp_citta.store_belief(old_claim)

    # New stronger claim
    new_claim = {
        "pratijna": "Red meat consumption decreases heart disease risk under low-carb protocol",
        "hetu": "Direct interventional RCT",
        "udaharana": "Intervention protocol",
        "upanaya": "RCT cohort",
        "nigamana": "Red meat consumption decreases heart disease risk under low-carb protocol",
        "pramana": "pratyaksa",
        "confidence": 0.92,
    }
    new_id = temp_citta.store_belief(new_claim)

    contradictions = temp_citta.find_contradictions(new_claim)
    assert len(contradictions) == 1
    assert temp_citta.is_epistemically_stronger(new_claim, contradictions[0])

    temp_citta.sublate_belief(contradictions[0]["id"], new_id)

    # Verify old belief is sublated
    with sqlite3.connect(temp_citta.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, sublated_by FROM beliefs WHERE id = ?", (old_id,))
        row = cursor.fetchone()
        assert row[0] == "sublated"
        assert row[1] == new_id

    # Active beliefs should only contain the new one
    active = temp_citta.get_active_beliefs()
    assert len(active) == 1
    assert active[0]["id"] == new_id


def test_pipeline_auto_sublation_integration(tmp_path):
    db_file = tmp_path / "pipeline_citta.db"
    config = PipelineConfig(provider="openai", db_path=str(db_file), abstain_threshold=0.3)
    pipeline = Pipeline(config)

    # Store an initial held belief
    initial_claim = {
        "pratijna": "Exercise increases chronic inflammation",
        "hetu": "Limited observation",
        "udaharana": "Single observation",
        "upanaya": "Case 1",
        "nigamana": "Exercise increases chronic inflammation",
        "pramana": "sabda",
        "confidence": 0.50,
    }
    old_id = pipeline.citta.store_belief(initial_claim)
    assert len(pipeline.citta.get_active_beliefs()) == 1

    # Mock Manas to propose a stronger contradicting claim
    strong_claim = Claim(
        pratijna="Exercise decreases chronic inflammation",
        hetu="Meta-analysis of clinical interventions",
        udaharana="Physical activity reduces markers systematically",
        upanaya="Applied across clinical trials",
        nigamana="Exercise decreases chronic inflammation",
        pramana=Pramana.ANUMANA,
        confidence=0.88,
        abstain=False,
    )

    with patch.object(pipeline.manas, "propose", return_value=strong_claim):
        result = pipeline.run("Does exercise reduce inflammation?")
        assert result.verdict == Verdict.ACCEPT

    # Check that old belief was sublated automatically
    active_beliefs = pipeline.citta.get_active_beliefs()
    assert len(active_beliefs) == 1
    assert active_beliefs[0]["pratijna"] == "Exercise decreases chronic inflammation"

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, sublated_by FROM beliefs WHERE id = ?", (old_id,))
        row = cursor.fetchone()
        assert row[0] == "sublated"
        assert row[1] is not None


def test_causal_sandbox_deterministic_seed():
    sandbox1 = MedicalSandbox(seed=123)
    sandbox2 = MedicalSandbox(seed=123)

    obs1 = sandbox1.observe(n_samples=500)
    obs2 = sandbox2.observe(n_samples=500)

    assert obs1["treatment_recovery_rate"] == obs2["treatment_recovery_rate"]
    assert obs1["control_recovery_rate"] == obs2["control_recovery_rate"]

    do1 = sandbox1.do(treatment=1, n_samples=500)
    do2 = sandbox2.do(treatment=1, n_samples=500)
    assert do1["recovery_rate"] == do2["recovery_rate"]


def test_llm_checker_graceful_fallback():
    claim = Claim(
        pratijna="Test claim",
        hetu="Test reason",
        udaharana="Rule",
        upanaya="App",
        nigamana="Conclusion",
        pramana=Pramana.ANUMANA,
        confidence=0.8,
    )
    # When keys are missing or invalid provider
    flags = run_llm_checkers(claim, "Query", provider="unknown_provider")
    assert flags == []


def test_citta_vector_embedding_search(temp_citta):
    b1 = {
        "pratijna": "Quantum computing enables polynomial time prime factorization",
        "hetu": "Shor's algorithm proof",
        "udaharana": "Quantum circuits factoring 15",
        "upanaya": "Application of Shor's algorithm",
        "nigamana": "Quantum computing enables polynomial time prime factorization",
        "pramana": "anumana",
        "confidence": 0.88,
    }
    b2 = {
        "pratijna": "Intermittent fasting improves cellular insulin sensitivity",
        "hetu": "Metabolic randomized trials",
        "udaharana": "Caloric restriction protocols",
        "upanaya": "Patients undergoing 16:8 fasting",
        "nigamana": "Intermittent fasting improves cellular insulin sensitivity",
        "pramana": "sabda",
        "confidence": 0.80,
    }
    temp_citta.store_belief(b1)
    temp_citta.store_belief(b2)

    # Search semantically
    results = temp_citta.search_beliefs("fasting protocol and insulin resistance in diabetes", limit=2)
    assert len(results) > 0
    assert "insulin sensitivity" in results[0]["pratijna"]
    assert "similarity_score" in results[0]


def test_structured_abstention_codes():
    buddhi = Buddhi(abstain_threshold=0.40)

    # Subjective opinion query
    subjective_claim = Claim(
        pratijna="Delhi situation is bad",
        hetu="The query solicits a subjective personal opinion and value judgment.",
        udaharana="",
        upanaya="",
        nigamana="",
        pramana=Pramana.SABDA,
        confidence=0.1,
        abstain=True,
    )
    res1 = buddhi.verify(subjective_claim, run_id="run-test-1", query="What do you think of Delhi?")
    assert res1.verdict == Verdict.ABSTAIN
    from pramana.schemas import AbstentionCode
    assert res1.abstention_code == AbstentionCode.SUBJECTIVE_QUERY
    assert res1.claim.abstention_code == AbstentionCode.SUBJECTIVE_QUERY


def test_compare_causal_effect_tool():
    from core.tools import execute_tool
    raw_output = execute_tool("compare_causal_effect", {"n_samples": 500})
    data = json.loads(raw_output)
    assert "average_treatment_effect_ate" in data
    assert "simpson_paradox_reversal" in data
    assert "observational_correlation_diff" in data

