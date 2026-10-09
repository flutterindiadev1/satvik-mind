from core.buddhi import Buddhi
from pramana.schemas import Claim, Pramana
import json

def test_checkers():
    buddhi = Buddhi()
    
    # Test 1: Self-output as evidence
    claim1 = Claim(
        pratijna="The sky is green.",
        hetu="As I previously stated, the sky has turned green.",
        udaharana="Rule: If I say it, it is so.",
        upanaya="I said it, so it is.",
        nigamana="The sky is green.",
        pramana=Pramana.PRATYAKSA,
        confidence=0.9
    )
    result1 = buddhi.verify(claim1, run_id="test-1")
    print("Test 1 (Self-output):", result1.verdict.value, [f.value for f in result1.claim.adhyasa_flags])
    
    # Test 2: Correlation -> Causation
    claim2 = Claim(
        pratijna="Eating ice cream causes shark attacks.",
        hetu="Ice cream sales are highly correlated with shark attacks.",
        udaharana="Rule: Things that co-occur are causally linked.",
        upanaya="Ice cream and sharks co-occur.",
        nigamana="Therefore, ice cream causes shark attacks.",
        pramana=Pramana.ANUMANA,
        confidence=0.9
    )
    result2 = buddhi.verify(claim2, run_id="test-2")
    print("Test 2 (Corr->Caus):", result2.verdict.value, [f.value for f in result2.claim.adhyasa_flags])
    
    # Test 3: Absence Confusion
    claim3 = Claim(
        pratijna="Unicorns do not exist.",
        hetu="I cannot think of any unicorns.",
        udaharana="Rule: If it cannot be thought of, it doesn't exist.",
        upanaya="Unicorns cannot be thought of.",
        nigamana="Therefore, no unicorn exists.",
        pramana=Pramana.SABDA, # Wrong pramana for absence
        confidence=0.9
    )
    result3 = buddhi.verify(claim3, run_id="test-3")
    print("Test 3 (Absence):", result3.verdict.value, [f.value for f in result3.claim.adhyasa_flags])

if __name__ == "__main__":
    test_checkers()
