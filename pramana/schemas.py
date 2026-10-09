"""
Sāttvic Mind – core schemas.

All claims in the system must be instances of these types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pramāṇas – valid means of knowledge
# ---------------------------------------------------------------------------

class Pramana(str, Enum):
    """The six Advaita pramāṇas, mapped to evidence source types."""

    PRATYAKSA   = "pratyaksa"    # direct observation: sensor / tool call / code execution
    ANUMANA     = "anumana"      # inference: causal / logical derivation
    UPAMANA     = "upamana"      # analogy / similarity retrieval
    SABDA       = "sabda"        # testimony: cited external source
    ARTHAPATTI  = "arthapatti"   # postulation / abduction
    ANUPALABDHI = "anupalabdhi"  # non-apprehension: logged absence

    def max_confidence(self) -> float:
        """
        Empirical confidence ceiling per pramāṇa.
        Ordering hypothesis (to be calibrated in Phase 2):
          pratyakṣa > anumāna > śabda > anupalabdhi > upamāna > arthāpatti
        These are starting values, not fixed rules.
        """
        _ceilings: dict[str, float] = {
            "pratyaksa":   0.95,
            "anumana":     0.90,
            "sabda":       0.80,
            "anupalabdhi": 0.75,
            "upamana":     0.70,
            "arthapatti":  0.65,
        }
        return _ceilings[self.value]


# ---------------------------------------------------------------------------
# Adhyāsa (superimposition) labels
# ---------------------------------------------------------------------------

class AdhyasaLabel(str, Enum):
    """Error taxonomy from §3.2 of the project spec."""

    CORRELATION_CAUSATION  = "correlation_causation"   # 1: association taken for intervention
    FREQUENCY_TRUTH        = "frequency_truth"          # 2: text popularity taken for fact
    SOURCE_CONFUSION       = "source_confusion"         # 3: testimony presented as perception
    SCOPE_INFLATION        = "scope_inflation"           # 4: particular stretched to universal
    SELF_OUTPUT_AS_EVIDENCE = "self_output_as_evidence" # 5: own prior output cited as observation
    ASKER_PROJECTION       = "asker_projection"         # 6: answer bent to questioner's wish
    ABSENCE_CONFUSION      = "absence_confusion"        # 7: "not retrieved" treated as "does not exist"


# ---------------------------------------------------------------------------
# Claim – five-member Nyāya format
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    """
    A single reasoned claim in the five-step Nyāya format, annotated with
    its pramāṇa (evidence source) and any detected adhyāsa flags.
    """

    # Five-step argument
    pratijna:   str = Field(description="The claim / thesis statement")
    hetu:       str = Field(description="The reason / ground for the claim")
    udaharana:  str = Field(description="General rule (vyāpti) + concrete instance")
    upanaya:    str = Field(description="Application of the rule to this specific case")
    nigamana:   str = Field(description="Conclusion restating the claim on the given ground")

    # Evidence metadata
    pramana:       Pramana
    evidence_refs: list[str] = Field(default_factory=list,
                                     description="IDs / URLs of supporting evidence")

    # Confidence and safety
    confidence:    float = Field(ge=0.0, le=1.0,
                                 description="Posterior confidence [0, 1]")
    abstain:       bool  = Field(default=False,
                                 description="True if the system should abstain from answering")

    # Error flags (filled by the adhyāsa checker, not the proposer)
    adhyasa_flags: list[AdhyasaLabel] = Field(default_factory=list)

    # Provenance
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def clamp_confidence(self) -> "Claim":
        """
        Apply pramāṇa ceiling.  Returns a new Claim with confidence capped.
        Call this after construction if you want enforced ceilings.
        """
        ceiling = self.pramana.max_confidence()
        if self.confidence > ceiling:
            return self.model_copy(update={"confidence": ceiling})
        return self

    def is_flagged(self) -> bool:
        return bool(self.adhyasa_flags)


# ---------------------------------------------------------------------------
# Belief – a claim in the provenance store with sublation tracking
# ---------------------------------------------------------------------------

class Belief(BaseModel):
    """
    A Claim that lives in the Citta (memory / provenance store).
    Beliefs can be held, under review, or sublated by stronger evidence.
    """

    id:           str
    claim:        Claim
    origin:       Literal["observed", "intervened", "read", "inferred", "postulated"]
    status:       Literal["held", "under_review", "sublated"] = "held"
    sublated_by:  Optional[str] = Field(default=None,
                                        description="ID of the belief/evidence that sublated this one")
    created_at:       datetime = Field(default_factory=datetime.utcnow)
    last_audited_at:  Optional[datetime] = None

    def sublate(self, by_id: str) -> "Belief":
        """Mark this belief as sublated by a stronger pramāṇa."""
        return self.model_copy(update={
            "status": "sublated",
            "sublated_by": by_id,
            "last_audited_at": datetime.utcnow(),
        })

    @property
    def is_active(self) -> bool:
        return self.status == "held"


# ---------------------------------------------------------------------------
# AbsenceRecord – required for any anupalabdhi (non-apprehension) claim
# ---------------------------------------------------------------------------

class AbsenceRecord(BaseModel):
    """
    A logged search that returned no results.
    An absence claim (pramāṇa=ANUPALABDHI) is only epistemically valid
    when linked to a record like this.
    """

    query:        str  = Field(description="What was searched for")
    search_scope: str  = Field(description="Corpus, tool, or time range searched")
    tools_used:   list[str]
    result_count: int  = Field(ge=0, description="Must be 0 for a valid absence claim")
    searched_at:  datetime = Field(default_factory=datetime.utcnow)

    def is_valid_absence(self) -> bool:
        return self.result_count == 0
