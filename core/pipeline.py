"""
Pipeline – top-level reasoning loop.

Wires together: Manas (proposer) → Buddhi (verifier) → output.
Phase 0: single-step, no memory.
Phase 4: will plug in Citta (memory) between verifier and output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.buddhi import Buddhi, BuddhiResult
from core.manas import Manas
from core.citta import Citta
from core.tracer import tracer, TraceEvent

@dataclass
class PipelineConfig:
    provider:           Optional[str] = None
    model:              Optional[str] = None
    temperature:        float = 0.2
    abstain_threshold:  float = 0.35
    db_path:            str = "citta.db"


class Pipeline:
    """
    The main antaḥkaraṇa loop.

    Usage:
        pipeline = Pipeline()
        result = pipeline.run("Does eating red meat cause heart disease?")
        print(result.output_text)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        cfg = config or PipelineConfig()
        self.manas  = Manas(provider=cfg.provider, model=cfg.model, temperature=cfg.temperature)
        self.buddhi = Buddhi(abstain_threshold=cfg.abstain_threshold)
        self.citta  = Citta(db_path=cfg.db_path)

    def run(self, query: str, extra_context: Optional[str] = None) -> BuddhiResult:
        run_id = tracer.new_run(query=query)
        
        # In the future, we will query Citta first for sublated/held beliefs on this query.
        
        claim  = self.manas.propose(query, run_id=run_id, step=0, extra_context=extra_context)
        result = self.buddhi.verify(claim, run_id=run_id, query=query, step=1)
        
        if result.verdict == "accept" and claim:
            # Convert the Pydantic Claim to json-serializable dict for storage
            claim_dict = claim.model_dump(mode="json")
            belief_id = self.citta.store_belief(claim_dict, origin="inferred")
            tracer.log(run_id, TraceEvent.BELIEF_CREATED, step=2, payload={"belief_id": belief_id, "status": "held"})
            
        return result
