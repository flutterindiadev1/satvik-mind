import random
from typing import Dict, Any, List

class MedicalSandbox:
    """
    A causal simulation of a medical trial with a hidden confounder (Simpson's Paradox).
    
    Variables:
    - Age (0 = Young, 1 = Old) -> Hidden confounder
    - Treatment (0 = Control, 1 = Drug X)
    - Recovery (0 = No, 1 = Yes)
    
    Causal Graph:
    Age -> Treatment
    Age -> Recovery
    Treatment -> Recovery
    """
    
    def __init__(self):
        self.description = (
            "A medical dataset tracking the effect of Drug X on patient recovery. "
            "Variables: Treatment (1=Drug X, 0=Placebo), Recovery (1=Recovered, 0=Not Recovered)."
        )

    def _generate_sample(self, do_treatment: int = None) -> Dict[str, int]:
        # Age ~ Bernoulli(0.5)
        age = 1 if random.random() < 0.5 else 0
        
        # Treatment ~ Bernoulli(0.1 + 0.7 * Age) if not intervened on
        if do_treatment is not None:
            treatment = do_treatment
        else:
            prob_treatment = 0.8 if age == 1 else 0.1
            treatment = 1 if random.random() < prob_treatment else 0
            
        # Recovery ~ Bernoulli(0.8 - 0.5 * Age + 0.1 * Treatment)
        prob_recovery = 0.8 - (0.5 * age) + (0.1 * treatment)
        recovery = 1 if random.random() < prob_recovery else 0
        
        return {
            "Age": age,  # Usually hidden, but kept here for internal logic
            "Treatment": treatment,
            "Recovery": recovery
        }

    def observe(self, n_samples: int = 1000) -> Dict[str, Any]:
        """Observational data: naturally generated without intervention."""
        samples = [self._generate_sample() for _ in range(n_samples)]
        
        treated = [s for s in samples if s["Treatment"] == 1]
        untreated = [s for s in samples if s["Treatment"] == 0]
        
        rec_treated = sum(s["Recovery"] for s in treated) / len(treated) if treated else 0
        rec_untreated = sum(s["Recovery"] for s in untreated) / len(untreated) if untreated else 0
        
        return {
            "type": "observational",
            "n_samples": n_samples,
            "treatment_recovery_rate": round(rec_treated, 3),
            "control_recovery_rate": round(rec_untreated, 3),
            "correlation_implies": "Treatment DECREASES recovery." if rec_treated < rec_untreated else "Treatment INCREASES recovery."
        }

    def do(self, treatment: int, n_samples: int = 1000) -> Dict[str, Any]:
        """Interventional data: forcing the treatment variable (do-calculus)."""
        samples = [self._generate_sample(do_treatment=treatment) for _ in range(n_samples)]
        
        recovery_rate = sum(s["Recovery"] for s in samples) / n_samples
        
        return {
            "type": "interventional",
            "do_treatment": treatment,
            "n_samples": n_samples,
            "recovery_rate": round(recovery_rate, 3)
        }
