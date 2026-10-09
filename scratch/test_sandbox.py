from dotenv import load_dotenv
load_dotenv(override=True)

from core.pipeline import Pipeline
import sys

def main():
    pipeline = Pipeline()
    query = (
        "There is a medical trial for Drug X. You have access to the sandbox tools. "
        "First, use observe_sandbox to see the natural correlation between Treatment (Drug X) and Recovery. "
        "Then, use intervene_sandbox with treatment=1 and treatment=0 to see the true causal effect. "
        "Based on your findings, does Drug X causally improve recovery rates, or is the observational correlation misleading?"
    )
    print(f"Query: {query}\n")
    print("Running pipeline...")
    result = pipeline.run(query)
    
    print("\n--- Output ---")
    print(f"Verdict: {result.verdict}")
    print(f"Nigamana (Conclusion): {result.claim.nigamana}")
    if result.claim.pramana:
        print(f"Pramana: {result.claim.pramana.value}")

if __name__ == "__main__":
    main()
