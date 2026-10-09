from dotenv import load_dotenv
load_dotenv(override=True)
from core.pipeline import Pipeline, PipelineConfig

def main():
    pipeline = Pipeline(PipelineConfig(provider="anthropic"))
    query = "Who was the 47th President of the United States in the year 1800?"
    print(f"Query: {query}\n")
    print("Running pipeline...")
    result = pipeline.run(query)
    
    print("\n--- Output ---")
    print(f"Verdict: {result.verdict}")
    print(f"Pramana: {result.claim.pramana.value}")
    print(f"Conclusion: {result.claim.nigamana}")
    if result.claim.adhyasa_flags:
        print("Flags:", [f.value for f in result.claim.adhyasa_flags])

if __name__ == "__main__":
    main()
