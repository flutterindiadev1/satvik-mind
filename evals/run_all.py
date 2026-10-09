#!/usr/bin/env python3
import os
import subprocess
import sys

# We'll run the evaluations for the baseline models.
MODELS = [
    "anthropic/claude-haiku-4-5-20251001",
    # Add other models here for the baseline audit
    # "openai/gpt-4o-mini",
    # "ollama/llama3",
]

EVAL_TASKS = [
    "evals/tamas.py",
    "evals/rajas.py",
    "evals/sattva.py",
]

def main():
    print("Running Phase 1 Baseline Audit...")
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating Model: {model}")
        print(f"{'='*60}")
        
        for task in EVAL_TASKS:
            print(f"\n--- Running {task} ---")
            # Try 'inspect' directly or fallback to 'python3 -m inspect_ai'
            cmd = ["python3", "-m", "inspect_ai", "eval", task, "--model", model]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running {task} on {model}: {e}")
                
if __name__ == "__main__":
    main()
