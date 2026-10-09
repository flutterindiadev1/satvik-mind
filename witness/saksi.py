import json
import sqlite3
from pathlib import Path
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def analyze_traces(trace_dir: str = "traces") -> dict:
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return {}
    
    stats = {
        "total_runs": 0,
        "verdicts": Counter(),
        "adhyasa_flags": Counter(),
        "pramanas": Counter(),
    }
    
    run_ids = set()
    
    for log_file in trace_path.glob("*.jsonl"):
        with open(log_file, "r") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                run_id = data.get("run_id")
                if run_id:
                    run_ids.add(run_id)
                
                event = data.get("event")
                if event == "claim_proposed":
                    pramana = data.get("pramana")
                    if pramana:
                        stats["pramanas"][pramana] += 1
                        
                elif event == "final_output":
                    verdict = data.get("verdict")
                    if verdict:
                        stats["verdicts"][verdict] += 1
                    reasons = data.get("reasons", [])
                    # Flags are usually stored in reasons for flag_and_accept
                    for reason in reasons:
                        stats["adhyasa_flags"][reason] += 1
                        
    stats["total_runs"] = len(run_ids)
    return stats

def analyze_citta(db_path: str = "citta.db") -> dict:
    if not Path(db_path).exists():
        return {}
    
    stats = {
        "active_beliefs": 0,
        "sublated_beliefs": 0,
        "total_absences": 0
    }
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM beliefs WHERE status = 'held'")
            stats["active_beliefs"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM beliefs WHERE status = 'sublated'")
            stats["sublated_beliefs"] = cursor.fetchone()[0]
            
            # Check if absences table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='absences'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM absences")
                stats["total_absences"] = cursor.fetchone()[0]
    except Exception as e:
        pass
        
    return stats

def main():
    console = Console()
    console.print(Panel.fit("[bold green]Sākṣī (Witness)[/bold green]\nRead-only Monitoring Service", border_style="green"))
    
    trace_stats = analyze_traces()
    citta_stats = analyze_citta()
    
    # 1. Traces Overview
    table_traces = Table(title="Trace Telemetry (Recent Runs)")
    table_traces.add_column("Metric", style="cyan")
    table_traces.add_column("Value", style="magenta")
    table_traces.add_row("Total Pipeline Runs", str(trace_stats.get("total_runs", 0)))
    
    verdicts = trace_stats.get("verdicts", {})
    table_traces.add_row("Accepted Claims", str(verdicts.get("accept", 0)))
    table_traces.add_row("Abstentions", str(verdicts.get("abstain", 0)))
    table_traces.add_row("Flagged Claims", str(verdicts.get("flag_and_accept", 0)))
    
    console.print(table_traces)
    console.print()
    
    # 2. Adhyasa Flags Caught
    table_flags = Table(title="Adhyāsa (Fallacies) Intercepted")
    table_flags.add_column("Fallacy Type", style="red")
    table_flags.add_column("Count", justify="right")
    
    flags = trace_stats.get("adhyasa_flags", {})
    if not flags:
        table_flags.add_row("No fallacies detected yet.", "-")
    else:
        for flag, count in flags.most_common():
            # Filter out non-flag reasons
            if "confidence" not in flag and "abstain" not in flag:
                table_flags.add_row(flag, str(count))
            
    console.print(table_flags)
    console.print()
    
    # 3. Citta Store
    table_citta = Table(title="Citta (Memory Store) State")
    table_citta.add_column("Entity", style="yellow")
    table_citta.add_column("Count", justify="right")
    table_citta.add_row("Active Beliefs", str(citta_stats.get("active_beliefs", 0)))
    table_citta.add_row("Sublated (Overwritten) Beliefs", str(citta_stats.get("sublated_beliefs", 0)))
    table_citta.add_row("Absence Records (Anupalabdhi)", str(citta_stats.get("total_absences", 0)))
    
    console.print(table_citta)
    console.print()

if __name__ == "__main__":
    main()
