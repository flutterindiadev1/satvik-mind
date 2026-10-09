import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

class Citta:
    """
    Citta (Memory) - Provenance memory and sublation log.
    Stores beliefs, their origins, and tracks if they are sublated by newer, stronger evidence.
    """
    def __init__(self, db_path: str = "citta.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS beliefs (
                    id TEXT PRIMARY KEY,
                    pratijna TEXT NOT NULL,
                    hetu TEXT,
                    udaharana TEXT,
                    upanaya TEXT,
                    nigamana TEXT,
                    pramana TEXT,
                    confidence REAL,
                    origin TEXT,
                    status TEXT DEFAULT 'held',
                    sublated_by TEXT,
                    created_at TIMESTAMP,
                    last_audited_at TIMESTAMP,
                    raw_claim JSON
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS absences (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    search_scope TEXT NOT NULL,
                    tools_used TEXT NOT NULL,
                    result_count INTEGER NOT NULL,
                    searched_at TIMESTAMP
                )
            ''')
            conn.commit()

    def store_belief(self, claim_dict: Dict[str, Any], origin: str = "inferred") -> str:
        """Stores a new belief derived from a claim."""
        belief_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO beliefs (
                    id, pratijna, hetu, udaharana, upanaya, nigamana, 
                    pramana, confidence, origin, status, created_at, raw_claim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'held', ?, ?)
            ''', (
                belief_id,
                claim_dict.get('pratijna', ''),
                claim_dict.get('hetu', ''),
                claim_dict.get('udaharana', ''),
                claim_dict.get('upanaya', ''),
                claim_dict.get('nigamana', ''),
                claim_dict.get('pramana', ''),
                claim_dict.get('confidence', 0.0),
                origin,
                now,
                json.dumps(claim_dict)
            ))
            conn.commit()
            
        return belief_id

    def sublate_belief(self, old_belief_id: str, new_belief_id: str):
        """
        Marks an old belief as sublated (invalidated) by a newer, stronger belief.
        This embodies 'abādhitatva' (Truth as non-sublation).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE beliefs 
                SET status = 'sublated', sublated_by = ? 
                WHERE id = ?
            ''', (new_belief_id, old_belief_id))
            conn.commit()

    def store_absence(self, query: str, search_scope: str, tools_used: List[str]) -> str:
        """Stores a logged absence search record."""
        absence_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO absences 
                (id, query, search_scope, tools_used, result_count, searched_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                absence_id,
                query,
                search_scope,
                json.dumps(tools_used),
                0,
                now
            ))
            conn.commit()
            
        return absence_id
    def get_active_beliefs(self) -> List[Dict[str, Any]]:
        """Retrieves all currently held (non-sublated) beliefs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM beliefs WHERE status = 'held'")
            return [dict(row) for row in cursor.fetchall()]

