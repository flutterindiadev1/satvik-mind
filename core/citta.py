import sqlite3
import json
import uuid
import re
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two numeric vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1)) or 1.0
    norm2 = math.sqrt(sum(b * b for b in v2)) or 1.0
    return dot / (norm1 * norm2)


class BeliefEmbedder:
    """
    Semantic embedding engine for Citta memory.
    Uses sentence-transformers ('all-MiniLM-L6-v2') when available,
    with an efficient normalized n-gram hash vector fallback if offline.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _get_model(self):
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None
            self._initialized = True
        return self._model

    def encode(self, text: str) -> List[float]:
        model = self._get_model()
        if model is not None:
            try:
                emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                return [float(x) for x in emb.tolist()]
            except Exception:
                pass
        return self._fallback_encode(text)

    def _fallback_encode(self, text: str, dim: int = 128) -> List[float]:
        import hashlib
        vec = [0.0] * dim
        words = text.lower().split()
        for w in words:
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [float(x / norm) for x in vec]


class Citta:
    """
    Citta (Memory) - Provenance memory, vector retrieval, and sublation log.
    Stores beliefs, their origins, vector embeddings, and tracks if they are 
    sublated by newer, stronger evidence.
    """
    def __init__(self, db_path: str = "citta.db"):
        self.db_path = db_path
        self.embedder = BeliefEmbedder()
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
                    raw_claim JSON,
                    embedding JSON
                )
            ''')
            # Add embedding column if missing in older databases
            try:
                conn.execute("ALTER TABLE beliefs ADD COLUMN embedding JSON")
            except Exception:
                pass

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
        """Stores a new belief derived from a claim with its semantic vector embedding."""
        belief_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        # Calculate semantic vector embedding
        text_to_embed = f"{claim_dict.get('pratijna', '')} {claim_dict.get('nigamana', '')}".strip()
        embedding = self.embedder.encode(text_to_embed) if text_to_embed else []

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO beliefs (
                    id, pratijna, hetu, udaharana, upanaya, nigamana, 
                    pramana, confidence, origin, status, created_at, raw_claim, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'held', ?, ?, ?)
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
                json.dumps(claim_dict),
                json.dumps(embedding)
            ))
            conn.commit()
            
        return belief_id

    def sublate_belief(self, old_belief_id: str, new_belief_id: str):
        """
        Marks an old belief as sublated (invalidated) by a newer, stronger belief.
        This embodies 'abādhitatva' (Truth as non-sublation).
        """
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE beliefs 
                SET status = 'sublated', sublated_by = ?, last_audited_at = ? 
                WHERE id = ?
            ''', (new_belief_id, now, old_belief_id))
            conn.commit()

    def store_absence(self, query: str, search_scope: str, tools_used: List[str]) -> str:
        """Stores a logged absence search record."""
        absence_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
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

    def search_beliefs(self, query: str, limit: int = 3, similarity_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Semantic vector search over active held beliefs.
        Falls back to keyword matching if vector similarities are below threshold.
        """
        active_beliefs = self.get_active_beliefs()
        if not active_beliefs:
            return []

        query_vec = self.embedder.encode(query)
        scored_beliefs = []

        for b in active_beliefs:
            emb_raw = b.get("embedding")
            b_emb = json.loads(emb_raw) if emb_raw else None
            if not b_emb:
                b_text = f"{b.get('pratijna', '')} {b.get('nigamana', '')}".strip()
                b_emb = self.embedder.encode(b_text)

            sim = cosine_similarity(query_vec, b_emb)
            if sim >= similarity_threshold:
                b_copy = dict(b)
                b_copy["similarity_score"] = round(sim, 4)
                scored_beliefs.append((sim, b_copy))

        if scored_beliefs:
            scored_beliefs.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored_beliefs[:limit]]

        # Fallback keyword search
        keywords = [word.lower() for word in query.split() if len(word) > 3]
        if not keywords:
            return []
            
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conditions = " OR ".join(["LOWER(pratijna) LIKE ? OR LOWER(nigamana) LIKE ?"] * len(keywords))
            params = []
            for k in keywords:
                params.extend([f"%{k}%", f"%{k}%"])
                
            cursor = conn.execute(f"SELECT * FROM beliefs WHERE status = 'held' AND ({conditions}) LIMIT ?", params + [limit])
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def is_epistemically_stronger(new_claim: Dict[str, Any], old_belief: Dict[str, Any]) -> bool:
        """
        Compares epistemic strength of new claim vs existing belief based on
        Pramāṇa hierarchy (pratyakṣa > anumāna > śabda > anupalabdhi > upamāna > arthāpatti)
        and calibrated confidence.
        """
        pramana_tiers = {
            "pratyaksa": 6,
            "anumana": 5,
            "sabda": 4,
            "anupalabdhi": 3,
            "upamana": 2,
            "arthapatti": 1
        }
        new_p = str(new_claim.get("pramana", "")).lower()
        old_p = str(old_belief.get("pramana", "")).lower()
        
        new_tier = pramana_tiers.get(new_p, 0)
        old_tier = pramana_tiers.get(old_p, 0)
        
        new_conf = float(new_claim.get("confidence", 0.0))
        old_conf = float(old_belief.get("confidence", 0.0))
        
        if new_tier > old_tier:
            return True
        elif new_tier == old_tier and new_conf > old_conf:
            return True
        return False

    def find_contradictions(self, claim_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Searches active held beliefs for propositions that semantically or logically
        contradict the new claim on the same topic using semantic embeddings and polarity analysis.
        """
        query_text = f"{claim_dict.get('pratijna', '')} {claim_dict.get('nigamana', '')}".lower()
        active_beliefs = self.get_active_beliefs()
        contradictions = []

        stop_words = {"this", "that", "with", "from", "have", "were", "been", "which", "about", "there", "their"}
        new_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", query_text) if w not in stop_words}
        if not new_words:
            return []

        query_vec = self.embedder.encode(query_text)

        opposing_pairs = [
            ("increase", "decrease"),
            ("increases", "decreases"),
            ("effective", "ineffective"),
            ("safe", "unsafe"),
            ("causes", "prevents"),
            ("reduces", "increases"),
            ("beneficial", "harmful"),
            ("true", "false"),
            ("valid", "invalid"),
        ]

        negation_markers = ["not", "no", "never", "cannot", "fails", "doesn't", "does not", "didn't", "prevent"]

        for belief in active_beliefs:
            old_text = f"{belief.get('pratijna', '')} {belief.get('nigamana', '')}".lower()
            old_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", old_text) if w not in stop_words}
            
            # Check topical alignment via semantic vector similarity or keyword overlap
            emb_raw = belief.get("embedding")
            b_emb = json.loads(emb_raw) if emb_raw else None
            if not b_emb:
                b_emb = self.embedder.encode(old_text)
            
            vec_sim = cosine_similarity(query_vec, b_emb)
            shared_words = new_words.intersection(old_words)
            keyword_overlap = len(shared_words) >= 2 or (len(shared_words) / max(1, len(new_words.union(old_words)))) >= 0.25

            # Must be topically relevant (either vector similarity > 0.40 or keyword overlap)
            if vec_sim < 0.40 and not keyword_overlap:
                continue

            is_contradiction = False

            # Check opposing semantic antonyms
            for pos, neg in opposing_pairs:
                if (pos in query_text and neg in old_text) or (neg in query_text and pos in old_text):
                    is_contradiction = True
                    break

            # Check negation divergence
            if not is_contradiction:
                new_has_neg = any(m in query_text for m in negation_markers)
                old_has_neg = any(m in old_text for m in negation_markers)
                if new_has_neg != old_has_neg:
                    is_contradiction = True

            if is_contradiction:
                contradictions.append(belief)

        return contradictions

