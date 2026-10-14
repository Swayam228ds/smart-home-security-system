"""
database.py — SQLite layer for Smart Home Security Demo
=========================================================
Handles: init, migrations, create visit, update capture info,
read visits, delete all (records + images)
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from config import DB_PATH

log = logging.getLogger(__name__)


# ── Internal helper ──────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    """Open a SQLite connection with row-factory set."""
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── Public API ───────────────────────────────────────────────────────────────

def init_db():
    """
    Create the visits table (and indexes) if they don't exist yet,
    and automatically migrate any existing database schemas to include
    new face-detection and capture-status columns.
    """
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                visit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         DATETIME NOT NULL,
                image_path        TEXT     NOT NULL DEFAULT '',
                human_detected    INTEGER  NOT NULL DEFAULT 1,
                human_count       INTEGER  NOT NULL DEFAULT 1,
                processing_status TEXT     NOT NULL DEFAULT 'pending',

                -- Face detection & quality metrics
                face_detected     INTEGER  DEFAULT 0,
                recognized_person TEXT,
                face_confidence   REAL,
                face_bbox         TEXT,
                face_quality      REAL,
                capture_status    TEXT     NOT NULL DEFAULT 'waiting_for_face'
            )
        """)

        # ── Migration check for pre-existing databases ───────────────────────
        cur = conn.execute("PRAGMA table_info(visits)")
        existing_cols = {row["name"] for row in cur.fetchall()}

        if "face_bbox" not in existing_cols:
            log.info("Migrating database: adding 'face_bbox' column to visits table.")
            conn.execute("ALTER TABLE visits ADD COLUMN face_bbox TEXT")

        if "face_quality" not in existing_cols:
            log.info("Migrating database: adding 'face_quality' column to visits table.")
            conn.execute("ALTER TABLE visits ADD COLUMN face_quality REAL")

        if "capture_status" not in existing_cols:
            log.info("Migrating database: adding 'capture_status' column to visits table.")
            conn.execute("ALTER TABLE visits ADD COLUMN capture_status TEXT NOT NULL DEFAULT 'captured'")

        if "person_id" not in existing_cols:
            log.info("Migrating database: adding 'person_id' column to visits table.")
            conn.execute("ALTER TABLE visits ADD COLUMN person_id INTEGER")

        # ── Create people table ───────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS people (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)

        # Useful indexes for dashboard queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_visits_timestamp "
            "ON visits(timestamp DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_visits_status "
            "ON visits(processing_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_visits_capture_status "
            "ON visits(capture_status)"
        )
        conn.commit()
    log.info(f"Database ready → {DB_PATH}")


def create_visit(
    timestamp: datetime,
    image_path: str = "",
    human_count: int = 1,
    capture_status: str = "waiting_for_face"
) -> int:
    """
    Insert one visit record and return the new visit_id.
    Initiates in 'waiting_for_face' state when triggered by YOLO.
    """
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO visits (
                timestamp, image_path, human_detected, human_count,
                processing_status, capture_status, face_detected
            )
            VALUES (?, ?, 1, ?, 'pending', ?, 0)
            """,
            (
                timestamp.isoformat(sep=" ", timespec="seconds"),
                image_path,
                human_count,
                capture_status
            ),
        )
        conn.commit()
        return cur.lastrowid


def update_visit_capture(
    visit_id: int,
    image_path: str,
    capture_status: str = "captured",
    face_detected: int = 1,
    face_confidence: Optional[float] = None,
    face_bbox: Optional[str] = None,
    face_quality: Optional[float] = None,
):
    """
    Update a visit record after MediaPipe face capture or timeout.
    """
    with _conn() as conn:
        conn.execute(
            """
            UPDATE visits
            SET image_path      = ?,
                capture_status  = ?,
                face_detected   = ?,
                face_confidence = ?,
                face_bbox       = ?,
                face_quality    = ?
            WHERE visit_id = ?
            """,
            (
                image_path,
                capture_status,
                face_detected,
                face_confidence,
                face_bbox,
                face_quality,
                visit_id
            ),
        )
        conn.commit()


def update_capture_status(visit_id: int, capture_status: str):
    """Update only the capture_status of a visit."""
    with _conn() as conn:
        conn.execute(
            "UPDATE visits SET capture_status = ? WHERE visit_id = ?",
            (capture_status, visit_id),
        )
        conn.commit()


def update_image_path(visit_id: int, image_path: str):
    """Update image_path after the JPEG has been written to disk (backward compatibility)."""
    with _conn() as conn:
        conn.execute(
            "UPDATE visits SET image_path = ? WHERE visit_id = ?",
            (image_path, visit_id),
        )
        conn.commit()


def get_visits() -> List[Dict[str, Any]]:
    """Return all visits, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM visits ORDER BY visit_id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_visit(visit_id: int) -> Optional[Dict[str, Any]]:
    """Return a single visit by ID, or None if not found."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM visits WHERE visit_id = ?", (visit_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_all_visits() -> int:
    """
    Delete every visit record AND matching image files from disk.

    Returns:
        Number of records deleted.
    """
    with _conn() as conn:
        rows  = conn.execute("SELECT image_path FROM visits").fetchall()
        count = len(rows)

        for row in rows:
            img_str = row["image_path"]
            if img_str:
                img = Path(img_str)
                if img.exists():
                    img.unlink()
                    log.info(f"  Deleted image → {img}")
                # Also delete face crop if exists
                face_crop = img.with_name(f"{img.stem}_face{img.suffix}")
                if face_crop.exists():
                    face_crop.unlink()
                    log.info(f"  Deleted face crop → {face_crop}")

        conn.execute("DELETE FROM visits")
        conn.commit()

    log.info(f"Cleared {count} visit record(s) from {DB_PATH}.")
    return count

# -- Face Recognition Queries ------------------------------------------------

def get_or_create_person(name: str) -> int:
    with _conn() as conn:
        row = conn.execute("SELECT person_id FROM people WHERE name = ?", (name,)).fetchone()
        if row:
            return row['person_id']
        cur = conn.execute("INSERT INTO people (name) VALUES (?)", (name,))
        conn.commit()
        return cur.lastrowid

def assign_person_to_visit(visit_id: int, person_id: int, name: str):
    with _conn() as conn:
        conn.execute(
            "UPDATE visits SET person_id = ?, recognized_person = ? WHERE visit_id = ?",
            (person_id, name, visit_id)
        )
        conn.commit()

def get_all_labeled_faces() -> List[Dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT v.visit_id, v.image_path, v.person_id, p.name 
            FROM visits v
            JOIN people p ON v.person_id = p.person_id
            WHERE v.image_path != '' AND v.person_id IS NOT NULL
            """
        ).fetchall()
    return [dict(r) for r in rows]
