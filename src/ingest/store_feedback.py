"""Answer feedback operations for DocStore."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations


class StoreFeedbackMixin:
    FEEDBACK_RATINGS = {"useful", "not_useful"}

    def set_answer_feedback(self, history_id: int, rating: str, note: str = "") -> dict:
        normalized_rating = str(rating or "").strip().lower()
        if normalized_rating not in self.FEEDBACK_RATINGS:
            raise ValueError(f"Unsupported answer feedback rating: {rating}")
        normalized_note = str(note or "").strip()[:500]
        with self._conn() as conn:
            history = conn.execute(
                "SELECT id FROM history WHERE id = ?",
                (int(history_id),),
            ).fetchone()
            if history is None:
                raise KeyError(f"History item not found: {history_id}")
            conn.execute(
                """
                INSERT INTO answer_feedback (history_id, rating, note)
                VALUES (?, ?, ?)
                ON CONFLICT(history_id) DO UPDATE SET
                    rating = excluded.rating,
                    note = excluded.note,
                    updated_at = datetime('now')
                """,
                (int(history_id), normalized_rating, normalized_note),
            )
            row = conn.execute(
                "SELECT * FROM answer_feedback WHERE history_id = ?",
                (int(history_id),),
            ).fetchone()
        return dict(row)

    def get_answer_feedback(self, history_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM answer_feedback WHERE history_id = ?",
                (int(history_id),),
            ).fetchone()
        return dict(row) if row else None

    def get_feedback_summary(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT rating, COUNT(*) AS count
                FROM answer_feedback
                GROUP BY rating
                """
            ).fetchall()
        counts = {"useful": 0, "not_useful": 0}
        for row in rows:
            if row["rating"] in counts:
                counts[row["rating"]] = int(row["count"])
        total = counts["useful"] + counts["not_useful"]
        return {
            **counts,
            "total": total,
            "useful_rate": round(counts["useful"] / total, 3) if total else 0.0,
        }
