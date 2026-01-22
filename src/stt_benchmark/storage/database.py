"""SQLite storage for benchmark results."""

import json
from datetime import datetime
from pathlib import Path

import aiosqlite
from loguru import logger

from stt_benchmark.config import get_config
from stt_benchmark.models import (
    AudioSample,
    BenchmarkResult,
    BenchmarkRun,
    GroundTruth,
    SemanticError,
    SemanticWERTrace,
    ServiceName,
    WERMetrics,
)


class Database:
    """SQLite database for storing benchmark results."""

    def __init__(self, db_path: Path | None = None):
        config = get_config()
        self.db_path = db_path or config.results_db
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row

        await self._create_tables()
        logger.debug(f"Database initialized at {self.db_path}")

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        await self._conn.executescript(
            """
            -- Audio samples table
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                audio_path TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                language TEXT DEFAULT 'eng',
                dataset_index INTEGER NOT NULL
            );

            -- Benchmark results table
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                service_name TEXT NOT NULL,
                model_name TEXT,
                ttfb_seconds REAL,
                transcription TEXT,
                audio_duration_seconds REAL NOT NULL,
                timestamp TEXT NOT NULL,
                error TEXT,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
                UNIQUE(sample_id, service_name, model_name)
            );

            -- Benchmark runs table
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                services TEXT NOT NULL,
                num_samples INTEGER NOT NULL,
                config_snapshot TEXT
            );

            -- Ground truth table
            CREATE TABLE IF NOT EXISTS ground_truth (
                sample_id TEXT PRIMARY KEY REFERENCES samples(sample_id),
                text TEXT NOT NULL,
                model_used TEXT NOT NULL DEFAULT 'gemini-3-flash-preview',
                generated_at TEXT NOT NULL
            );

            -- Semantic WER metrics table
            CREATE TABLE IF NOT EXISTS wer_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                service_name TEXT NOT NULL,
                model_name TEXT,
                wer REAL NOT NULL,
                substitutions INTEGER NOT NULL,
                deletions INTEGER NOT NULL,
                insertions INTEGER NOT NULL,
                reference_words INTEGER NOT NULL,
                errors TEXT,
                normalized_reference TEXT,
                normalized_hypothesis TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
                UNIQUE(sample_id, service_name, model_name)
            );

            -- Semantic WER reasoning traces table
            CREATE TABLE IF NOT EXISTS semantic_wer_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                service_name TEXT NOT NULL,
                model_name TEXT,
                session_id TEXT NOT NULL,
                conversation_trace TEXT NOT NULL,
                tool_calls TEXT NOT NULL,
                normalized_reference TEXT,
                normalized_hypothesis TEXT,
                wer REAL NOT NULL,
                substitutions INTEGER NOT NULL,
                deletions INTEGER NOT NULL,
                insertions INTEGER NOT NULL,
                reference_words INTEGER NOT NULL,
                errors TEXT,
                duration_ms INTEGER,
                num_turns INTEGER NOT NULL DEFAULT 1,
                model_used TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
                UNIQUE(sample_id, service_name, model_name)
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_results_service ON results(service_name);
            CREATE INDEX IF NOT EXISTS idx_results_sample ON results(sample_id);
            CREATE INDEX IF NOT EXISTS idx_wer_service ON wer_metrics(service_name);
            CREATE INDEX IF NOT EXISTS idx_traces_service ON semantic_wer_traces(service_name);
            """
        )
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ========== Sample Operations ==========

    async def insert_sample(self, sample: AudioSample) -> None:
        """Insert a single audio sample."""
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO samples (sample_id, audio_path, duration_seconds, language, dataset_index)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                sample.sample_id,
                sample.audio_path,
                sample.duration_seconds,
                sample.language,
                sample.dataset_index,
            ),
        )
        await self._conn.commit()

    async def insert_samples_batch(self, samples: list[AudioSample]) -> None:
        """Insert multiple audio samples in a batch."""
        await self._conn.executemany(
            """
            INSERT OR REPLACE INTO samples (sample_id, audio_path, duration_seconds, language, dataset_index)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (s.sample_id, s.audio_path, s.duration_seconds, s.language, s.dataset_index)
                for s in samples
            ],
        )
        await self._conn.commit()

    async def get_sample(self, sample_id: str) -> AudioSample | None:
        """Get a sample by ID."""
        cursor = await self._conn.execute("SELECT * FROM samples WHERE sample_id = ?", (sample_id,))
        row = await cursor.fetchone()
        if row:
            return AudioSample(
                sample_id=row["sample_id"],
                audio_path=row["audio_path"],
                duration_seconds=row["duration_seconds"],
                language=row["language"],
                dataset_index=row["dataset_index"],
            )
        return None

    async def get_all_samples(self) -> list[AudioSample]:
        """Get all samples."""
        cursor = await self._conn.execute("SELECT * FROM samples ORDER BY dataset_index")
        rows = await cursor.fetchall()
        return [
            AudioSample(
                sample_id=row["sample_id"],
                audio_path=row["audio_path"],
                duration_seconds=row["duration_seconds"],
                language=row["language"],
                dataset_index=row["dataset_index"],
            )
            for row in rows
        ]

    async def get_sample_count(self) -> int:
        """Get the number of samples."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM samples")
        row = await cursor.fetchone()
        return row[0]

    # ========== Result Operations ==========

    async def insert_result(self, result: BenchmarkResult) -> None:
        """Insert a benchmark result."""
        # Use empty string instead of NULL for model_name to ensure UNIQUE constraint works
        model_name = result.model_name or ""
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO results
            (sample_id, service_name, model_name, ttfb_seconds, transcription,
             audio_duration_seconds, timestamp, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.sample_id,
                result.service_name.value,
                model_name,
                result.ttfb_seconds,
                result.transcription,
                result.audio_duration_seconds,
                result.timestamp.isoformat(),
                result.error,
            ),
        )
        await self._conn.commit()

    async def insert_results_batch(self, results: list[BenchmarkResult]) -> None:
        """Insert multiple results in a batch."""
        # Use empty string instead of NULL for model_name to ensure UNIQUE constraint works
        await self._conn.executemany(
            """
            INSERT OR REPLACE INTO results
            (sample_id, service_name, model_name, ttfb_seconds, transcription,
             audio_duration_seconds, timestamp, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.sample_id,
                    r.service_name.value,
                    r.model_name or "",  # Normalize NULL to empty string
                    r.ttfb_seconds,
                    r.transcription,
                    r.audio_duration_seconds,
                    r.timestamp.isoformat(),
                    r.error,
                )
                for r in results
            ],
        )
        await self._conn.commit()

    async def get_results_for_service(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> list[BenchmarkResult]:
        """Get all results for a service."""
        if model_name:
            cursor = await self._conn.execute(
                "SELECT * FROM results WHERE service_name = ? AND model_name = ?",
                (service_name.value, model_name),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM results WHERE service_name = ?",
                (service_name.value,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_result(row) for row in rows]

    async def get_samples_without_results(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> list[AudioSample]:
        """Get samples that don't have results for a service."""
        if model_name:
            cursor = await self._conn.execute(
                """
                SELECT s.* FROM samples s
                LEFT JOIN results r ON s.sample_id = r.sample_id
                    AND r.service_name = ? AND r.model_name = ?
                WHERE r.id IS NULL
                ORDER BY s.dataset_index
                """,
                (service_name.value, model_name),
            )
        else:
            cursor = await self._conn.execute(
                """
                SELECT s.* FROM samples s
                LEFT JOIN results r ON s.sample_id = r.sample_id
                    AND r.service_name = ?
                WHERE r.id IS NULL
                ORDER BY s.dataset_index
                """,
                (service_name.value,),
            )
        rows = await cursor.fetchall()
        return [
            AudioSample(
                sample_id=row["sample_id"],
                audio_path=row["audio_path"],
                duration_seconds=row["duration_seconds"],
                language=row["language"],
                dataset_index=row["dataset_index"],
            )
            for row in rows
        ]

    def _row_to_result(self, row) -> BenchmarkResult:
        """Convert a database row to a BenchmarkResult."""
        return BenchmarkResult(
            sample_id=row["sample_id"],
            service_name=ServiceName(row["service_name"]),
            model_name=row["model_name"],
            ttfb_seconds=row["ttfb_seconds"],
            transcription=row["transcription"],
            audio_duration_seconds=row["audio_duration_seconds"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            error=row["error"],
        )

    # ========== Run Operations ==========

    async def insert_run(self, run: BenchmarkRun) -> None:
        """Insert a benchmark run."""
        await self._conn.execute(
            """
            INSERT INTO runs (run_id, started_at, completed_at, services, num_samples, config_snapshot)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.started_at.isoformat(),
                run.completed_at.isoformat() if run.completed_at else None,
                json.dumps([s.value for s in run.services]),
                run.num_samples,
                json.dumps(run.config_snapshot) if run.config_snapshot else None,
            ),
        )
        await self._conn.commit()

    async def update_run_completed(self, run_id: str) -> None:
        """Mark a run as completed."""
        await self._conn.execute(
            "UPDATE runs SET completed_at = ? WHERE run_id = ?",
            (datetime.now().isoformat(), run_id),
        )
        await self._conn.commit()

    async def get_all_results(self) -> list[BenchmarkResult]:
        """Get all benchmark results."""
        cursor = await self._conn.execute("SELECT * FROM results ORDER BY timestamp DESC")
        rows = await cursor.fetchall()
        return [self._row_to_result(row) for row in rows]

    # ========== Ground Truth Operations ==========

    async def insert_ground_truth(self, gt: GroundTruth) -> None:
        """Insert a ground truth transcription."""
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO ground_truth (sample_id, text, model_used, generated_at)
            VALUES (?, ?, ?, ?)
            """,
            (gt.sample_id, gt.text, gt.model_used, gt.generated_at.isoformat()),
        )
        await self._conn.commit()

    async def get_ground_truth(self, sample_id: str) -> GroundTruth | None:
        """Get ground truth for a sample."""
        cursor = await self._conn.execute(
            "SELECT * FROM ground_truth WHERE sample_id = ?", (sample_id,)
        )
        row = await cursor.fetchone()
        if row:
            return GroundTruth(
                sample_id=row["sample_id"],
                text=row["text"],
                model_used=row["model_used"],
                generated_at=datetime.fromisoformat(row["generated_at"]),
            )
        return None

    async def get_samples_without_ground_truth(self) -> list[AudioSample]:
        """Get samples that don't have ground truth."""
        cursor = await self._conn.execute(
            """
            SELECT s.* FROM samples s
            LEFT JOIN ground_truth gt ON s.sample_id = gt.sample_id
            WHERE gt.sample_id IS NULL
            ORDER BY s.dataset_index
            """
        )
        rows = await cursor.fetchall()
        return [
            AudioSample(
                sample_id=row["sample_id"],
                audio_path=row["audio_path"],
                duration_seconds=row["duration_seconds"],
                language=row["language"],
                dataset_index=row["dataset_index"],
            )
            for row in rows
        ]

    async def get_ground_truth_count(self) -> int:
        """Get number of ground truth entries."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM ground_truth")
        row = await cursor.fetchone()
        return row[0]

    async def clear_all_ground_truths(self) -> int:
        """Clear all ground truth entries.

        Returns:
            Number of entries deleted.
        """
        cursor = await self._conn.execute("SELECT COUNT(*) FROM ground_truth")
        row = await cursor.fetchone()
        count = row[0]

        await self._conn.execute("DELETE FROM ground_truth")
        await self._conn.commit()

        return count

    # ========== WER Metrics Operations ==========

    async def insert_wer_metrics(self, metrics: WERMetrics) -> None:
        """Insert semantic WER metrics."""
        # Use empty string instead of NULL for model_name to ensure UNIQUE constraint works
        model_name = metrics.model_name or ""
        # Serialize errors to JSON if present
        errors_json = None
        if metrics.errors:
            errors_json = json.dumps([e.model_dump() for e in metrics.errors])

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO wer_metrics
            (sample_id, service_name, model_name, wer, substitutions,
             deletions, insertions, reference_words, errors,
             normalized_reference, normalized_hypothesis, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics.sample_id,
                metrics.service_name.value,
                model_name,
                metrics.wer,
                metrics.substitutions,
                metrics.deletions,
                metrics.insertions,
                metrics.reference_words,
                errors_json,
                metrics.normalized_reference,
                metrics.normalized_hypothesis,
                metrics.timestamp.isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_wer_metrics_for_service(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> list[WERMetrics]:
        """Get all WER metrics for a service."""
        if model_name:
            cursor = await self._conn.execute(
                "SELECT * FROM wer_metrics WHERE service_name = ? AND model_name = ?",
                (service_name.value, model_name),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM wer_metrics WHERE service_name = ?",
                (service_name.value,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_wer_metrics(row) for row in rows]

    async def delete_wer_metrics_for_service(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> int:
        """Delete all WER metrics for a service.

        Returns:
            Number of rows deleted.
        """
        if model_name:
            cursor = await self._conn.execute(
                "DELETE FROM wer_metrics WHERE service_name = ? AND model_name = ?",
                (service_name.value, model_name),
            )
        else:
            cursor = await self._conn.execute(
                "DELETE FROM wer_metrics WHERE service_name = ?",
                (service_name.value,),
            )
        await self._conn.commit()
        return cursor.rowcount

    async def get_samples_without_wer(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> list[AudioSample]:
        """Get samples that have results but no WER metrics for a service."""
        if model_name:
            cursor = await self._conn.execute(
                """
                SELECT s.* FROM samples s
                INNER JOIN results r ON s.sample_id = r.sample_id
                    AND r.service_name = ? AND r.model_name = ?
                INNER JOIN ground_truth gt ON s.sample_id = gt.sample_id
                LEFT JOIN wer_metrics w ON s.sample_id = w.sample_id
                    AND w.service_name = ? AND w.model_name = ?
                WHERE w.id IS NULL AND r.transcription IS NOT NULL
                ORDER BY s.dataset_index
                """,
                (service_name.value, model_name, service_name.value, model_name),
            )
        else:
            cursor = await self._conn.execute(
                """
                SELECT s.* FROM samples s
                INNER JOIN results r ON s.sample_id = r.sample_id
                    AND r.service_name = ?
                INNER JOIN ground_truth gt ON s.sample_id = gt.sample_id
                LEFT JOIN wer_metrics w ON s.sample_id = w.sample_id
                    AND w.service_name = ?
                WHERE w.id IS NULL AND r.transcription IS NOT NULL
                ORDER BY s.dataset_index
                """,
                (service_name.value, service_name.value),
            )
        rows = await cursor.fetchall()
        return [
            AudioSample(
                sample_id=row["sample_id"],
                audio_path=row["audio_path"],
                duration_seconds=row["duration_seconds"],
                language=row["language"],
                dataset_index=row["dataset_index"],
            )
            for row in rows
        ]

    def _row_to_wer_metrics(self, row) -> WERMetrics:
        """Convert a database row to WERMetrics."""
        # Parse errors from JSON if present
        errors = None
        if row["errors"]:
            errors_data = json.loads(row["errors"])
            errors = [SemanticError(**e) for e in errors_data]

        return WERMetrics(
            sample_id=row["sample_id"],
            service_name=ServiceName(row["service_name"]),
            model_name=row["model_name"] or None,
            wer=row["wer"],
            substitutions=row["substitutions"],
            deletions=row["deletions"],
            insertions=row["insertions"],
            reference_words=row["reference_words"],
            errors=errors,
            normalized_reference=row["normalized_reference"],
            normalized_hypothesis=row["normalized_hypothesis"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    # ========== Semantic WER Trace Operations ==========

    async def insert_semantic_wer_trace(self, trace: SemanticWERTrace) -> None:
        """Insert a semantic WER reasoning trace."""
        model_name = trace.model_name or ""
        errors_json = None
        if trace.errors:
            errors_json = json.dumps([e.model_dump() for e in trace.errors])

        await self._conn.execute(
            """
            INSERT OR REPLACE INTO semantic_wer_traces
            (sample_id, service_name, model_name, session_id, conversation_trace,
             tool_calls, normalized_reference, normalized_hypothesis, wer,
             substitutions, deletions, insertions, reference_words, errors,
             duration_ms, num_turns, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.sample_id,
                trace.service_name.value,
                model_name,
                trace.session_id,
                json.dumps(trace.conversation_trace),
                json.dumps(trace.tool_calls),
                trace.normalized_reference,
                trace.normalized_hypothesis,
                trace.wer,
                trace.substitutions,
                trace.deletions,
                trace.insertions,
                trace.reference_words,
                errors_json,
                trace.duration_ms,
                trace.num_turns,
                trace.model_used,
                trace.timestamp.isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_semantic_wer_trace(
        self, sample_id: str, service_name: ServiceName, model_name: str | None = None
    ) -> SemanticWERTrace | None:
        """Get the semantic WER trace for a sample."""
        model_filter = model_name or ""
        cursor = await self._conn.execute(
            """
            SELECT * FROM semantic_wer_traces
            WHERE sample_id = ? AND service_name = ? AND model_name = ?
            """,
            (sample_id, service_name.value, model_filter),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        # Parse errors from JSON if present
        errors = None
        if row["errors"]:
            errors_data = json.loads(row["errors"])
            errors = [SemanticError(**e) for e in errors_data]

        return SemanticWERTrace(
            sample_id=row["sample_id"],
            service_name=ServiceName(row["service_name"]),
            model_name=row["model_name"] or None,
            session_id=row["session_id"],
            conversation_trace=json.loads(row["conversation_trace"]),
            tool_calls=json.loads(row["tool_calls"]),
            normalized_reference=row["normalized_reference"],
            normalized_hypothesis=row["normalized_hypothesis"],
            wer=row["wer"],
            substitutions=row["substitutions"],
            deletions=row["deletions"],
            insertions=row["insertions"],
            reference_words=row["reference_words"],
            errors=errors,
            duration_ms=row["duration_ms"],
            num_turns=row["num_turns"],
            model_used=row["model_used"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    async def delete_semantic_wer_traces_for_service(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> int:
        """Delete all semantic WER traces for a service.

        Returns:
            Number of rows deleted.
        """
        if model_name:
            cursor = await self._conn.execute(
                "DELETE FROM semantic_wer_traces WHERE service_name = ? AND model_name = ?",
                (service_name.value, model_name),
            )
        else:
            cursor = await self._conn.execute(
                "DELETE FROM semantic_wer_traces WHERE service_name = ?",
                (service_name.value,),
            )
        await self._conn.commit()
        return cursor.rowcount

    async def get_result_with_ground_truth(
        self, sample_id: str, service_name: ServiceName, model_name: str | None = None
    ) -> tuple[BenchmarkResult | None, GroundTruth | None]:
        """Get a result and its ground truth for WER calculation."""
        # Get result
        if model_name:
            cursor = await self._conn.execute(
                "SELECT * FROM results WHERE sample_id = ? AND service_name = ? AND model_name = ?",
                (sample_id, service_name.value, model_name),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM results WHERE sample_id = ? AND service_name = ?",
                (sample_id, service_name.value),
            )
        row = await cursor.fetchone()
        result = self._row_to_result(row) if row else None

        # Get ground truth
        gt = await self.get_ground_truth(sample_id)

        return result, gt

    async def get_wer_metrics_count(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> int:
        """Get count of WER metrics for a service."""
        model_filter = model_name if model_name else ""
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM wer_metrics WHERE service_name = ? AND model_name = ?",
            (service_name.value, model_filter),
        )
        row = await cursor.fetchone()
        return row[0]

    async def get_services_with_wer_metrics(self) -> list[tuple[ServiceName, str | None]]:
        """Get all services that have WER metrics.

        Returns a list of (service_name, model_name) tuples.
        """
        cursor = await self._conn.execute(
            "SELECT DISTINCT service_name, model_name FROM wer_metrics ORDER BY service_name"
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            try:
                service = ServiceName(row["service_name"])
                model = row["model_name"] if row["model_name"] else None
                result.append((service, model))
            except ValueError:
                # Skip unknown services
                continue
        return result

    async def get_service_summary(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> dict | None:
        """Get summary statistics for a service.

        Returns a dict with wer_mean, wer_median, ttfb_mean, ttfb_median, ttfb_p95, sample_count.
        """
        model_filter = model_name if model_name else ""
        cursor = await self._conn.execute(
            """
            SELECT
                COUNT(*) as sample_count,
                AVG(w.wer) as wer_mean,
                AVG(r.ttfb_seconds) as ttfb_mean
            FROM wer_metrics w
            JOIN results r ON w.sample_id = r.sample_id
                AND w.service_name = r.service_name
                AND w.model_name = r.model_name
            WHERE w.service_name = ? AND w.model_name = ?
            """,
            (service_name.value, model_filter),
        )
        row = await cursor.fetchone()

        if not row or row["sample_count"] == 0:
            return None

        # Get median and percentiles (need all values)
        cursor = await self._conn.execute(
            """
            SELECT w.wer, r.ttfb_seconds as ttfb
            FROM wer_metrics w
            JOIN results r ON w.sample_id = r.sample_id
                AND w.service_name = r.service_name
                AND w.model_name = r.model_name
            WHERE w.service_name = ? AND w.model_name = ?
            ORDER BY w.wer
            """,
            (service_name.value, model_filter),
        )
        rows = await cursor.fetchall()

        wer_values = [r["wer"] for r in rows]
        ttfb_values = [r["ttfb"] for r in rows if r["ttfb"] is not None]

        # Calculate median
        n = len(wer_values)
        wer_median = wer_values[n // 2] if n > 0 else 0

        ttfb_median = 0
        ttfb_p95 = 0
        if ttfb_values:
            ttfb_sorted = sorted(ttfb_values)
            ttfb_median = ttfb_sorted[len(ttfb_sorted) // 2]
            ttfb_p95 = (
                ttfb_sorted[int(len(ttfb_sorted) * 0.95)]
                if len(ttfb_sorted) > 1
                else ttfb_sorted[0]
            )

        return {
            "sample_count": row["sample_count"],
            "wer_mean": row["wer_mean"] or 0,
            "wer_median": wer_median,
            "ttfb_mean": row["ttfb_mean"] or 0,
            "ttfb_median": ttfb_median,
            "ttfb_p95": ttfb_p95,
        }

    async def get_report_data(
        self, service_name: ServiceName, model_name: str | None = None
    ) -> list[dict]:
        """Get all data needed for validation report.

        Returns a list of dicts with sample, result, ground truth, and WER data.
        """
        model_filter = model_name if model_name else ""
        cursor = await self._conn.execute(
            """
            SELECT
                w.sample_id,
                s.duration_seconds as duration,
                r.ttfb_seconds as ttfb,
                w.wer,
                w.substitutions,
                w.deletions,
                w.insertions,
                w.reference_words as ref_words,
                w.normalized_reference,
                w.normalized_hypothesis,
                g.text as ground_truth,
                r.transcription
            FROM wer_metrics w
            JOIN ground_truth g ON w.sample_id = g.sample_id
            JOIN results r ON w.sample_id = r.sample_id
                AND w.service_name = r.service_name
                AND w.model_name = r.model_name
            JOIN samples s ON w.sample_id = s.sample_id
            WHERE w.service_name = ? AND w.model_name = ?
            ORDER BY w.wer DESC
            """,
            (service_name.value, model_filter),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
