"""SQLite database helper for trade history and AI memory."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.logger import logger


class TradeDatabase:
    """SQLite wrapper for trade persistence and AI memory."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------ #
    # Connection helpers
    # ------------------------------------------------------------------ #

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy-create and return a database connection."""
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            logger.info("Database connection opened: %s", self._db_path)
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                pair            TEXT NOT NULL,
                timeframe       TEXT NOT NULL,
                decision        TEXT NOT NULL,
                confidence      REAL,
                entry_price     REAL,
                stop_loss       REAL,
                take_profit     REAL,
                risk_reward     REAL,
                technical_score REAL,
                correlation_score REAL,
                trend           TEXT,
                reason          TEXT,
                profit_loss     REAL DEFAULT 0.0,
                outcome         TEXT DEFAULT 'pending'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS indicator_memory (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                pair            TEXT NOT NULL,
                indicator_name  TEXT NOT NULL,
                weight          REAL NOT NULL,
                accuracy        REAL DEFAULT 0.0,
                sample_count    INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id        INTEGER REFERENCES trade_history(id),
                timestamp       TEXT NOT NULL,
                was_correct     INTEGER NOT NULL,
                notes           TEXT
            )
        """)
        self.conn.commit()
        logger.info("Database tables verified/created")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    # ------------------------------------------------------------------ #
    # Trade CRUD
    # ------------------------------------------------------------------ #

    def save_trade(self, trade: dict[str, Any]) -> int:
        """Insert a trade record and return the new row id.

        Args:
            trade: Dictionary with trade fields matching the table schema.

        Returns:
            The auto-generated row id.
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO trade_history
                (timestamp, pair, timeframe, decision, confidence, entry_price,
                 stop_loss, take_profit, risk_reward, technical_score,
                 correlation_score, trend, reason, profit_loss, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                trade.get("pair", ""),
                trade.get("timeframe", ""),
                trade.get("decision", ""),
                trade.get("confidence", 0.0),
                trade.get("entry", 0.0),
                trade.get("sl", 0.0),
                trade.get("tp", 0.0),
                trade.get("risk_reward", 0.0),
                trade.get("technical_score", 0.0),
                trade.get("correlation_score", 0.0),
                trade.get("trend", ""),
                trade.get("reason", ""),
                trade.get("profit_loss", 0.0),
                "pending",
            ),
        )
        self.conn.commit()
        row_id = cur.lastrowid
        logger.info("Trade saved: id=%s pair=%s decision=%s", row_id, trade.get("pair"), trade.get("decision"))
        return row_id  # type: ignore[return-value]

    def update_trade_outcome(
        self,
        trade_id: int,
        outcome: str,
        profit_loss: float,
    ) -> None:
        """Mark a trade's outcome after it has been resolved."""
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE trade_history SET outcome=?, profit_loss=? WHERE id=?",
            (outcome, profit_loss, trade_id),
        )
        self.conn.commit()
        logger.info("Trade %s outcome updated: %s P/L=%.5f", trade_id, outcome, profit_loss)

    def get_trades(
        self,
        pair: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch recent trades, optionally filtered by pair.

        Args:
            pair: Currency pair filter (e.g. ``'EURUSD'``).
            limit: Maximum rows to return.

        Returns:
            List of trade dictionaries.
        """
        if pair:
            rows = self.conn.execute(
                "SELECT * FROM trade_history WHERE pair=? ORDER BY id DESC LIMIT ?",
                (pair, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM trade_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_trades(self) -> list[dict[str, Any]]:
        """Return every trade in the database."""
        rows = self.conn.execute("SELECT * FROM trade_history ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # AI Memory
    # ------------------------------------------------------------------ #

    def save_indicator_memory(self, record: dict[str, Any]) -> None:
        """Upsert indicator weight memory for adaptive learning."""
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO indicator_memory
                (timestamp, pair, indicator_name, weight, accuracy, sample_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                weight=excluded.weight,
                accuracy=excluded.accuracy,
                sample_count=excluded.sample_count
            """,
            (
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                record.get("pair", ""),
                record.get("indicator_name", ""),
                record.get("weight", 0.0),
                record.get("accuracy", 0.0),
                record.get("sample_count", 0),
            ),
        )
        self.conn.commit()

    def get_indicator_memory(
        self,
        pair: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve stored indicator weight memories."""
        if pair:
            rows = self.conn.execute(
                "SELECT * FROM indicator_memory WHERE pair=? ORDER BY id DESC",
                (pair,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM indicator_memory ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def save_evaluation(self, trade_id: int, was_correct: bool, notes: str = "") -> None:
        """Log whether a past trade's signal was correct for AI learning."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO evaluation_log (trade_id, timestamp, was_correct, notes)
            VALUES (?, ?, ?, ?)
            """,
            (trade_id, now, int(was_correct), notes),
        )
        self.conn.commit()
        logger.info("Evaluation saved: trade_id=%s correct=%s", trade_id, was_correct)

    def evaluate_pending_trades(self, current_price: float, pair: str) -> None:
        """Evaluate pending trades against the current price.

        Args:
            current_price: Current market price for the pair.
            pair: Currency pair to evaluate.
        """
        pending = self.conn.execute(
            "SELECT * FROM trade_history WHERE pair=? AND outcome='pending'",
            (pair,),
        ).fetchall()

        for row in pending:
            trade = dict(row)
            trade_id = trade["id"]
            decision = trade["decision"]
            entry = trade["entry_price"]
            tp = trade["take_profit"]
            sl = trade["stop_loss"]

            if not entry or not tp or not sl:
                continue

            profit_loss = 0.0
            outcome = "pending"
            was_correct = False

            if decision in ("Strong Buy", "Buy"):
                if current_price >= tp:
                    profit_loss = tp - entry
                    outcome = "win"
                    was_correct = True
                elif current_price <= sl:
                    profit_loss = sl - entry
                    outcome = "loss"
            elif decision in ("Strong Sell", "Sell"):
                if current_price <= tp:
                    profit_loss = entry - tp
                    outcome = "win"
                    was_correct = True
                elif current_price >= sl:
                    profit_loss = entry - sl
                    outcome = "loss"
            else:
                outcome = "no_trade"

            if outcome != "pending":
                self.update_trade_outcome(trade_id, outcome, profit_loss)
                self.save_evaluation(trade_id, was_correct, f"Auto-evaluated at {current_price}")