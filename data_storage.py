import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
from pathlib import Path

class DataStorage:
    """
    Manages data storage and retrieval for shipping tracking operations.

    Stores:
    - Successful interaction patterns
    - Cached tracking results
    - Performance metrics
    - Error logs
    """

    def __init__(self, db_path: str = "shipping_tracking.db"):
        """
        Initialize the data storage.

        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.ensure_database_exists()

    def ensure_database_exists(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tracking results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracking_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    booking_id TEXT NOT NULL,
                    voyage_number TEXT,
                    arrival_date TEXT,
                    vessel_name TEXT,
                    port_of_loading TEXT,
                    port_of_discharge TEXT,
                    status TEXT,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)

            # Interaction patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    website_url TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    success_rate REAL DEFAULT 1.0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Error logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    booking_id TEXT,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    stack_trace TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    async def get_cached_result(self, booking_id: str) -> Optional[Dict]:
        """
        Get cached tracking result if still valid.

        Args:
            booking_id (str): The booking ID to look up

        Returns:
            Optional[Dict]: Cached result or None if not found/expired
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT voyage_number, arrival_date, vessel_name, 
                           port_of_loading, port_of_discharge, status, raw_data
                    FROM tracking_results 
                    WHERE booking_id = ? AND (expires_at IS NULL OR expires_at > datetime('now'))
                    ORDER BY created_at DESC LIMIT 1
                """, (booking_id,))

                result = cursor.fetchone()
                if result:
                    return {
                        "booking_id": booking_id,
                        "voyage_number": result[0],
                        "arrival_date": result[1],
                        "vessel_name": result[2],
                        "port_of_loading": result[3],
                        "port_of_discharge": result[4],
                        "status": result[5],
                        "source": "cache",
                        "retrieved_at": datetime.now().isoformat()
                    }

        except Exception as e:
            await self.log_error(booking_id, "cache_lookup", str(e))

        return None

    async def store_tracking_result(self, tracking_data: Dict, cache_hours: int = 24):
        """
        Store tracking result in the database.

        Args:
            tracking_data (Dict): The tracking result to store
            cache_hours (int): How many hours to cache the result
        """
        try:
            expires_at = datetime.now() + timedelta(hours=cache_hours)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO tracking_results 
                    (booking_id, voyage_number, arrival_date, vessel_name, 
                     port_of_loading, port_of_discharge, status, raw_data, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tracking_data.get("booking_id"),
                    tracking_data.get("voyage_number"),
                    tracking_data.get("arrival_date"),
                    tracking_data.get("vessel_name"),
                    tracking_data.get("port_of_loading"),
                    tracking_data.get("port_of_discharge"),
                    tracking_data.get("status"),
                    json.dumps(tracking_data),
                    expires_at
                ))

                conn.commit()
                print(f"✅ Stored tracking result for {tracking_data.get('booking_id')}")

        except Exception as e:
            await self.log_error(
                tracking_data.get("booking_id", "unknown"), 
                "store_result", 
                str(e)
            )

    async def store_interaction_pattern(self, booking_id: str, result_data: Dict):
        """
        Store successful interaction pattern for future reuse.

        Args:
            booking_id (str): The booking ID that was tracked
            result_data (Dict): The successful tracking result
        """
        if "error" not in result_data:
            await self.store_tracking_result(result_data)

    async def log_error(self, booking_id: str, error_type: str, error_message: str, 
                       stack_trace: str = None):
        """
        Log an error to the database.

        Args:
            booking_id (str): Related booking ID
            error_type (str): Type of error
            error_message (str): Error message
            stack_trace (str): Optional stack trace
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO error_logs (booking_id, error_type, error_message, stack_trace)
                    VALUES (?, ?, ?, ?)
                """, (booking_id, error_type, error_message, stack_trace))

                conn.commit()

        except Exception as e:
            print(f"Failed to log error: {e}")

    async def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.

        Returns:
            Dict: Performance metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get success rate
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN voyage_number IS NOT NULL THEN 1 ELSE 0 END) as successes
                    FROM tracking_results 
                    WHERE created_at > datetime('now', '-7 days')
                """)

                result = cursor.fetchone()
                total, successes = result if result else (0, 0)

                success_rate = (successes / total * 100) if total > 0 else 0

                return {
                    "total_attempts_7_days": total,
                    "successful_attempts_7_days": successes,
                    "success_rate_percent": round(success_rate, 2),
                    "generated_at": datetime.now().isoformat()
                }

        except Exception as e:
            return {"error": str(e)}
