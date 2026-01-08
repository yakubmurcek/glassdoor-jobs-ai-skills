#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint manager for resumable large CSV processing.

Provides backup creation, incremental saves, and resume detection
for processing large datasets over multiple sessions.
"""

import atexit
import logging
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Set

import pandas as pd

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints, backups, and resume detection for CSV processing.
    
    Features:
    - Timestamped backups before modifying existing output
    - ID-based tracking of processed rows
    - Atomic writes to prevent corruption (write to temp, then rename)
    - Graceful shutdown on Ctrl+C
    """
    
    def __init__(
        self,
        output_path: Path | str,
        *,
        id_column: str = "id",
        checkpoint_interval: int = 100,
        backup_dir: Path | str | None = None,
    ) -> None:
        """Initialize checkpoint manager.
        
        Args:
            output_path: Path to the output CSV file
            id_column: Column name containing unique row IDs
            checkpoint_interval: Number of rows between checkpoints
            backup_dir: Directory for backups (defaults to output_path.parent/backups)
        """
        self.output_path = Path(output_path)
        self.id_column = id_column
        self.checkpoint_interval = max(1, checkpoint_interval)
        
        if backup_dir is None:
            self.backup_dir = self.output_path.parent / "backups"
        else:
            self.backup_dir = Path(backup_dir)
            
        self._processed_ids: Set[int | str] = set()
        self._pending_df: pd.DataFrame | None = None
        self._rows_since_checkpoint = 0
        self._total_saved = 0
        self._backup_created = False
        self._shutdown_registered = False
        
    def create_backup(self) -> Path | None:
        """Create a timestamped backup of the output file.
        
        Returns:
            Path to backup file, or None if no backup was needed.
        """
        if not self.output_path.exists():
            logger.debug("No existing output file to backup.")
            return None
            
        if self._backup_created:
            logger.debug("Backup already created this session.")
            return None
            
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_name = f"{self.output_path.stem}_{timestamp}.csv.bak"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(self.output_path, backup_path)
        self._backup_created = True
        
        logger.info(f"Created backup: {backup_path}")
        return backup_path
        
    def get_processed_ids(self) -> Set[int | str]:
        """Read existing output file and return set of already-processed IDs.
        
        Returns:
            Set of IDs that have been processed.
        """
        if self._processed_ids:
            return self._processed_ids
            
        if not self.output_path.exists():
            logger.debug("No existing output file found.")
            return set()
            
        try:
            df = pd.read_csv(
                self.output_path, 
                sep=";", 
                usecols=[self.id_column],
                dtype=str,
                low_memory=False,
                encoding="utf-8-sig"
            )
            self._processed_ids = set(df[self.id_column].dropna().unique())
            logger.info(f"Found {len(self._processed_ids)} already-processed IDs in {self.output_path}")
            return self._processed_ids
        except Exception as e:
            logger.warning(f"Could not read existing output file: {e}")
            return set()
            
    def filter_unprocessed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to only rows that haven't been processed yet.
        
        Args:
            df: Input DataFrame with id_column
            
        Returns:
            Filtered DataFrame containing only unprocessed rows.
        """
        processed_ids = self.get_processed_ids()
        
        if not processed_ids:
            logger.info(f"No prior progress found. Processing all {len(df)} rows.")
            return df
            
        # Convert ID column to string for comparison
        df_ids = df[self.id_column].astype(str)
        processed_ids_str = {str(id_) for id_ in processed_ids}
        
        mask = ~df_ids.isin(processed_ids_str)
        filtered_df = df[mask].copy()
        
        skipped = len(df) - len(filtered_df)
        logger.info(
            f"Resuming: Skipping {skipped} already-processed rows. "
            f"{len(filtered_df)} rows remaining."
        )
        
        return filtered_df
        
    def save_checkpoint(self, df: pd.DataFrame, *, force: bool = False) -> bool:
        """Save a checkpoint with new processed rows.
        
        Uses atomic write pattern: write to temp file, then rename.
        
        Args:
            df: DataFrame with newly processed rows to append
            force: Force save regardless of checkpoint interval
            
        Returns:
            True if checkpoint was saved, False if skipped.
        """
        if df.empty:
            return False
            
        self._rows_since_checkpoint += len(df)
        
        # Accumulate pending data
        if self._pending_df is None:
            self._pending_df = df
        else:
            self._pending_df = pd.concat([self._pending_df, df], ignore_index=True)
            
        if not force and self._rows_since_checkpoint < self.checkpoint_interval:
            return False
            
        # Time to save
        return self._write_checkpoint()
        
    def _write_checkpoint(self) -> bool:
        """Actually write the checkpoint to disk."""
        if self._pending_df is None or self._pending_df.empty:
            return False
            
        # Create backup before first write to existing file
        if self.output_path.exists() and not self._backup_created:
            self.create_backup()
            
        # Register shutdown handler on first write
        if not self._shutdown_registered:
            self._register_shutdown_handler()
            
        temp_path = self.output_path.with_suffix(".tmp")
        
        try:
            if self.output_path.exists():
                # Append mode: read existing, concat, write all
                existing_df = pd.read_csv(
                    self.output_path, 
                    sep=";", 
                    dtype=str, 
                    low_memory=False,
                    encoding="utf-8-sig"
                )
                combined_df = pd.concat([existing_df, self._pending_df], ignore_index=True)
            else:
                combined_df = self._pending_df
                
            # Write to temp file first (atomic pattern)
            combined_df.to_csv(temp_path, sep=";", index=False, encoding="utf-8-sig")
            
            # Atomic rename
            temp_path.replace(self.output_path)
            
            # Update tracking
            saved_count = len(self._pending_df)
            self._total_saved += saved_count
            
            # Track new IDs
            new_ids = self._pending_df[self.id_column].astype(str).unique()
            self._processed_ids.update(new_ids)
            
            logger.info(
                f"Checkpoint saved: {saved_count} rows "
                f"(total: {self._total_saved} this session, {len(combined_df)} in file)"
            )
            
            # Reset pending state
            self._pending_df = None
            self._rows_since_checkpoint = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise
            
    def flush(self) -> bool:
        """Force save any pending data.
        
        Call this at the end of processing or on shutdown.
        
        Returns:
            True if data was saved, False if nothing pending.
        """
        if self._pending_df is None or self._pending_df.empty:
            return False
        return self._write_checkpoint()
        
    def _register_shutdown_handler(self) -> None:
        """Register handlers for graceful shutdown on Ctrl+C."""
        def shutdown_handler(signum, frame):
            logger.warning("\nInterrupt received! Saving progress...")
            self.flush()
            logger.info(f"Progress saved. Processed {self._total_saved} rows this session.")
            sys.exit(0)
            
        # Register for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, shutdown_handler)
        
        # Also register atexit for normal exits
        atexit.register(self.flush)
        
        self._shutdown_registered = True
        logger.debug("Shutdown handlers registered.")
        
    @property
    def pending_count(self) -> int:
        """Number of rows pending save."""
        if self._pending_df is None:
            return 0
        return len(self._pending_df)
        
    @property
    def total_saved_this_session(self) -> int:
        """Total rows saved in this session."""
        return self._total_saved
