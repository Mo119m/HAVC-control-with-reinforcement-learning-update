"""
Google Drive Backup Utility

Automatically backs up pipeline outputs to Google Drive after each stage
and supports resuming from saved checkpoints.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DriveBackup:
    """Manages backup and restore of pipeline outputs to Google Drive"""

    def __init__(self, drive_path: Optional[str] = None, local_path: str = "./pipeline_output"):
        """
        Initialize Drive backup manager.

        Args:
            drive_path: Path to Google Drive backup directory (e.g., /content/drive/MyDrive/HVAC-RL-Backup)
            local_path: Local pipeline output directory
        """
        self.drive_path = Path(drive_path) if drive_path else None
        self.local_path = Path(local_path)
        self.enabled = drive_path is not None

        if self.enabled:
            self.drive_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Drive backup enabled: {self.drive_path}")
        else:
            logger.info("Drive backup disabled (no drive_path provided)")

    def backup_stage(self, stage_name: str, stage_dir: str) -> bool:
        """
        Backup a specific stage's outputs to Drive.

        Args:
            stage_name: Name of the stage (e.g., "01_ppo_training")
            stage_dir: Local directory containing stage outputs

        Returns:
            True if backup successful, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Skipping backup for {stage_name} (Drive backup disabled)")
            return True

        try:
            source = Path(stage_dir)
            if not source.exists():
                logger.warning(f"Stage directory not found: {source}")
                return False

            # Create backup destination
            dest = self.drive_path / stage_name

            # Backup the stage directory
            logger.info(f"📤 Backing up {stage_name} to Drive...")
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)

            # Save metadata
            metadata = {
                "stage": stage_name,
                "timestamp": datetime.now().isoformat(),
                "local_path": str(source),
                "drive_path": str(dest),
            }

            metadata_file = dest / "backup_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Backup completed: {dest}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup {stage_name}: {e}")
            return False

    def restore_stage(self, stage_name: str, stage_dir: str) -> bool:
        """
        Restore a stage's outputs from Drive.

        Args:
            stage_name: Name of the stage (e.g., "01_ppo_training")
            stage_dir: Local directory to restore to

        Returns:
            True if restore successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            source = self.drive_path / stage_name
            if not source.exists():
                logger.debug(f"No backup found for {stage_name}")
                return False

            dest = Path(stage_dir)

            logger.info(f"📥 Restoring {stage_name} from Drive...")

            # Remove existing local directory if it exists
            if dest.exists():
                shutil.rmtree(dest)

            # Copy from Drive
            shutil.copytree(source, dest)

            # Verify metadata
            metadata_file = dest / "backup_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✅ Restored from backup (created: {metadata.get('timestamp', 'unknown')})")
            else:
                logger.info(f"✅ Restored from backup")

            return True

        except Exception as e:
            logger.error(f"Failed to restore {stage_name}: {e}")
            return False

    def check_stage_exists(self, stage_name: str) -> bool:
        """
        Check if a stage backup exists in Drive.

        Args:
            stage_name: Name of the stage

        Returns:
            True if backup exists, False otherwise
        """
        if not self.enabled:
            return False

        backup_dir = self.drive_path / stage_name
        return backup_dir.exists() and backup_dir.is_dir()

    def backup_reports(self, reports_dir: str) -> bool:
        """
        Backup visualization reports to Drive.

        Args:
            reports_dir: Local reports directory

        Returns:
            True if backup successful, False otherwise
        """
        return self.backup_stage("reports", reports_dir)

    def get_backup_info(self) -> Dict[str, Any]:
        """
        Get information about all backups in Drive.

        Returns:
            Dictionary with backup information
        """
        if not self.enabled:
            return {"enabled": False, "backups": []}

        backups = []

        for stage_dir in self.drive_path.iterdir():
            if not stage_dir.is_dir():
                continue

            metadata_file = stage_dir / "backup_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                backups.append(metadata)
            else:
                backups.append({
                    "stage": stage_dir.name,
                    "timestamp": "unknown",
                    "drive_path": str(stage_dir),
                })

        return {
            "enabled": True,
            "drive_path": str(self.drive_path),
            "backups": backups
        }

    def print_backup_status(self):
        """Print current backup status"""
        info = self.get_backup_info()

        if not info["enabled"]:
            logger.info("📁 Drive backup: DISABLED")
            return

        logger.info(f"📁 Drive backup: ENABLED")
        logger.info(f"   Location: {info['drive_path']}")

        if info["backups"]:
            logger.info(f"   Saved stages:")
            for backup in info["backups"]:
                logger.info(f"     ✅ {backup['stage']} ({backup.get('timestamp', 'unknown')})")
        else:
            logger.info("   No backups found")
