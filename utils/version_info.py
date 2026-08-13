"""
Version and Release Information Utilities

Provides utilities for tracking and logging version information during pipeline execution.
Automatically retrieves git commit info and creates deployment manifests.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class VersionInfo:
    """Manages version and deployment information."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize version info tracker.
        
        Args:
            project_root: Root directory of the project (defaults to parent of this file)
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root)
        self.version_file = self.project_root / "VERSION.txt"
        self.manifest_file = self.project_root / "MANIFEST.json"
    
    def get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def get_git_branch(self) -> Optional[str]:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def get_git_tag(self) -> Optional[str]:
        """Get current git version tag if available."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def read_version_file(self) -> Dict[str, str]:
        """
        Read VERSION.txt if it exists (from release archive).
        
        Returns:
            Dictionary of version info
        """
        version_info = {}
        if self.version_file.exists():
            with open(self.version_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        version_info[key.strip().lower().replace(" ", "_")] = value.strip()
        return version_info
    
    def read_manifest(self) -> Dict:
        """
        Read MANIFEST.json if it exists (from release archive).
        
        Returns:
            Dictionary of manifest data
        """
        if self.manifest_file.exists():
            with open(self.manifest_file, "r") as f:
                return json.load(f)
        return {}
    
    def get_info(self) -> Dict:
        """
        Get comprehensive version and deployment information.
        
        Returns:
            Dictionary containing version, git, and deployment info
        """
        # Try to read from release files first
        version_file_info = self.read_version_file()
        manifest = self.read_manifest()
        
        # Build comprehensive info
        info = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        
        # Add release archive info if available
        if version_file_info:
            info.update(version_file_info)
            info["source"] = "release_archive"
        
        if manifest:
            info.update(manifest)
        
        # Add git info (if not in release archive, or as supplement)
        git_info = {
            "git_commit": self.get_git_commit(),
            "git_branch": self.get_git_branch(),
            "git_tag": self.get_git_tag(),
        }
        
        # Only include git info if git info is missing from archive
        if "git_commit" not in info:
            info.update(git_info)
        elif git_info["git_commit"] and "git_commit" in info:
            # Archive info takes precedence, but note if different
            if git_info["git_commit"] != info.get("git_commit"):
                info["git_commit_current"] = git_info["git_commit"]
                info["git_commit_archived"] = info["git_commit"]
        
        return info
    
    def format_info(self, info: Optional[Dict] = None) -> str:
        """
        Format version info as human-readable string.
        
        Args:
            info: Version info dict (calls get_info() if not provided)
        
        Returns:
            Formatted version string
        """
        if info is None:
            info = self.get_info()
        
        lines = ["REMADE Pipeline - Version Information"]
        lines.append("=" * 50)
        
        for key, value in info.items():
            if value is not None:
                # Format key for display
                display_key = key.replace("_", " ").title()
                lines.append(f"{display_key}: {value}")
        
        return "\n".join(lines)
    
    def log_version_info(self, logger=None, level="info"):
        """
        Log version information using provided logger.
        
        Args:
            logger: Python logger object (or None to print to stdout)
            level: Log level ('info', 'debug', 'warning')
        """
        info = self.get_info()
        formatted = self.format_info(info)
        
        if logger:
            log_fn = getattr(logger, level, logger.info)
            log_fn(formatted)
        else:
            print(formatted)
    
    def create_run_manifest(self, output_dir: Path, run_config: Optional[Dict] = None) -> Path:
        """
        Create a manifest file for a pipeline run.
        
        Args:
            output_dir: Directory for output files
            run_config: Optional dictionary with run configuration
        
        Returns:
            Path to created manifest file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        # Add version info
        manifest.update(self.get_info())
        
        # Add run configuration
        if run_config:
            manifest["run_config"] = run_config
        
        # Write manifest
        manifest_file = output_dir / "RUN_MANIFEST.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_file


if __name__ == "__main__":
    """Simple CLI for version information."""
    
    version_info = VersionInfo()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        # Output as JSON
        import json
        print(json.dumps(version_info.get_info(), indent=2))
    else:
        # Output formatted text
        print(version_info.format_info())
