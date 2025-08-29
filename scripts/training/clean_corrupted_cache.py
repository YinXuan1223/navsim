#!/usr/bin/env python3
"""
Script to detect and handle corrupted cache files in NavSim training cache.
"""

import argparse
import gzip
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gzip_file(file_path: Path) -> bool:
    """
    Test if a gzip file is valid.
    
    Args:
        file_path: Path to the gzip file
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        with gzip.open(file_path, "rb") as f:
            # Try to read the entire file
            pickle.load(f)
        return True
    except (gzip.BadGzipFile, OSError, EOFError, pickle.UnpicklingError, Exception):
        return False


def find_corrupted_files(cache_path: Path) -> List[Path]:
    """
    Find all corrupted .gz files in the cache directory.
    
    Args:
        cache_path: Path to cache directory
        
    Returns:
        List of corrupted file paths
    """
    corrupted_files = []
    
    if not cache_path.exists():
        logger.error(f"Cache path {cache_path} does not exist!")
        return corrupted_files
    
    logger.info(f"Scanning cache directory: {cache_path}")
    
    # Find all .gz files
    gz_files = list(cache_path.rglob("*.gz"))
    logger.info(f"Found {len(gz_files)} .gz files to check")
    
    for gz_file in gz_files:
        if not test_gzip_file(gz_file):
            corrupted_files.append(gz_file)
            logger.warning(f"CORRUPTED: {gz_file}")
    
    return corrupted_files


def clean_corrupted_tokens(cache_path: Path, corrupted_files: List[Path]) -> Tuple[int, int]:
    """
    Remove entire token directories that contain corrupted files.
    
    Args:
        cache_path: Path to cache directory
        corrupted_files: List of corrupted file paths
        
    Returns:
        Tuple of (tokens_removed, files_removed)
    """
    tokens_to_remove = set()
    
    # Group corrupted files by token directory
    for corrupted_file in corrupted_files:
        # Find the token directory (parent of the corrupted file)
        token_dir = corrupted_file.parent
        tokens_to_remove.add(token_dir)
    
    tokens_removed = 0
    files_removed = 0
    
    for token_dir in tokens_to_remove:
        if token_dir.exists():
            logger.info(f"Removing corrupted token directory: {token_dir}")
            # Count files before removal
            files_in_token = len(list(token_dir.glob("*")))
            files_removed += files_in_token
            
            # Remove the entire token directory
            import shutil
            shutil.rmtree(token_dir)
            tokens_removed += 1
    
    return tokens_removed, files_removed


def main():
    parser = argparse.ArgumentParser(description="Clean corrupted cache files from NavSim training cache")
    parser.add_argument("cache_path", type=str, help="Path to the cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Only report corrupted files, don't remove them")
    parser.add_argument("--remove-tokens", action="store_true", help="Remove entire token directories containing corrupted files")
    
    args = parser.parse_args()
    
    cache_path = Path(args.cache_path)
    
    # Find corrupted files
    corrupted_files = find_corrupted_files(cache_path)
    
    if not corrupted_files:
        logger.info("No corrupted files found! 🎉")
        return 0
    
    logger.error(f"Found {len(corrupted_files)} corrupted files:")
    for f in corrupted_files:
        logger.error(f"  - {f}")
    
    if args.dry_run:
        logger.info("Dry run mode - no files will be removed")
        return 0
    
    if args.remove_tokens:
        tokens_removed, files_removed = clean_corrupted_tokens(cache_path, corrupted_files)
        logger.info(f"Removed {tokens_removed} token directories containing {files_removed} files")
    else:
        # Just remove the corrupted files
        for corrupted_file in corrupted_files:
            logger.info(f"Removing corrupted file: {corrupted_file}")
            corrupted_file.unlink()
        logger.info(f"Removed {len(corrupted_files)} corrupted files")
    
    logger.info("Cleanup completed! ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
