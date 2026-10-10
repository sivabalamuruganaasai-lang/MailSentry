"""
Dataset Acquisition Script for Apache SpamAssassin Public Corpus.

Downloads raw tar.bz2 archives from official SpamAssassin mirrors
and extracts them into data/raw/spamassassin/.
"""

import os
import tarfile
import urllib.request
import sys
from pathlib import Path

# Official SpamAssassin archive URLs
ARCHIVE_URLS = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
]

# Alternate URL fallbacks if needed
FALLBACK_URLS = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2",
]

RAW_DIR = Path("data/raw/spamassassin")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def setup_directories():
    """Create data directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Directories initialized: {RAW_DIR}, {INTERIM_DIR}, {PROCESSED_DIR}")


def download_file(url: str, target_path: Path):
    """Download a file with user-agent and progress reporting."""
    if target_path.exists():
        print(f"[=] File already downloaded: {target_path.name} ({target_path.stat().st_size} bytes)")
        return True

    print(f"[>] Downloading {url} -> {target_path}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SpamClassifierBot/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[+] Successfully downloaded {target_path.name} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"[!] Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extract .tar.bz2 archive safely."""
    print(f"[>] Extracting {archive_path.name} into {extract_dir}...")
    try:
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=extract_dir)
        print(f"[+] Extraction complete for {archive_path.name}")
        return True
    except Exception as e:
        print(f"[!] Failed to extract {archive_path.name}: {e}")
        return False


def acquire_corpus():
    """Main acquisition function."""
    setup_directories()

    downloaded_files = []
    for url in ARCHIVE_URLS:
        filename = url.split("/")[-1]
        target_path = RAW_DIR / filename
        success = download_file(url, target_path)
        if not success:
            # Try fallback URL if applicable
            for fallback in FALLBACK_URLS:
                if fallback.endswith(filename.replace("20050311", "20030228")):
                    print(f"[*] Trying fallback URL: {fallback}")
                    download_file(fallback, target_path)
        if target_path.exists():
            downloaded_files.append(target_path)

    # Extract all archives
    extracted_dirs = set()
    for archive in downloaded_files:
        extract_archive(archive, RAW_DIR)

    print("\n[+] Dataset acquisition summary:")
    print(f"    Raw archives downloaded: {len(downloaded_files)}")
    subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    print(f"    Extracted directories ({len(subdirs)}): {[d.name for d in subdirs]}")


if __name__ == "__main__":
    acquire_corpus()
