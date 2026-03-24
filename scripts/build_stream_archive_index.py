#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a SQLite index for split tar.gz archives stored in GCS.

When ``--cache-dir`` is given, archive parts are downloaded to local disk
first and reused across runs — the same part is never downloaded twice.
Without ``--cache-dir``, archives are streamed directly from GCS (no local
disk needed, but every run re-downloads).
"""

import argparse
import io
import json
import logging
import os
import posixpath
import re
import sqlite3
import sys
import tarfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from swift.utils import get_logger
    logger = get_logger()
except ModuleNotFoundError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PART_SUFFIX_RE = re.compile(r'^(?P<stem>.+\.tar\.gz)\.(?P<part>\d+)$')


def _normalize_relative_posix_path(path: str) -> str:
    normalized = posixpath.normpath(str(path).replace('\\', '/'))
    if normalized in {'', '.'}:
        raise ValueError('Path must not be empty')
    normalized = normalized.lstrip('/')
    if normalized in {'', '.'} or normalized == '..' or normalized.startswith('../'):
        raise ValueError(f'Path escapes archive root: {path}')
    return normalized


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    match = re.match(r'^gs://([^/]+)(?:/(.*))?$', gs_uri)
    if not match:
        raise ValueError(f'Invalid GCS URI: {gs_uri}')
    bucket = match.group(1)
    prefix = (match.group(2) or '').strip('/')
    return bucket, prefix


def _join_gcs_object_path(prefix: str, relative_path: str) -> str:
    relative_path = _normalize_relative_posix_path(relative_path)
    if not prefix:
        return relative_path
    return f'{prefix.rstrip("/")}/{relative_path}'


def _get_storage_client(storage_client: Optional[Any] = None):
    if storage_client is not None:
        return storage_client
    from google.cloud import storage
    return storage.Client()


# ---------------------------------------------------------------------------
# GCS blob listing
# ---------------------------------------------------------------------------

def _list_archive_parts(gcs_prefix: str, storage_client: Optional[Any] = None) -> Dict[str, List[str]]:
    client = _get_storage_client(storage_client)
    bucket_name, blob_prefix = _parse_gs_uri(gcs_prefix)

    groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for blob in tqdm(client.list_blobs(bucket_name, prefix=blob_prefix),
                     desc='Listing GCS blobs', unit=' blobs', dynamic_ncols=True):
        relative_name = blob.name
        if blob_prefix:
            relative_name = relative_name[len(blob_prefix):].lstrip('/')
        if not relative_name or relative_name.endswith('/'):
            continue

        match = PART_SUFFIX_RE.match(relative_name)
        if match:
            groups[_normalize_relative_posix_path(match.group('stem'))].append(
                (int(match.group('part')), _normalize_relative_posix_path(relative_name)))
        elif relative_name.endswith('.tar.gz'):
            normalized = _normalize_relative_posix_path(relative_name)
            groups[normalized].append((0, normalized))

    return {
        archive_stem: [path for _, path in sorted(parts, key=lambda item: item[0])]
        for archive_stem, parts in groups.items()
    }


# ---------------------------------------------------------------------------
# Streaming tar member enumeration (no local download)
# ---------------------------------------------------------------------------

class _ConcatenatedStream:
    """Concatenate multiple GCS blobs into one readable stream.

    Blobs are opened lazily — only the currently-read part has an active
    connection.  This avoids idle-connection timeouts and reduces the number
    of concurrent HTTP sessions.
    """

    def __init__(self, blob_openers: List):
        """``blob_openers`` is a list of zero-arg callables, each returning a
        readable file-like object (e.g. ``lambda: blob.open('rb')``)."""
        self._openers = blob_openers
        self._idx = 0
        self._current: Optional[io.IOBase] = None

    def _ensure_stream(self) -> bool:
        """Open the next stream if needed.  Return False when exhausted."""
        if self._current is not None:
            return True
        if self._idx >= len(self._openers):
            return False
        self._current = self._openers[self._idx]()
        return True

    def read(self, size: int = -1) -> bytes:
        chunks: List[bytes] = []
        remaining = size
        while self._ensure_stream():
            chunk = self._current.read(remaining if remaining > 0 else -1)
            if not chunk:
                self._current.close()
                self._current = None
                self._idx += 1
                continue
            chunks.append(chunk)
            if remaining > 0:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
        return b''.join(chunks)

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
        self._idx = len(self._openers)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _ProgressStream:
    """Wrap a readable stream with a tqdm byte-level progress bar."""

    def __init__(self, stream: _ConcatenatedStream, progress: tqdm):
        self._stream = stream
        self._progress = progress

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self._progress.update(len(data))
        return data

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _get_total_blob_size(
    bucket,
    blob_prefix: str,
    parts: List[str],
) -> Optional[int]:
    """Return combined byte size of all parts, or None if unavailable."""
    total = 0
    for part in parts:
        object_name = _join_gcs_object_path(blob_prefix, part)
        blob = bucket.blob(object_name)
        blob.reload()
        if blob.size is None:
            return None
        total += blob.size
    return total


# ---------------------------------------------------------------------------
# Local-cache download & read
# ---------------------------------------------------------------------------

def _download_archive_parts(
    *,
    gcs_prefix: str,
    archive_stem: str,
    parts: List[str],
    cache_dir: str,
    storage_client: Optional[Any] = None,
    tqdm_position: Optional[int] = None,
) -> List[str]:
    """Download archive parts to *cache_dir*.  Already-cached parts (whose
    local size matches the remote blob size) are skipped."""
    client = _get_storage_client(storage_client)
    bucket_name, blob_prefix = _parse_gs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)

    local_paths: List[str] = []
    for part in parts:
        object_name = _join_gcs_object_path(blob_prefix, part)
        blob = bucket.blob(object_name)
        blob.reload()

        local_path = os.path.join(cache_dir, part)
        local_paths.append(local_path)

        # Skip if already cached with correct size
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            if blob.size is not None and local_size == blob.size:
                logger.debug(f'Cache hit: {part} ({local_size:,} bytes)')
                continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        partial_path = f'{local_path}.partial'

        progress = tqdm(
            total=blob.size,
            desc=f'  DL {os.path.basename(part)}',
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            dynamic_ncols=True,
            position=tqdm_position,
        )
        with progress:
            with blob.open('rb') as src, open(partial_path, 'wb') as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)  # 8 MiB
                    if not chunk:
                        break
                    dst.write(chunk)
                    progress.update(len(chunk))

        os.replace(partial_path, local_path)
        logger.debug(f'Downloaded {part} → {local_path}')

    return local_paths


def _read_local_archive_members(
    *,
    local_paths: List[str],
    archive_stem: str,
    tqdm_position: Optional[int] = None,
) -> List[str]:
    """Read tar member file paths from locally-cached archive parts."""
    total_size = sum(os.path.getsize(p) for p in local_paths)

    file_openers = [lambda p=p: open(p, 'rb') for p in local_paths]

    members: List[str] = []
    with _ConcatenatedStream(file_openers) as raw_stream:
        progress = tqdm(
            total=total_size,
            desc=f'  Reading {archive_stem}',
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            dynamic_ncols=True,
            position=tqdm_position,
        )
        with progress, _ProgressStream(raw_stream, progress) as stream:
            with tarfile.open(fileobj=stream, mode='r|gz') as tar:
                for member in tar:
                    if member.isfile():
                        members.append(_normalize_relative_posix_path(member.name))
                        progress.set_postfix(files=len(members), refresh=False)
    return members


# ---------------------------------------------------------------------------
# GCS-only streaming (fallback when no cache_dir)
# ---------------------------------------------------------------------------

def _stream_archive_members(
    *,
    gcs_prefix: str,
    archive_stem: str,
    parts: List[str],
    storage_client: Optional[Any] = None,
    tqdm_position: Optional[int] = None,
) -> List[str]:
    """Stream a (possibly split) tar.gz from GCS and return member file paths.

    Only tar headers are inspected — file contents are discarded on the fly,
    so no local disk space is consumed.
    """
    client = _get_storage_client(storage_client)
    bucket_name, blob_prefix = _parse_gs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)

    total_size = _get_total_blob_size(bucket, blob_prefix, parts)

    blob_openers = []
    for part in parts:
        object_name = _join_gcs_object_path(blob_prefix, part)
        blob = bucket.blob(object_name)
        blob_openers.append(lambda b=blob: b.open('rb'))

    members: List[str] = []
    with _ConcatenatedStream(blob_openers) as raw_stream:
        progress = tqdm(
            total=total_size,
            desc=f'  Streaming {archive_stem}',
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            dynamic_ncols=True,
            position=tqdm_position,
        )
        with progress, _ProgressStream(raw_stream, progress) as stream:
            # 'r|gz' is the sequential/streaming mode — no seeking required.
            with tarfile.open(fileobj=stream, mode='r|gz') as tar:
                for member in tar:
                    if member.isfile():
                        members.append(_normalize_relative_posix_path(member.name))
                        progress.set_postfix(files=len(members), refresh=False)
    return members


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS archives (
            archive_id TEXT PRIMARY KEY,
            gcs_prefix TEXT NOT NULL,
            archive_stem TEXT NOT NULL,
            parts_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            logical_path TEXT PRIMARY KEY,
            archive_id TEXT NOT NULL,
            member_path TEXT NOT NULL,
            FOREIGN KEY (archive_id) REFERENCES archives(archive_id)
        );
        CREATE INDEX IF NOT EXISTS idx_files_archive_id ON files(archive_id);
    """)


# ---------------------------------------------------------------------------
# Main index builder
# ---------------------------------------------------------------------------

def build_archive_index(
    *,
    gcs_prefix: str,
    output_path: str,
    cache_dir: Optional[str] = None,
    storage_client: Optional[Any] = None,
    num_workers: int = 1,
) -> str:
    """Build a SQLite index for archives in GCS.

    When *cache_dir* is given, archive parts are downloaded once and reused
    on subsequent runs.  Without it, archives are streamed directly from GCS
    each time.

    ``num_workers`` controls how many archives are processed in parallel.
    """
    if cache_dir is not None:
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f'Using local cache: {cache_dir}')

    archive_parts = _list_archive_parts(gcs_prefix, storage_client=storage_client)
    if not archive_parts:
        raise ValueError(f'No archive parts found under {gcs_prefix}')

    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_workers = max(1, num_workers)

    # -- detect already-indexed archives for resume -------------------------
    done_stems: set = set()
    if os.path.exists(output_path):
        with sqlite3.connect(output_path) as conn:
            _create_schema(conn)  # ensure tables exist (handles older DBs)
            rows = conn.execute('SELECT archive_stem FROM archives').fetchall()
            done_stems = {r[0] for r in rows}
        if done_stems:
            logger.info(
                f'Resuming: {len(done_stems)} archive(s) already indexed, '
                f'skipping: {sorted(done_stems)}')

    pending_stems = sorted(s for s in archive_parts if s not in done_stems)
    if not pending_stems:
        logger.info('All archives already indexed — nothing to do.')
        return output_path

    # -- tqdm position pool for concurrent per-archive progress bars --------
    _position_lock = threading.Lock()
    _free_positions: List[int] = list(range(1, num_workers + 1))

    def _acquire_position() -> int:
        with _position_lock:
            return _free_positions.pop(0)

    def _release_position(pos: int) -> None:
        with _position_lock:
            _free_positions.append(pos)

    def _worker(archive_stem: str) -> Tuple[str, List[str], List[str]]:
        parts = archive_parts[archive_stem]
        pos = _acquire_position()
        try:
            if cache_dir is not None:
                local_paths = _download_archive_parts(
                    gcs_prefix=gcs_prefix,
                    archive_stem=archive_stem,
                    parts=parts,
                    cache_dir=cache_dir,
                    storage_client=storage_client,
                    tqdm_position=pos,
                )
                members = _read_local_archive_members(
                    local_paths=local_paths,
                    archive_stem=archive_stem,
                    tqdm_position=pos,
                )
            else:
                members = _stream_archive_members(
                    gcs_prefix=gcs_prefix,
                    archive_stem=archive_stem,
                    parts=parts,
                    storage_client=storage_client,
                    tqdm_position=pos,
                )
        finally:
            _release_position(pos)
        return archive_stem, parts, members

    # -- run workers and collect results into SQLite ------------------------
    with sqlite3.connect(output_path) as conn:
        _create_schema(conn)

        overall = tqdm(
            total=len(pending_stems),
            initial=0,
            desc='Processing archives',
            unit=' archives',
            dynamic_ncols=True,
            position=0,
        )
        if done_stems:
            overall.set_postfix(skipped=len(done_stems), refresh=True)
        with overall:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_worker, stem): stem
                    for stem in pending_stems
                }
                for future in as_completed(futures):
                    archive_stem, parts, member_paths = future.result()
                    archive_id = archive_stem

                    conn.execute(
                        'INSERT INTO archives (archive_id, gcs_prefix, archive_stem, parts_json) '
                        'VALUES (?, ?, ?, ?)',
                        (archive_id, gcs_prefix, archive_stem,
                         json.dumps(parts, ensure_ascii=False)))

                    rows = [(mp, archive_id, mp) for mp in member_paths]
                    try:
                        conn.executemany(
                            'INSERT INTO files (logical_path, archive_id, member_path) '
                            'VALUES (?, ?, ?)',
                            rows)
                    except sqlite3.IntegrityError as e:
                        raise ValueError(
                            f'Duplicate logical_path detected while indexing '
                            f'`{archive_stem}`: {e}') from e

                    conn.commit()
                    overall.update(1)
                    overall.set_postfix(last=archive_stem, refresh=False)
                    logger.info(f'Indexed {len(rows)} files from `{archive_stem}`')

    logger.info(f'Saved archive index to {output_path}')
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a SQLite index for split tar.gz archives in GCS')
    parser.add_argument('--gcs-prefix', required=True, help='GCS prefix that contains archive shard parts')
    parser.add_argument('--output', required=True, help='Output SQLite path')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of archives to process in parallel (default: 1)')
    parser.add_argument('--cache-dir',
                        help='Local directory to cache downloaded archive parts. '
                             'Cached parts are reused across runs so the same data '
                             'is never downloaded twice.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_archive_index(
        gcs_prefix=args.gcs_prefix,
        output_path=args.output,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
