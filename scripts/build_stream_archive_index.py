#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a SQLite index for split tar.gz archives stored in GCS.

Archives are streamed directly from GCS — no local download or disk space
is required.  Only tar member headers are inspected; file contents are
discarded during the streaming read.
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
from collections import defaultdict
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
                     desc='Listing GCS blobs', unit=' blobs'):
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
    """Concatenate multiple file-like objects into one readable stream."""

    def __init__(self, streams: List[io.IOBase]):
        self._streams = streams
        self._idx = 0

    def read(self, size: int = -1) -> bytes:
        chunks: List[bytes] = []
        remaining = size
        while self._idx < len(self._streams):
            chunk = self._streams[self._idx].read(remaining if remaining > 0 else -1)
            if not chunk:
                self._streams[self._idx].close()
                self._idx += 1
                continue
            chunks.append(chunk)
            if remaining > 0:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
        return b''.join(chunks)

    def close(self) -> None:
        for stream in self._streams[self._idx:]:
            stream.close()
        self._idx = len(self._streams)

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


def _stream_archive_members(
    *,
    gcs_prefix: str,
    archive_stem: str,
    parts: List[str],
    storage_client: Optional[Any] = None,
) -> List[str]:
    """Stream a (possibly split) tar.gz from GCS and return member file paths.

    Only tar headers are inspected — file contents are discarded on the fly,
    so no local disk space is consumed.
    """
    client = _get_storage_client(storage_client)
    bucket_name, blob_prefix = _parse_gs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)

    total_size = _get_total_blob_size(bucket, blob_prefix, parts)

    blob_streams: List[io.IOBase] = []
    for part in parts:
        object_name = _join_gcs_object_path(blob_prefix, part)
        blob = bucket.blob(object_name)
        blob_streams.append(blob.open('rb'))

    members: List[str] = []
    with _ConcatenatedStream(blob_streams) as raw_stream:
        progress = tqdm(
            total=total_size,
            desc=f'  Streaming {archive_stem}',
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
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
        CREATE TABLE archives (
            archive_id TEXT PRIMARY KEY,
            gcs_prefix TEXT NOT NULL,
            archive_stem TEXT NOT NULL,
            parts_json TEXT NOT NULL
        );
        CREATE TABLE files (
            logical_path TEXT PRIMARY KEY,
            archive_id TEXT NOT NULL,
            member_path TEXT NOT NULL,
            FOREIGN KEY (archive_id) REFERENCES archives(archive_id)
        );
        CREATE INDEX idx_files_archive_id ON files(archive_id);
    """)


# ---------------------------------------------------------------------------
# Main index builder
# ---------------------------------------------------------------------------

def build_archive_index(
    *,
    gcs_prefix: str,
    output_path: str,
    scratch_dir: Optional[str] = None,
    storage_client: Optional[Any] = None,
) -> str:
    """Build a SQLite index by streaming archives from GCS.

    ``scratch_dir`` is accepted for backward compatibility but ignored —
    archives are no longer downloaded to local disk.
    """
    del scratch_dir  # no longer used

    archive_parts = _list_archive_parts(gcs_prefix, storage_client=storage_client)
    if not archive_parts:
        raise ValueError(f'No archive parts found under {gcs_prefix}')

    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_output = f'{output_path}.tmp'
    if os.path.exists(temp_output):
        os.remove(temp_output)

    with sqlite3.connect(temp_output) as conn:
        _create_schema(conn)

        sorted_stems = sorted(archive_parts)
        for archive_stem in tqdm(sorted_stems, desc='Processing archives',
                                 unit=' archives'):
            parts = archive_parts[archive_stem]
            archive_id = archive_stem

            member_paths = _stream_archive_members(
                gcs_prefix=gcs_prefix,
                archive_stem=archive_stem,
                parts=parts,
                storage_client=storage_client,
            )

            conn.execute(
                'INSERT INTO archives (archive_id, gcs_prefix, archive_stem, parts_json) VALUES (?, ?, ?, ?)',
                (archive_id, gcs_prefix, archive_stem, json.dumps(parts, ensure_ascii=False)))

            rows = [(mp, archive_id, mp) for mp in member_paths]
            try:
                conn.executemany(
                    'INSERT INTO files (logical_path, archive_id, member_path) VALUES (?, ?, ?)',
                    rows)
            except sqlite3.IntegrityError as e:
                raise ValueError(
                    f'Duplicate logical_path detected while indexing `{archive_stem}`: {e}') from e

            conn.commit()
            logger.info(f'Indexed {len(rows)} files from `{archive_stem}`')

    os.replace(temp_output, output_path)
    logger.info(f'Saved archive index to {output_path}')
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a SQLite index for split tar.gz archives in GCS')
    parser.add_argument('--gcs-prefix', required=True, help='GCS prefix that contains archive shard parts')
    parser.add_argument('--output', required=True, help='Output SQLite path')
    parser.add_argument('--scratch-dir', help='(ignored) Kept for backward compatibility')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_archive_index(
        gcs_prefix=args.gcs_prefix,
        output_path=args.output,
        scratch_dir=args.scratch_dir,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
