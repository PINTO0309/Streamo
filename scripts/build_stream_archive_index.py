#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a SQLite index for split tar.gz archives stored in GCS."""

import argparse
import json
import os
import posixpath
import re
import shutil
import sqlite3
import tarfile
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import get_logger

logger = get_logger()

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


def _list_archive_parts(gcs_prefix: str, storage_client: Optional[Any] = None) -> Dict[str, List[str]]:
    client = _get_storage_client(storage_client)
    bucket_name, blob_prefix = _parse_gs_uri(gcs_prefix)

    groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for blob in client.list_blobs(bucket_name, prefix=blob_prefix):
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


def _download_and_merge_archive(
    *,
    gcs_prefix: str,
    archive_stem: str,
    parts: List[str],
    output_path: str,
    scratch_dir: str,
    storage_client: Optional[Any] = None,
) -> str:
    client = _get_storage_client(storage_client)
    bucket_name, blob_prefix = _parse_gs_uri(gcs_prefix)
    bucket = client.bucket(bucket_name)

    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_part_paths = []
    try:
        for relative_part in parts:
            object_name = _join_gcs_object_path(blob_prefix, relative_part)
            local_part_path = os.path.join(scratch_dir, os.path.basename(relative_part))
            bucket.blob(object_name).download_to_filename(local_part_path)
            temp_part_paths.append(local_part_path)

        tmp_output_path = f'{output_path}.tmp'
        try:
            with open(tmp_output_path, 'wb') as out_file:
                for part_path in temp_part_paths:
                    with open(part_path, 'rb') as in_file:
                        shutil.copyfileobj(in_file, out_file, length=4 * 1024 * 1024)
            os.replace(tmp_output_path, output_path)
        finally:
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
    finally:
        for part_path in temp_part_paths:
            if os.path.exists(part_path):
                os.remove(part_path)

    logger.info(f'Prepared archive `{archive_stem}` at {output_path}')
    return output_path


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


def build_archive_index(
    *,
    gcs_prefix: str,
    output_path: str,
    scratch_dir: Optional[str] = None,
    storage_client: Optional[Any] = None,
) -> str:
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

        managed_temp_dir = None
        if scratch_dir is None:
            managed_temp_dir = tempfile.TemporaryDirectory(prefix='stream_archive_index_')
            scratch_dir = managed_temp_dir.name
        else:
            scratch_dir = os.path.abspath(os.path.expanduser(scratch_dir))
            os.makedirs(scratch_dir, exist_ok=True)

        try:
            for archive_stem in sorted(archive_parts):
                parts = archive_parts[archive_stem]
                archive_id = archive_stem
                archive_path = os.path.join(scratch_dir, archive_stem.replace('/', os.sep))
                _download_and_merge_archive(
                    gcs_prefix=gcs_prefix,
                    archive_stem=archive_stem,
                    parts=parts,
                    output_path=archive_path,
                    scratch_dir=os.path.join(scratch_dir, 'parts', archive_stem.replace('/', '_')),
                    storage_client=storage_client,
                )

                conn.execute(
                    'INSERT INTO archives (archive_id, gcs_prefix, archive_stem, parts_json) VALUES (?, ?, ?, ?)',
                    (archive_id, gcs_prefix, archive_stem, json.dumps(parts, ensure_ascii=False)))

                rows = []
                with tarfile.open(archive_path, 'r:gz') as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        member_path = _normalize_relative_posix_path(member.name)
                        rows.append((member_path, archive_id, member_path))

                try:
                    conn.executemany(
                        'INSERT INTO files (logical_path, archive_id, member_path) VALUES (?, ?, ?)',
                        rows)
                except sqlite3.IntegrityError as e:
                    raise ValueError(
                        f'Duplicate logical_path detected while indexing `{archive_stem}`: {e}') from e

                conn.commit()
                logger.info(f'Indexed {len(rows)} files from `{archive_stem}`')
        finally:
            if managed_temp_dir is not None:
                managed_temp_dir.cleanup()

    os.replace(temp_output, output_path)
    logger.info(f'Saved archive index to {output_path}')
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a SQLite index for split tar.gz archives in GCS')
    parser.add_argument('--gcs-prefix', required=True, help='GCS prefix that contains archive shard parts')
    parser.add_argument('--output', required=True, help='Output SQLite path')
    parser.add_argument('--scratch-dir', help='Optional local scratch directory for temporary archives')
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
