#!/usr/bin/env python3
"""Reproducible OOI ERDDAP acquisition for SAFE-SILENCE.

Downloads raw, selected-variable CSV chunks and writes a provenance sidecar
with SHA-256 for every chunk. The script does not clean, impute, resample, or
filter QARTOD flags.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

UTC = UTC


def parse_utc(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def filename_time(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {"server", "sites", "stream_templates", "acquisition_defaults"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Manifest missing keys: {sorted(missing)}")
    return manifest


def request_bytes(
    url: str,
    *,
    timeout: int,
    max_attempts: int,
    backoff: float,
    user_agent: str,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            delay = backoff * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    raise RuntimeError(f"Request failed after {max_attempts} attempts: {url}") from last_error


def available_variables(
    server: str,
    dataset_id: str,
    *,
    timeout: int,
    max_attempts: int,
    backoff: float,
    user_agent: str,
) -> set[str]:
    url = f"{server.rstrip('/')}/info/{dataset_id}/index.csv"
    payload = request_bytes(
        url,
        timeout=timeout,
        max_attempts=max_attempts,
        backoff=backoff,
        user_agent=user_agent,
    )
    text = payload.decode("utf-8-sig")
    rows = csv.DictReader(text.splitlines())
    return {row["Variable Name"] for row in rows if row.get("Row Type") == "variable"}


def iter_chunks(start: datetime, end: datetime, days: int) -> Iterable[tuple[datetime, datetime]]:
    if days <= 0:
        raise ValueError("chunk_days must be positive")
    cursor = start
    delta = timedelta(days=days)
    while cursor < end:
        stop = min(cursor + delta, end)
        yield cursor, stop
        cursor = stop


def build_url(
    server: str,
    dataset_id: str,
    variables: list[str],
    start: datetime,
    end: datetime,
) -> str:
    # Half-open interval prevents duplicate boundary timestamps across chunks.
    query = (
        ",".join(variables)
        + f"&time>={iso_utc(start)}"
        + f"&time<{iso_utc(end)}"
        + '&orderBy("time")'
    )
    encoded = quote(query, safe=',&><=:-()"')
    return f"{server.rstrip('/')}/tabledap/{dataset_id}.csv?{encoded}"


def validate_csv(path: Path, expected_variables: list[str]) -> tuple[int, str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        units = next(reader, None)
        if header is None:
            raise ValueError(f"Empty ERDDAP response: {path}")
        if header != expected_variables:
            raise ValueError(
                f"Unexpected header in {path}: expected {expected_variables}, got {header}"
            )
        if units is None:
            raise ValueError(f"Missing units row: {path}")
        count = 0
        first_time = ""
        last_time = ""
        for row in reader:
            if not row:
                continue
            count += 1
            if not first_time:
                first_time = row[0]
            last_time = row[0]
    return count, first_time, last_time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def download_chunk(
    *,
    url: str,
    output_path: Path,
    expected_variables: list[str],
    provenance: dict[str, Any],
    timeout: int,
    max_attempts: int,
    backoff: float,
    user_agent: str,
    force: bool,
) -> dict[str, Any]:
    sidecar = output_path.with_suffix(".provenance.json")
    if output_path.exists() and sidecar.exists() and not force:
        with sidecar.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if prior.get("sha256") == sha256_file(output_path):
            prior["status"] = "verified_existing"
            return prior

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part = output_path.with_suffix(output_path.suffix + ".part")
    if part.exists():
        part.unlink()

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=timeout) as response, part.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            rows, first_time, last_time = validate_csv(part, expected_variables)
            os.replace(part, output_path)
            result = {
                **provenance,
                "source_url": url,
                "downloaded_utc": iso_utc(datetime.now(UTC)),
                "file_name": output_path.name,
                "file_size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "data_rows": rows,
                "first_time": first_time,
                "last_time": last_time,
                "status": "downloaded",
            }
            write_json_atomic(sidecar, result)
            return result
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            error_body = ""
            if isinstance(exc, HTTPError):
                try:
                    error_body = exc.read().decode("utf-8", errors="replace").lower()
                except Exception:
                    error_body = ""
            if (
                isinstance(exc, HTTPError)
                and exc.code == 404
                and "no matching results" in error_body
            ):
                # ERDDAP returns 404 when a valid interval has no matching rows.
                result = {
                    **provenance,
                    "source_url": url,
                    "downloaded_utc": iso_utc(datetime.now(UTC)),
                    "file_name": "",
                    "file_size_bytes": 0,
                    "sha256": "",
                    "data_rows": 0,
                    "first_time": "",
                    "last_time": "",
                    "status": "no_data",
                    "http_status": 404,
                }
                write_json_atomic(sidecar, result)
                return result
            if isinstance(exc, HTTPError) and exc.code == 404 and error_body:
                raise RuntimeError(
                    f"ERDDAP returned HTTP 404 for a nonempty/invalid request: {error_body[:500]}"
                ) from exc
            if attempt == max_attempts:
                break
            delay = backoff * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    raise RuntimeError(f"Failed download: {url}") from last_error


def compile_inventory(output_root: Path, records: list[dict[str, Any]]) -> Path:
    inventory = output_root / "acquisition_inventory.csv"
    fields = [
        "site",
        "stream",
        "dataset_id",
        "request_start",
        "request_end",
        "status",
        "file_name",
        "file_size_bytes",
        "sha256",
        "data_rows",
        "first_time",
        "last_time",
        "downloaded_utc",
        "source_url",
    ]
    temp = inventory.with_suffix(".csv.part")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    os.replace(temp, inventory)
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "ooi_acquisition_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/ooi"))
    parser.add_argument(
        "--site",
        action="append",
        help="Site code; repeat as needed. Default: all sites in the manifest.",
    )
    parser.add_argument(
        "--stream",
        action="append",
        help="Stream name; repeat as needed. Default: all manifest streams.",
    )
    parser.add_argument("--start", help="UTC lower bound, e.g. 2024-07-01T00:00:00Z")
    parser.add_argument("--end", help="UTC exclusive upper bound")
    parser.add_argument("--chunk-days", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest.resolve())
    defaults = manifest["acquisition_defaults"]
    server = manifest["server"]
    timeout = int(defaults["timeout_seconds"])
    attempts = int(defaults["max_attempts"])
    backoff = float(defaults["backoff_seconds"])
    user_agent = str(defaults["user_agent"])
    chunk_days = args.chunk_days or int(defaults["chunk_days"])
    output_root = args.output_dir.resolve()

    selected_sites = args.site or list(manifest["sites"])
    unknown = set(selected_sites).difference(manifest["sites"])
    if unknown:
        raise ValueError(f"Unknown site code(s): {sorted(unknown)}")
    selected_streams = args.stream or list(manifest["stream_templates"])
    unknown_streams = set(selected_streams).difference(manifest["stream_templates"])
    if unknown_streams:
        raise ValueError(f"Unknown stream name(s): {sorted(unknown_streams)}")

    records: list[dict[str, Any]] = []
    for site_code in selected_sites:
        site = manifest["sites"][site_code]
        site_start = parse_utc(site["common_start"])
        site_end = parse_utc(site["common_end"])
        start = max(site_start, parse_utc(args.start) if args.start else site_start)
        end = min(site_end, parse_utc(args.end) if args.end else site_end)
        if start >= end:
            print(f"SKIP {site_code}: empty requested interval", file=sys.stderr)
            continue

        for stream_name, dataset_id in site["streams"].items():
            if stream_name not in selected_streams:
                continue
            variables = manifest["stream_templates"][stream_name]["variables"]
            actual = available_variables(
                server,
                dataset_id,
                timeout=timeout,
                max_attempts=attempts,
                backoff=backoff,
                user_agent=user_agent,
            )
            missing = set(variables).difference(actual)
            if missing:
                raise ValueError(f"{dataset_id} is missing manifest variables: {sorted(missing)}")
            print(f"VALID {site_code}/{stream_name}: {dataset_id}")
            if args.validate_only:
                continue

            for chunk_start, chunk_end in iter_chunks(start, end, chunk_days):
                url = build_url(server, dataset_id, variables, chunk_start, chunk_end)
                name = (
                    f"{site_code}__{stream_name}__"
                    f"{filename_time(chunk_start)}__{filename_time(chunk_end)}.csv"
                )
                output_path = output_root / site_code / stream_name / name
                provenance = {
                    "manifest_version": manifest["manifest_version"],
                    "site": site_code,
                    "stream": stream_name,
                    "dataset_id": dataset_id,
                    "request_start": iso_utc(chunk_start),
                    "request_end": iso_utc(chunk_end),
                    "variables": variables,
                }
                if args.dry_run:
                    print(f"DRY-RUN {output_path}\n  {url}")
                    continue
                result = download_chunk(
                    url=url,
                    output_path=output_path,
                    expected_variables=variables,
                    provenance=provenance,
                    timeout=timeout,
                    max_attempts=attempts,
                    backoff=backoff,
                    user_agent=user_agent,
                    force=args.force,
                )
                records.append(result)
                print(
                    f"{result['status'].upper()} {site_code}/{stream_name} "
                    f"{result['request_start']} rows={result['data_rows']}"
                )

    if not args.validate_only and not args.dry_run:
        inventory = compile_inventory(output_root, records)
        print(f"INVENTORY {inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
