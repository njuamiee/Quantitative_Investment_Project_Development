"""Read-only SFTP snapshot helpers for the cross-sectional research pipeline.

Credentials are parsed at runtime from the user-provided connection note.  They
are never written to manifests or logs.  The server host key is pinned so an
unexpected server identity aborts the transfer.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import paramiko


PINNED_HOST_KEY_SHA256 = "SHA256:aLRBngVfqlfyqAsSpPF8b7W0W0Bv1v3CiIvu1i+ptEA"
REMOTE_UNADJUSTED_DIR = "/home/quant/day-data/不复权"
REMOTE_DATA_DICTIONARY = "/home/quant/factors/数据字典.csv"


@dataclass(frozen=True)
class Credentials:
    host: str
    username: str
    password: str


def read_credentials(path: str | Path) -> Credentials:
    text = Path(path).read_text(encoding="utf-8-sig")

    def find(pattern: str) -> str:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Connection note is missing field matching {pattern!r}")
        return match.group(1).strip()

    return Credentials(
        host=find(r"IP\s*[:：]\s*([^\s]+)"),
        username=find(r"user\s*[:：]\s*([^\s]+)"),
        password=find(r"(?:pw|password|密码)\s*[:：]\s*([^\s]+)"),
    )


def connect(credentials_path: str | Path) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    cred = read_credentials(credentials_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=cred.host,
        port=22,
        username=cred.username,
        password=cred.password,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    key = client.get_transport().get_remote_server_key()
    fingerprint = "SHA256:" + base64.b64encode(
        hashlib.sha256(key.asbytes()).digest()
    ).decode("ascii").rstrip("=")
    if fingerprint != PINNED_HOST_KEY_SHA256:
        client.close()
        raise RuntimeError(
            f"Server host key changed: expected {PINNED_HOST_KEY_SHA256}, got {fingerprint}"
        )
    return client, client.open_sftp()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_bse_snapshot(
    credentials_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Download 920xxx unadjusted daily bars and the provider data dictionary."""
    output = Path(output_dir).resolve()
    price_dir = output / "unadjusted_daily"
    metadata_dir = output / "metadata"
    price_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    client, sftp = connect(credentials_path)
    manifest_rows: list[dict] = []
    try:
        attrs = sftp.listdir_attr(REMOTE_UNADJUSTED_DIR)
        selected = sorted(
            (
                attr
                for attr in attrs
                if not stat.S_ISDIR(attr.st_mode)
                and re.match(r"^920\d{3}.*\.csv$", attr.filename, flags=re.IGNORECASE)
            ),
            key=lambda item: item.filename,
        )
        if not selected:
            raise RuntimeError("No 920xxx daily price files found on the server")

        for number, attr in enumerate(selected, start=1):
            remote_path = f"{REMOTE_UNADJUSTED_DIR}/{attr.filename}"
            local_path = price_dir / attr.filename
            if not local_path.exists() or local_path.stat().st_size != attr.st_size:
                temp_path = local_path.with_suffix(local_path.suffix + ".part")
                sftp.get(remote_path, str(temp_path))
                temp_path.replace(local_path)
            manifest_rows.append(
                {
                    "remote_path": remote_path,
                    "local_relative_path": local_path.relative_to(output).as_posix(),
                    "size": attr.st_size,
                    "remote_mtime_utc": datetime.fromtimestamp(
                        attr.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "sha256": sha256_file(local_path),
                }
            )
            if number % 50 == 0 or number == len(selected):
                print(f"prices {number}/{len(selected)}", flush=True)

        dictionary_path = metadata_dir / "provider_data_dictionary.csv"
        dictionary_attr = sftp.stat(REMOTE_DATA_DICTIONARY)
        if (
            not dictionary_path.exists()
            or dictionary_path.stat().st_size != dictionary_attr.st_size
        ):
            temp_path = dictionary_path.with_suffix(".csv.part")
            sftp.get(REMOTE_DATA_DICTIONARY, str(temp_path))
            temp_path.replace(dictionary_path)
        manifest_rows.append(
            {
                "remote_path": REMOTE_DATA_DICTIONARY,
                "local_relative_path": dictionary_path.relative_to(output).as_posix(),
                "size": dictionary_attr.st_size,
                "remote_mtime_utc": datetime.fromtimestamp(
                    dictionary_attr.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": sha256_file(dictionary_path),
            }
        )
    finally:
        sftp.close()
        client.close()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_key_sha256": PINNED_HOST_KEY_SHA256,
        "price_file_count": len(manifest_rows) - 1,
        "files": manifest_rows,
    }
    manifest_path = output / "snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sync_bse_snapshot(args.credentials, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
