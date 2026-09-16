import hashlib

import pytest

import brainsniffer.data.figshare as figshare


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size: int = -1):
        if not self.content:
            return b""
        chunk = self.content[:size]
        self.content = self.content[size:]
        return chunk


def _manifest(content: bytes):
    return {
        "files": [
            {
                "name": "case1.mat",
                "size": len(content),
                "computed_md5": hashlib.md5(content).hexdigest(),
                "download_url": "https://example.invalid/case1.mat",
            }
        ]
    }


def test_parse_data_files_keeps_official_checksum():
    item = figshare.parse_data_files(_manifest(b"abc"))[0]
    assert item.md5 == hashlib.md5(b"abc").hexdigest()


def test_download_dataset_verifies_checksum(monkeypatch, tmp_path):
    content = b"synthetic-mat-content"
    monkeypatch.setattr(figshare, "fetch_manifest", lambda: _manifest(content))
    monkeypatch.setattr(figshare, "urlopen", lambda request, timeout: _Response(content))

    paths = figshare.download_dataset(tmp_path)

    assert paths == [tmp_path / "case1.mat"]
    assert paths[0].read_bytes() == content


def test_download_dataset_rejects_inconsistent_existing_file(monkeypatch, tmp_path):
    content = b"expected"
    monkeypatch.setattr(figshare, "fetch_manifest", lambda: _manifest(content))
    target = tmp_path / "case1.mat"
    target.write_bytes(b"corrupted")

    with pytest.raises(OSError, match="inconsistente"):
        figshare.download_dataset(tmp_path)
