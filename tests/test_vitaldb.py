import io

import numpy as np

from brainsniffer.data.vitaldb import (
    VitalTrack,
    VitalTrackData,
    _align_numeric_to_eeg,
    download_vitaldb_case,
    read_track,
)


def test_read_vitaldb_waveform_encoding(monkeypatch):
    content = b"Time,BIS/EEG1_WAV\n0,1\n0.0078125,2\n,3\n,4\n"

    import gzip

    compressed = gzip.compress(content)
    stream = io.BytesIO(compressed)

    def fake_open(url):
        return _open_gzip_text_from_stream(stream)

    monkeypatch.setattr("brainsniffer.data.vitaldb._open_gzip_text", fake_open)
    track = read_track(VitalTrack(1, "BIS/EEG1_WAV", "eeg"), waveform=True)

    assert track.sampling_rate == 128.0
    assert np.allclose(track.values, [1, 2, 3, 4])
    assert np.allclose(track.times, np.arange(4) / 128)


def _open_gzip_text_from_stream(stream):
    import gzip

    compressed = gzip.GzipFile(fileobj=stream)
    return io.TextIOWrapper(compressed, encoding="utf-8")


def test_align_numeric_track_uses_relative_one_second_grid():
    eeg = VitalTrackData(
        name="BIS/EEG1_WAV",
        times=np.arange(257, dtype=np.float64) / 128,
        values=np.zeros(257, dtype=np.float32),
        sampling_rate=128.0,
    )
    bis = VitalTrackData(
        name="BIS/BIS",
        times=np.asarray([0.2, 1.2, 2.2]),
        values=np.asarray([40.0, 50.0, 60.0], dtype=np.float32),
        sampling_rate=None,
    )
    aligned = _align_numeric_to_eeg(eeg, bis)
    assert aligned.shape == (3,)
    assert np.isnan(aligned[0])
    assert np.allclose(aligned[1:], [48.0, 58.0])


def test_download_vitaldb_case_records_source_metadata(monkeypatch, tmp_path):
    tracks = {
        "BIS/EEG1_WAV": VitalTrack(1, "BIS/EEG1_WAV", "eeg-track"),
        "BIS/BIS": VitalTrack(1, "BIS/BIS", "bis-track"),
    }
    monkeypatch.setattr(
        "brainsniffer.data.vitaldb.list_case_tracks",
        lambda *args, **kwargs: tracks,
    )

    def fake_read_track(track, *, waveform):
        if waveform:
            return VitalTrackData(
                name=track.name,
                times=np.arange(256, dtype=np.float64) / 128,
                values=np.zeros(256, dtype=np.float32),
                sampling_rate=128.0,
            )
        return VitalTrackData(
            name=track.name,
            times=np.asarray([0.0, 1.0]),
            values=np.asarray([45.0, 55.0], dtype=np.float32),
            sampling_rate=None,
        )

    monkeypatch.setattr("brainsniffer.data.vitaldb.read_track", fake_read_track)
    path = download_vitaldb_case(1, tmp_path, subject_id="subject-42")

    with np.load(path, allow_pickle=False) as data:
        assert data["source_dataset"].item() == "VitalDB Open Dataset"
        assert data["eeg_unit"].item() == "uV"
        assert data["eeg_track_name"].item() == "BIS/EEG1_WAV"
        assert data["bis_track_name"].item() == "BIS/BIS"
        assert data["eeg_track_id"].item() == "eeg-track"
        assert data["bis_track_id"].item() == "bis-track"
        assert data["subject_id"].item() == "subject-42"
        assert data["group_id"].item() == "vitaldb:subject:subject-42"
