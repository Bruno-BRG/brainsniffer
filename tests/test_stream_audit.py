import numpy as np
import pytest

from brainsniffer.pipeline.stream_audit import StreamAudit


def test_stream_audit_reports_finite_monotonic_stream():
    audit = StreamAudit()
    audit.push([0.0, 1.0, 2.0], source_rate=100, timestamps=[10.0, 10.01, 10.02])
    audit.push([3.0, 4.0], source_rate=100, timestamps=[10.03, 10.04])

    report = audit.report()

    assert report.ok
    assert report.sample_count == 5
    assert report.chunk_count == 2
    assert report.timestamps_present is True
    assert report.timestamp_nonmonotonic_count == 0
    assert report.timestamp_gap_count == 0
    assert report.finite_fraction == 1.0


def test_stream_audit_flags_nonfinite_flatline_and_timestamp_gap():
    audit = StreamAudit(max_gap_factor=1.5)
    audit.push(
        np.asarray([0.0, 0.0, np.nan, 100.0], dtype=np.float32),
        source_rate=100,
        timestamps=[0.0, 0.01, 0.02, 0.2],
    )

    report = audit.report()

    assert not report.ok
    assert report.finite_fraction == 0.75
    assert report.saturation_fraction > 0
    assert report.timestamp_gap_count == 1
    assert "há amostras não finitas" in report.warnings
    assert "há lacunas de timestamp acima do limite" in report.warnings


def test_stream_audit_can_require_timestamps():
    audit = StreamAudit(require_timestamps=True)
    audit.push([0.0, 1.0], source_rate=100)

    report = audit.report()

    assert report.ok is False
    assert report.timestamps_required is True
    assert report.timestamps_present is False
    assert "timestamps obrigatórios ausentes" in report.warnings


def test_stream_audit_rejects_rate_change_and_timestamp_mode_change():
    audit = StreamAudit()
    audit.push([1.0], source_rate=128)
    with pytest.raises(ValueError, match="source_rate"):
        audit.push([1.0], source_rate=256)

    with pytest.raises(ValueError, match="timestamps"):
        audit.push([1.0], source_rate=128, timestamps=[1.0])


def test_stream_audit_flags_nonfinite_timestamps():
    audit = StreamAudit()
    audit.push([0.0, 1.0, 2.0], source_rate=100, timestamps=[10.0, np.nan, 10.02])

    report = audit.report()

    assert not report.ok
    assert report.timestamp_nonfinite_count == 1
    assert "há timestamps não finitos" in report.warnings


def test_stream_audit_records_complete_source_metadata():
    audit = StreamAudit(require_metadata=True)
    audit.set_metadata(
        {
            "unit": "uV",
            "channel_name": "Fpz",
            "reference": "linked ears",
            "montage": "frontal referenced",
            "source_name": "bench",
        }
    )
    audit.push([0.0, 1.0], source_rate=128, timestamps=[1.0, 1.01])

    report = audit.report()

    assert report.ok
    assert report.metadata_complete
    assert report.metadata_missing == ()
    assert report.metadata is not None
    assert report.metadata["channel_name"] == "Fpz"


def test_stream_audit_requires_metadata_without_accepting_a_partial_manifest():
    audit = StreamAudit(require_metadata=True)
    audit.set_metadata({"unit": "uV", "channel_name": "Fpz"})
    audit.push([0.0, 1.0], source_rate=128, timestamps=[1.0, 1.01])

    report = audit.report()

    assert not report.ok
    assert report.metadata_complete is False
    assert report.metadata_missing == ("reference", "montage")
    assert "metadata obrigatório incompleto" in report.warnings[-1]


def test_stream_audit_rejects_conflicting_metadata():
    audit = StreamAudit()
    audit.set_metadata({"unit": "uV"})
    with pytest.raises(ValueError, match="metadata"):
        audit.set_metadata({"unit": "mV"})


def test_stream_audit_rejects_metadata_rate_that_differs_from_chunks():
    audit = StreamAudit()
    audit.set_metadata({"sampling_rate": 128})

    with pytest.raises(ValueError, match="sampling_rate do metadata"):
        audit.push([0.0], source_rate=256)


def test_stream_audit_rejects_invalid_metadata_rate():
    audit = StreamAudit()

    with pytest.raises(ValueError, match="sampling_rate do metadata"):
        audit.set_metadata({"sampling_rate": "unknown"})
