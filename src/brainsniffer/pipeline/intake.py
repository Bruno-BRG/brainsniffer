"""Equipment-manifest checks that precede a research bench stream."""

from __future__ import annotations

from collections.abc import Mapping

from .stream_audit import StreamAudit

REQUIRED_INTAKE_FIELDS = (
    "device_manufacturer",
    "device_model",
    "firmware",
    "bridge",
    "sampling_rate",
    "unit",
    "channel_name",
    "reference",
    "montage",
    "nominal_range",
    "processing_applied",
)

SUPPORTED_INPUT_UNITS = ("uv", "µv", "μv", "microvolt", "microvolts")


def validate_intake_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    """Validate the minimum equipment manifest without inspecting EEG samples.

    A successful result means that the manifest is ready for a technical bench
    check. It never means that the device, model, or workflow is clinically
    validated.
    """

    audit = StreamAudit()
    audit.set_metadata(metadata or {})
    normalized = dict(audit.report().metadata or {})
    missing: list[str] = []
    for field in REQUIRED_INTAKE_FIELDS:
        value = normalized.get(field)
        if field == "sampling_rate":
            if value is None:
                missing.append(field)
            continue
        if not isinstance(value, str) or not value.strip():
            missing.append(field)

    unit = str(normalized.get("unit", "")).strip().casefold().replace(" ", "")
    compatibility_issues: list[str] = []
    if "unit" not in missing and unit not in SUPPORTED_INPUT_UNITS:
        compatibility_issues.append(
            "unit deve ser microvolt (uV/µV); converta no bridge antes da inferência"
        )
    ready = not missing and not compatibility_issues
    return {
        "status": (
            "ready_for_bench"
            if ready
            else ("incompatible" if compatibility_issues else "incomplete")
        ),
        "ready_for_bench": ready,
        "required_fields": list(REQUIRED_INTAKE_FIELDS),
        "missing_fields": missing,
        "compatibility_issues": compatibility_issues,
        "metadata": normalized,
        "scope": "research_only",
        "clinical_decision_support": False,
        "controls_anesthetic_delivery": False,
    }
