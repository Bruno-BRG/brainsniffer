"""Synthetic checkpoint checks; no training, historical artifacts or shell payloads."""

import ast
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from brainsniffer.config import PreprocessConfig
from brainsniffer.models.cnn import Conv1DDepthEstimator
from brainsniffer.pipeline import training


class UnapprovedMetadata:
    """Inert stand-in for an unapproved pickle global (no reduce or shell code)."""

    restored = False

    def __init__(self):
        self.value = "benign"

    def __setstate__(self, state):
        type(self).restored = True
        self.__dict__.update(state)


@pytest.fixture
def payload():
    return {
        "schema_version": training.CHECKPOINT_SCHEMA_VERSION,
        "model_name": "Conv1DDepthEstimator",
        "model_state": Conv1DDepthEstimator().state_dict(),
        "preprocess_config": asdict(PreprocessConfig()),
        "environment": training.runtime_metadata(),
    }


def save(tmp_path, payload, sidecar=None):
    path = tmp_path / "synthetic.pt"
    torch.save(payload, path)
    if sidecar is not None:
        path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


@pytest.mark.parametrize("legacy", [False, True])
def test_restricted_baseline_roundtrip(tmp_path, payload, legacy):
    if legacy:
        payload.pop("schema_version")
    path = save(tmp_path, payload)
    model, preprocess, loaded = training.load_checkpoint(path)
    assert not model.training
    assert preprocess == PreprocessConfig()
    assert loaded.keys() == payload.keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, payload["model_state"][key])
    assert type(loaded["environment"]["torch"]) is str
    assert ("schema_version" in loaded) is not legacy


def test_unapproved_global_rejected_without_execution(tmp_path, payload):
    UnapprovedMetadata.restored = False
    payload["extra"] = UnapprovedMetadata()
    path = save(tmp_path, payload)
    with pytest.raises(ValueError, match="loader restrito") as error:
        training.load_checkpoint(path)
    assert isinstance(error.value.__cause__, pickle.UnpicklingError)
    assert not UnapprovedMetadata.restored


def test_legacy_torch_version_metadata_restricted_roundtrip(tmp_path, payload):
    from torch.torch_version import TorchVersion

    payload.pop("schema_version")
    payload["environment"]["torch"] = TorchVersion("2.14.0+cu130")
    path = save(tmp_path, payload)
    digest = training.sha256_file(path)
    before = torch.serialization.get_safe_globals()
    with pytest.raises(pickle.UnpicklingError):
        torch.load(path, map_location="cpu", weights_only=True)
    model, _, loaded = training.load_checkpoint(path)
    assert not model.training
    assert str(loaded["environment"]["torch"]) == "2.14.0+cu130"
    assert "schema_version" not in loaded
    assert training.sha256_file(path) == digest
    assert torch.serialization.get_safe_globals() == before


def test_reviewed_torch_version_has_only_string_reconstruction():
    from torch.torch_version import TorchVersion

    assert TorchVersion.__bases__ == (str,)
    assert TorchVersion.__slots__ == ()
    assert TorchVersion.__new__ is str.__new__
    assert TorchVersion.__init__ is str.__init__
    assert not hasattr(TorchVersion, "__setstate__")


def test_preexisting_torch_version_registration_is_preserved(tmp_path, payload):
    from torch.torch_version import TorchVersion

    payload["environment"]["torch"] = TorchVersion("2.14.0+cu130")
    path = save(tmp_path, payload)
    with torch.serialization.safe_globals([TorchVersion]):
        before = torch.serialization.get_safe_globals()
        training.load_checkpoint(path)
        assert torch.serialization.get_safe_globals() == before


def test_legacy_metadata_does_not_allow_other_globals(tmp_path, payload):
    from torch.torch_version import TorchVersion

    UnapprovedMetadata.restored = False
    payload["environment"]["torch"] = TorchVersion("2.14.0+cu130")
    payload["extra"] = UnapprovedMetadata()
    path = save(tmp_path, payload)
    before = torch.serialization.get_safe_globals()
    with pytest.raises(ValueError, match="loader restrito"):
        training.load_checkpoint(path)
    assert not UnapprovedMetadata.restored
    assert torch.serialization.get_safe_globals() == before


def test_no_unsafe_retry(tmp_path, monkeypatch):
    calls = []

    def reject(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("weights_only unsupported by old runtime")

    monkeypatch.setattr(training.torch, "load", reject)
    with pytest.raises(ValueError, match="Sem fallback inseguro"):
        training.load_checkpoint(tmp_path / "not-read.pt")
    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("key,value,match", [
    ("schema_version", 2, "schema_version"),
    ("schema_version", True, "schema_version"),
    ("model_name", "RobustConv1DDepthEstimator", "model_name"),
    ("model_name", None, "model_name"),
    ("model_state", {}, "model_state"),
    ("model_state", {"x": "not a tensor"}, "model_state"),
    ("preprocess_config", [], "preprocess_config"),
    ("preprocess_config", {"unknown": 1}, "preprocess_config"),
    ("preprocess_config", {"sampling_rate": "128"}, "preprocess_config"),
    ("preprocess_config", {"window_seconds": -1}, "preprocess_config"),
    ("preprocess_config", {"lowcut_hz": float("nan")}, "preprocess_config"),
])
def test_invalid_schema(tmp_path, payload, key, value, match):
    payload[key] = value
    with pytest.raises(ValueError, match=match):
        training.load_checkpoint(save(tmp_path, payload))


@pytest.mark.parametrize("key", ["model_name", "model_state", "preprocess_config"])
def test_missing_required_fields(tmp_path, payload, key):
    del payload[key]
    with pytest.raises(ValueError, match=key):
        training.load_checkpoint(save(tmp_path, payload))


def test_non_dictionary_payload(tmp_path):
    with pytest.raises(ValueError, match="dicionário"):
        training.load_checkpoint(save(tmp_path, [1, 2]))


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nan", "missing", "extra"])
def test_incompatible_state(tmp_path, payload, mutation):
    state = payload["model_state"]
    key = "features.0.weight"
    if mutation == "shape":
        state[key] = torch.zeros(1)
    elif mutation == "dtype":
        state[key] = state[key].double()
    elif mutation == "nan":
        state[key].fill_(float("nan"))
    elif mutation == "missing":
        del state[key]
    else:
        state["unexpected"] = torch.zeros(1)
    with pytest.raises(ValueError, match="model_state incompatível"):
        training.load_checkpoint(save(tmp_path, payload))


@pytest.mark.parametrize("sidecar,match", [
    ([], "objeto JSON"),
    ({"checkpoint_sha256": "not a digest"}, "SHA-256"),
    ({"checkpoint_sha256": "0" * 64}, "SHA-256"),
    ({"model_name": "OtherModel"}, "divergem"),
    ({"schema_version": 2}, "divergem"),
    ({"preprocess_config": {}}, "divergem"),
])
def test_invalid_sidecar(tmp_path, payload, sidecar, match):
    with pytest.raises(ValueError, match=match):
        training.load_checkpoint(save(tmp_path, payload, sidecar))


def test_valid_checksum(tmp_path, payload):
    path = save(tmp_path, payload)
    digest = training.sha256_file(path)
    path.with_suffix(".json").write_text(json.dumps({"checkpoint_sha256": digest}))
    assert training.load_checkpoint(path)[2]["checkpoint_sha256"] == digest


def test_future_writer_metadata_static_without_training():
    """Inspect writer wiring only; does not claim an end-to-end training check."""
    source = Path(training.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    writer = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "train_model")
    assignment = next(node for node in ast.walk(writer) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "checkpoint_payload"
                              for target in node.targets))
    fields = {key.value: value for key, value in zip(assignment.value.keys,
                                                   assignment.value.values)}
    expected = {"schema_version", "best_epoch", "training_config", "effective_training"}
    assert expected <= fields.keys()
    controls = {key.value for key in fields["effective_training"].keys}
    assert {"scheduler", "gradient_clip_norm", "mixed_precision", "amp_dtype",
            "deterministic_algorithms", "deterministic_warn_only"} <= controls
    # The JSON sidecar is derived from the same payload, not a second recipe.
    assert "**checkpoint_payload" in ast.unparse(writer)
    assert "if key != 'model_state'" in ast.unparse(writer)
