"""Tests for the embedding cache (torch-free) and extraction (torch-guarded)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.config import ArtifactPaths
from src.noise.embeddings import (
    EXTRACTORS,
    METADATA_COLUMNS,
    SENTENCE_MODEL_ID,
    EmbeddingConfig,
    build_metadata_frame,
    default_embeddings_dir,
    load_embeddings,
    save_embeddings,
)


def test_embedding_config_rejects_bad_dtype() -> None:
    with pytest.raises(ValueError, match="output_dtype"):
        EmbeddingConfig(output_dtype="float64")


def test_build_metadata_frame_columns_and_row_index() -> None:
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "condition": ["depression", "bipolar", "depression"],
            "source": ["low_et_al"] * 3,
            "author_id": ["u1", "u2", "u3"],
        }
    )
    meta = build_metadata_frame(df)
    assert list(meta.columns) == list(METADATA_COLUMNS)
    assert list(meta["row_index"]) == [0, 1, 2]
    assert list(meta["condition"]) == ["depression", "bipolar", "depression"]
    assert list(meta["extractor"]) == ["finetuned"] * 3  # default


def test_build_metadata_frame_stamps_custom_extractor() -> None:
    df = pd.DataFrame({"text": ["a", "b"], "condition": ["bipolar", "depression"]})
    meta = build_metadata_frame(df, extractor="base")
    assert list(meta["extractor"]) == ["base", "base"]


def test_default_embeddings_dir_distinct_per_extractor(tmp_path) -> None:
    artifacts = ArtifactPaths(root=tmp_path)
    finetuned = default_embeddings_dir(artifacts, "train", "finetuned")
    base = default_embeddings_dir(artifacts, "train", "base")
    # Fine-tuned keeps the original path; base is a distinct sibling so it cannot
    # overwrite the D-034 cache.
    assert finetuned == artifacts.root / "embeddings" / "train"
    assert base != finetuned
    assert base == artifacts.root / "embeddings" / "train__base"


def test_build_metadata_frame_stamps_mpnet_extractor() -> None:
    # D-040 Arm B: the metadata must carry the new extractor value, or a later
    # comparison could not tell Arm B's features from the other two arms'.
    df = pd.DataFrame({"text": ["a", "b"], "condition": ["bipolar", "schizophrenia"]})
    meta = build_metadata_frame(df, extractor="mpnet")
    assert list(meta["extractor"]) == ["mpnet", "mpnet"]
    assert list(meta.columns) == list(METADATA_COLUMNS)


def test_default_embeddings_dir_mpnet_cannot_collide(tmp_path) -> None:
    # D-040: Arm B must not clobber the D-034 fine-tuned cache or the D-036 base one.
    artifacts = ArtifactPaths(root=tmp_path)
    dirs = {e: default_embeddings_dir(artifacts, "train", e) for e in EXTRACTORS}
    assert dirs["mpnet"] == artifacts.root / "embeddings" / "train__mpnet"
    # All three extractors resolve to three distinct directories.
    assert len(set(dirs.values())) == len(EXTRACTORS)
    # And no arm's directory is nested inside another's, which a suffix scheme could
    # otherwise allow to happen silently.
    for a, path_a in dirs.items():
        for b, path_b in dirs.items():
            if a != b:
                assert path_b not in path_a.parents


def test_extractors_registry_matches_expected_arms() -> None:
    # Guards against an arm being added without a cache path or a metadata value.
    assert EXTRACTORS == ("finetuned", "base", "mpnet")
    assert SENTENCE_MODEL_ID == "sentence-transformers/all-mpnet-base-v2"


def test_save_load_round_trip(tmp_path) -> None:
    features = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "condition": ["depression", "bipolar", "schizophrenia"],
            "source": ["low_et_al"] * 3,
            "author_id": ["u1", "u2", "u3"],
        }
    )
    meta = build_metadata_frame(df)
    save_embeddings(tmp_path, features, meta)

    loaded_features, loaded_meta = load_embeddings(tmp_path)
    np.testing.assert_array_equal(loaded_features, features)
    assert list(loaded_meta["condition"]) == ["depression", "bipolar", "schizophrenia"]


def test_load_missing_cache_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_embeddings(tmp_path)


def test_save_length_mismatch_raises(tmp_path) -> None:
    features = np.zeros((3, 4), dtype=np.float32)
    meta = build_metadata_frame(pd.DataFrame({"text": ["a", "b"]}))
    with pytest.raises(ValueError, match="length mismatch"):
        save_embeddings(tmp_path, features, meta)


# --- torch-dependent extraction path (skips where torch is absent) ---------------

torch = pytest.importorskip("torch")

from src.noise.embeddings import extract_pooled_embeddings  # noqa: E402


class _FakeTokenizer:
    def __call__(self, text, truncation=True, max_length=8, padding="max_length"):
        return {"input_ids": [1] * 4, "attention_mask": [1] * 4}


class _PooledOutput:
    def __init__(self, tensor: "torch.Tensor") -> None:
        self.pooler_output = tensor


class _FakeEncoder(torch.nn.Module):
    """Stand-in for BertModel: returns a fixed pooler_output sized to `hidden`."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self._hidden = hidden

    def forward(self, **inputs):
        batch = next(iter(inputs.values())).shape[0]
        vals = torch.arange(batch * self._hidden, dtype=torch.float32).reshape(batch, self._hidden)
        return _PooledOutput(vals)


class _FakeModel(torch.nn.Module):
    """Stand-in for AutoModelForSequenceClassification exposing `.base_model`."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self._encoder = _FakeEncoder(hidden)

    @property
    def base_model(self):  # matches transformers' PreTrainedModel.base_model
        return self._encoder


def test_extract_pooled_embeddings_shape_and_dtype(monkeypatch) -> None:
    import src.modeling.hf_model as hf

    hidden = 6
    monkeypatch.setattr(hf, "load_model", lambda cfg, name_or_path=None: _FakeModel(hidden))
    monkeypatch.setattr(hf, "load_tokenizer", lambda name_or_path: _FakeTokenizer())

    df = pd.DataFrame({"text": ["a", "b", "c", "d", "e"], "condition": ["depression"] * 5})
    feats = extract_pooled_embeddings(
        df, "unused-checkpoint", embed_config=EmbeddingConfig(batch_size=2, device="cpu")
    )
    assert feats.shape == (5, hidden)
    assert feats.dtype == np.float32


def test_extract_pooled_embeddings_float16_cache(monkeypatch) -> None:
    import src.modeling.hf_model as hf

    monkeypatch.setattr(hf, "load_model", lambda cfg, name_or_path=None: _FakeModel(4))
    monkeypatch.setattr(hf, "load_tokenizer", lambda name_or_path: _FakeTokenizer())

    df = pd.DataFrame({"text": ["a", "b"], "condition": ["bipolar", "depression"]})
    feats = extract_pooled_embeddings(
        df, "unused", embed_config=EmbeddingConfig(batch_size=2, output_dtype="float16", device="cpu")
    )
    assert feats.dtype == np.float16


# --- D-040 Arm B: masked mean pooling and the sentence-encoder path ---------------

from src.noise.embeddings import (  # noqa: E402
    extract_mean_pooled_embeddings,
    masked_mean_pool,
)


def test_masked_mean_pool_ignores_padding_positions() -> None:
    # Hand-built, known answer. Row 0's third position is padding and carries a huge
    # value, so an unmasked mean would be nowhere near the expected [2, 3]: this is
    # the failure the mask exists to prevent.
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],  # mask [1, 1, 0] -> mean of first two
            [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],      # mask [1, 1, 1] -> mean of all three
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = masked_mean_pool(hidden, mask)

    assert pooled.shape == (2, 2)
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0], [3.0, 3.0]]))
    # And confirm the unmasked mean really would have differed, so the test cannot
    # pass with the masking silently removed.
    assert not torch.allclose(pooled[0], hidden[0].mean(dim=0))


def test_masked_mean_pool_all_padding_row_is_zero_not_nan() -> None:
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    pooled = masked_mean_pool(hidden, torch.tensor([[0, 0]]))
    assert torch.isfinite(pooled).all()
    torch.testing.assert_close(pooled, torch.zeros(1, 2))


def test_masked_mean_pool_rejects_mismatched_mask() -> None:
    hidden = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="attention_mask"):
        masked_mean_pool(hidden, torch.ones(2, 5, dtype=torch.long))


class _HiddenStateOutput:
    def __init__(self, tensor: "torch.Tensor") -> None:
        self.last_hidden_state = tensor


class _FakeSentenceEncoder(torch.nn.Module):
    """Stand-in for an AutoModel encoder: returns a fixed last_hidden_state."""

    def __init__(self, hidden: int, seq_len: int = 4) -> None:
        super().__init__()
        self._hidden = hidden
        self._seq_len = seq_len

    def forward(self, **inputs):
        batch = next(iter(inputs.values())).shape[0]
        n = batch * self._seq_len * self._hidden
        vals = torch.arange(1, n + 1, dtype=torch.float32).reshape(
            batch, self._seq_len, self._hidden
        )
        return _HiddenStateOutput(vals)


def _patch_sentence_encoder(monkeypatch, hidden: int) -> None:
    """Point the lazy ``from transformers import AutoModel`` at a fake, so no model
    is downloaded (the test suite must stay offline)."""
    import transformers

    import src.modeling.hf_model as hf

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name_or_path, *args, **kwargs):
            return _FakeSentenceEncoder(hidden)

    monkeypatch.setattr(transformers, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(hf, "load_tokenizer", lambda name_or_path: _FakeTokenizer())


def test_extract_mean_pooled_embeddings_shape_dtype_and_normalised(monkeypatch) -> None:
    hidden = 6
    _patch_sentence_encoder(monkeypatch, hidden)

    df = pd.DataFrame({"text": ["a", "b", "c"], "condition": ["depression"] * 3})
    feats = extract_mean_pooled_embeddings(
        df, "unused-hub-id", EmbeddingConfig(batch_size=2, device="cpu")
    )

    assert feats.shape == (3, hidden)
    assert feats.dtype == np.float32
    # L2-normalised, per the module DECISION (the published encoder's third module).
    np.testing.assert_allclose(np.linalg.norm(feats, axis=1), np.ones(3), rtol=1e-5)


def test_extract_mean_pooled_embeddings_honours_output_dtype(monkeypatch) -> None:
    _patch_sentence_encoder(monkeypatch, 4)

    df = pd.DataFrame({"text": ["a", "b"], "condition": ["bipolar", "depression"]})
    feats = extract_mean_pooled_embeddings(
        df,
        "unused-hub-id",
        EmbeddingConfig(batch_size=2, output_dtype="float16", device="cpu"),
    )
    assert feats.dtype == np.float16


def test_extract_mean_pooled_embeddings_rejects_missing_hidden_state(monkeypatch) -> None:
    # A non-BERT-like encoder that exposes only pooler_output must fail loudly rather
    # than silently substituting a different tensor (mirrors the pooler_output guard).
    import transformers

    import src.modeling.hf_model as hf

    class _PoolerOnlyEncoder(torch.nn.Module):
        def forward(self, **inputs):
            return _PooledOutput(torch.zeros(next(iter(inputs.values())).shape[0], 3))

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name_or_path, *args, **kwargs):
            return _PoolerOnlyEncoder()

    monkeypatch.setattr(transformers, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(hf, "load_tokenizer", lambda name_or_path: _FakeTokenizer())

    df = pd.DataFrame({"text": ["a"], "condition": ["bipolar"]})
    with pytest.raises(RuntimeError, match="last_hidden_state"):
        extract_mean_pooled_embeddings(
            df, "unused-hub-id", EmbeddingConfig(batch_size=1, device="cpu")
        )
