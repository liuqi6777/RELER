"""Behavioral protocol checks with real local tokenizers and a tiny Qwen backbone."""

import json
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3Model,
    TrainingArguments,
)

from reler.config import ModelArguments
from reler.data.embedding import EmbeddingDataCollator, document_key
from reler.data.protocol import (
    load_embedding_protocol,
    pool_embeddings,
    tokenization_metadata,
    tokenize_embedding_texts,
)
from reler.evaluation.qwen3_embedding_model import TransformersTextEmbedder
from reler.objectives.contrastive import cross_query_scores
from reler.training.supervised import BaselineTrainer


def tokenizer(kind="embedding", padding_side="left"):
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "word": 2, "other": 3, "[CLS]": 4, "[SEP]": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    if kind == "embedding":
        backend.post_processor = TemplateProcessing(single="$A [PAD]", special_tokens=[("[PAD]", 1)])
    elif kind == "encoder":
        backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 4), ("[SEP]", 5)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                  cls_token="[CLS]", sep_token="[SEP]", eos_token="[SEP]",
                                  padding_side=padding_side, model_input_names=["input_ids", "attention_mask"])


def test_strong_cl_metadata_preserves_collator_false_negative_filters():
    collator = EmbeddingDataCollator(tokenizer=tokenizer(), include_cross_batch_metadata=True)
    records = [dict(query='word', document=['word', 'other'], ranking=[1, 2], pos_index=1, source='s',
                    document_ids=['a', 'b'], original_relevant_docids=['c']),
               dict(query='other', document=['word word', 'word other', 'other word'],
                    ranking=[1, 2, 3], pos_index=1, source='s', document_ids=['c', 'd', 'e'],
                    known_positive_keys=[document_key('other')])]
    batch = collator(records)
    assert batch['candidate_mask'].tolist() == [[True, True, False], [True, True, True]]
    _, valid, _ = cross_query_scores(torch.randn(2, 5), torch.randn(2, 3, 5), batch['cross_batch_metadata'],
                                    include_negatives=True, cross_device=False, detach_documents=False)
    assert torch.equal(valid.reshape(2, 2, 3), batch['in_batch_candidate_mask'])
    collator.include_cross_batch_metadata = False
    assert 'cross_batch_metadata' not in collator(records)


@pytest.mark.parametrize("kind", ["base", "embedding"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("max_length", [1, 2, 4])
def test_one_attended_terminal_survives_truncation(kind, padding_side, max_length):
    tok = tokenizer(kind, padding_side)
    batch = tokenize_embedding_texts(["word", "word other word other word", ""], tok, "pad", max_length=max_length)
    expected = [[2][:max_length-1]+[1], [2, 3, 2, 3, 2][:max_length-1]+[1], [1]]
    for ids, mask, wanted in zip(batch["input_ids"], batch["attention_mask"], expected):
        assert ids[mask.bool()].tolist() == wanted
        assert (ids[~mask.bool()] == tok.pad_token_id).all()
    hidden = torch.arange(batch["input_ids"].numel()*3).reshape(*batch["input_ids"].shape, 3).float()
    pooled = pool_embeddings(hidden, batch["attention_mask"], pooling_method="last", normalize=False)
    last_positions = (batch["attention_mask"] * torch.arange(1, hidden.shape[1]+1)).argmax(-1)
    torch.testing.assert_close(pooled, hidden[torch.arange(3), last_positions])


def test_explicit_terminal_is_id_based_and_normalizes_existing_boundary():
    tok = tokenizer()
    batch = tokenize_embedding_texts(["word [PAD] [PAD]", "word [PAD] other"], tok, "pad", max_length=8)
    assert batch["input_ids"][0][batch["attention_mask"][0].bool()].tolist() == [2, 1]
    assert batch["input_ids"][1].tolist() == [2, 1, 3, 1]
    eos = tokenize_embedding_texts(["word"], tok, "eos", max_length=8)
    assert eos["input_ids"].tolist() == [[2, 5]]


def test_none_preserves_native_encoder_special_tokens():
    tok = tokenizer("encoder", "right")
    texts = ["word", "word other word other word"]
    actual = tokenize_embedding_texts(texts, tok, "none", max_length=4)
    native = tok(texts, padding=True, truncation=True, max_length=4, return_tensors="pt")
    for key in native:
        torch.testing.assert_close(actual[key], native[key])
    assert actual["input_ids"].tolist() == [[4, 2, 5, 1], [4, 2, 3, 5]]


def tiny_model():
    return Qwen3Model(Qwen3Config(vocab_size=6, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=2, head_dim=8, max_position_embeddings=64)).eval()


def test_training_mteb_and_native_embedding_agree():
    tok, model = tokenizer(), tiny_model()
    texts = ["word", "word other word other word"]
    records = [dict(query=text, document=["word", "word other word other word"],
                    ranking=[1, 2], pos_index=1) for text in texts]
    training = EmbeddingDataCollator(tok, query_max_length=4, doc_max_length=4)(records)
    with patch("reler.evaluation.qwen3_embedding_model.AutoModel.from_pretrained", return_value=model), \
         patch("reler.evaluation.qwen3_embedding_model.AutoTokenizer.from_pretrained", return_value=tok):
        evaluator = TransformersTextEmbedder("local-tiny", do_norm=True)
    evaluation = evaluator.tokenize(texts, max_length=4)
    native = tok(texts, padding=True, truncation=True, max_length=4, return_tensors="pt")
    for key in native:
        torch.testing.assert_close(training["query"][key], native[key])
        torch.testing.assert_close(evaluation[key], native[key])
    documents = torch.cat([training["positive_document"]["input_ids"], training["negative_document"]["input_ids"]])
    assert documents.tolist() == [[1, 1, 2, 1], [1, 1, 2, 1], [2, 3, 2, 1], [2, 3, 2, 1]]
    with torch.inference_mode():
        expected = pool_embeddings(model(**native).last_hidden_state, native["attention_mask"], pooling_method="last")
        torch.testing.assert_close(evaluator(**evaluation), expected)


def test_checkpoint_save_roundtrip_without_eval_callback(tmp_path):
    tok = tokenizer()
    wrapper = torch.nn.Module()
    wrapper.model = tiny_model()
    trainer = BaselineTrainer(
        model=wrapper, model_args=ModelArguments(model_name_or_path="local-tiny"),
        processing_class=tok, args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
    )
    checkpoint = tmp_path / "checkpoint-1"
    trainer.save_model(str(checkpoint))
    saved = load_embedding_protocol(checkpoint)
    restored = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    for key, expected in tokenization_metadata(restored, "pad").items():
        assert saved[key] == expected
    assert saved["pooling_compute_dtype"] == "float32"
    assert saved["add_special_tokens"] is False
    assert saved["terminal_token_id"] == 1
    assert saved["terminal_after_truncation"] is True
    batch = tokenize_embedding_texts(["word other word other"], restored, saved["append_token"], max_length=3)
    assert batch["input_ids"].tolist() == [[2, 3, 1]]
    no_precision = {key: value for key, value in saved.items() if key != "pooling_compute_dtype"}
    (checkpoint / "embedding_protocol.json").write_text(json.dumps(no_precision))
    with pytest.raises(ValueError, match="pooling precision"):
        load_embedding_protocol(checkpoint)
    saved.pop("tokenization_version")
    (checkpoint / "embedding_protocol.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="tokenization protocol"):
        load_embedding_protocol(checkpoint)
