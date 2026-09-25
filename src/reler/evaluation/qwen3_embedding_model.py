from __future__ import annotations

import json
import logging
import queue
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import mteb
import numpy as np
import torch
from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from torch.utils.data._utils.worker import ManagerWatchdog
from tqdm.autonotebook import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from reler.data.protocol import (
    TOKENIZATION_VERSION,
    format_embedding_text,
    pool_embeddings,
    tokenize_embedding_texts,
)
from reler.evaluation.task_prompts import TASK_PROMPTS

logger = logging.getLogger(__name__)


class TransformersTextEmbedder(torch.nn.Module):
    def __init__(
        self,
        model: str,
        pooler_type: str = "last",
        do_norm: bool = False,
        truncate_dim: int = 0,
        padding_left: bool = False,
        padding_side: str = "left",
        append_token: str = "pad",
        attn_type: str = "causal",
        **kwargs,
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.tokenizer.padding_side = padding_side
        self.pooler_type = pooler_type
        self.do_norm = do_norm
        self.truncate_dim = truncate_dim
        self.padding_left = padding_left
        self.append_token = append_token
        self.attn_type = attn_type
        if pooler_type in {"first", "cls"}:
            assert padding_left is False
        elif pooler_type not in {"last", "mean"}:
            raise ValueError(f"Wrong pooler: {self.pooler_type}")

    def embed(
        self,
        sentences: Sequence[str],
        max_length: int,
        prompt: str | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        inputs = self.tokenize(sentences, max_length, prompt).to(device)
        embeddings = self.forward(**inputs.data)
        return embeddings

    def tokenize(self, texts, max_length: int, prompt=None) -> BatchEncoding:
        if prompt:
            texts = [prompt + t for t in texts]
        return tokenize_embedding_texts(
            texts, self.tokenizer, self.append_token, max_length=max_length
        )

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        canonical_pooler = "cls" if self.pooler_type == "first" else self.pooler_type
        embeddings = pool_embeddings(
            output.last_hidden_state,
            attention_mask,
            pooling_method=canonical_pooler,
            normalize=self.do_norm,
        )
        if self.truncate_dim > 0:
            embeddings = embeddings[:, : self.truncate_dim]
        if self.do_norm and self.truncate_dim > 0:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def _encode_loop(
    model: TransformersTextEmbedder,
    input_queue,
    output_queue,
    device: torch.device,
    qsize: int = 4,
    amp_dtype=None,
):
    model = model.to(device)
    watchdog = ManagerWatchdog()
    keep_queue = queue.Queue(qsize + 1)

    with torch.inference_mode():
        with (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        ):
            while watchdog.is_alive():
                r = input_queue.get()
                if r is None:
                    break

                n, inputs = r
                embeddings = model.embed(*inputs, device=device)
                output_queue.put((n, embeddings))
                if keep_queue.full():
                    i = keep_queue.get()
                    del i
                keep_queue.put(embeddings)
                del r, n, inputs

    while not keep_queue.empty():
        i = keep_queue.get()
        del i
    del model, watchdog
    return


class Qwen3Embedding(Wrapper):
    _model_class = TransformersTextEmbedder
    # `_model_class` needs to implement `embed(batch, max_length, prompt_name, self.device)`.

    def __init__(
        self,
        model: str,
        use_instruction: bool = False,
        device: str = "cuda",
        max_length: int = 512,
        max_query_length: int | None = None,
        max_doc_length: int | None = None,
        precision: str = "fp32",
        mp_qsize: int = 4,
        instruction_dict_path=None,
        instruction_template=None,
        query_prompt_template: str | None = None,
        document_prompt_template: str = "{text}",
        **kwargs,  # For `TransformersTextEmbedder`
    ) -> None:

        model_name = model.split("/")
        if model_name[-1] == "":
            model_name = "/".join(model_name[-3:-1])
        else:
            model_name = "/".join(model_name[-2:])
        model_name = kwargs.pop("model_name", model_name)
        model_name = f"{model_name}__tokens-v{TOKENIZATION_VERSION}__pool-fp32"
        self.model = self._model_class(model, **kwargs)
        self.mteb_model_meta = ModelMeta(
            name=model_name,
            revision=kwargs.get("revision", None),
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=False,
            public_training_code=None,
            public_training_data=None,
            framework=["Sentence Transformers"],
            similarity_fn_name="cosine",
            use_instructions=True,
            training_datasets=None,
        )

        self.use_instruction = use_instruction
        self.device = device
        self.max_query_length = max_query_length or max_length
        self.max_doc_length = max_doc_length or max_length
        self.amp_dtype = None
        if precision == "fp16":
            self.model.half()
        elif precision == "bf16":
            self.model.bfloat16()
        elif precision.startswith("amp_"):
            self.amp_dtype = (
                torch.float16 if precision.endswith("fp16") else torch.bfloat16
            )
        self.mp_qsize = mp_qsize
        n_gpu = torch.cuda.device_count()
        self.world_size = n_gpu
        assert n_gpu > 0, "woho, no no no!"
        logger.info(f"We have {n_gpu=}, good.")
        self._input_queues = list()
        self._output_queues = list()
        self._workers = list()
        self.instruction_dict = dict()
        self.query_prompt_template = query_prompt_template
        self.document_prompt_template = document_prompt_template
        self.instruction_template = instruction_template
        if instruction_dict_path is None and use_instruction:
            self.instruction_dict = dict(TASK_PROMPTS)
        elif instruction_dict_path is not None:
            with open(instruction_dict_path, encoding="utf-8") as f:
                self.instruction_dict = json.load(f)

    def get_instruction(self, task_name, prompt_type):
        sym_task = False
        if task_name in self.instruction_dict:
            instruction = self.instruction_dict[task_name]
            if isinstance(instruction, dict):
                instruction = instruction.get(prompt_type, "")
                sym_task = True
        else:
            instruction = super().get_instruction(task_name, prompt_type)
        try:
            task_type = mteb.get_tasks(tasks=[task_name])[0].metadata.type
        except Exception:
            task_type = "Retrieval"  # hotfix for additional retrieval tasks
        if "Retrieval" in task_type and not sym_task and prompt_type != "query":
            return ""
        if task_type in ["STS", "PairClassification"]:
            return "Retrieve semantically similar text"
        if task_type in "Bitext Mining":
            return "Retrieve parallel sentences"
        if "Retrieval" in task_type and prompt_type == "query" and instruction is None:
            instruction = "Retrieval relevant passage for the given query."
        return instruction

    def format_instruction(self, instruction, prompt_type):
        if instruction is not None and len(instruction.strip()) > 0:
            instruction = self.instruction_template.format(instruction)
            return instruction
        return ""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        task_instruction: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        instruction = None
        if self.use_instruction:
            instruction = (
                task_instruction
                if task_instruction is not None and prompt_type == PromptType.query
                else self.get_instruction(task_name, prompt_type)
            )
            if self.instruction_template and self.query_prompt_template is None:
                instruction = self.format_instruction(instruction, prompt_type)
            logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")

        num_texts = len(sentences)
        logger.info(f"Encoding {num_texts} sentences.")
        num_batches = num_texts // batch_size + int(num_texts % batch_size > 0)

        def _receive(oq, timeout=0.00125):
            try:
                n, embed = oq.get(timeout=timeout)
                result_dict[n] = embed.cpu()
                pbar.update(1)
                del embed
            except queue.Empty:
                pass

        max_length = (
            self.max_query_length
            if prompt_type == PromptType.query
            else self.max_doc_length
        )

        pbar = tqdm(
            total=num_batches,
            disable=not show_progress_bar,
            desc="encode",
            mininterval=1,
            miniters=10,
        )
        result_dict = dict()
        if not self._workers:
            self.model.to(self.device)

        with nullcontext() if self._workers else torch.inference_mode():
            with (
                nullcontext()
                if self._workers or self.amp_dtype is None
                else torch.autocast(device_type=self.device, dtype=self.amp_dtype)
            ):
                for n, i in enumerate(range(0, num_texts, batch_size)):
                    batch = sentences[i : i + batch_size]
                    prompt = instruction
                    if self.query_prompt_template is not None:
                        # MTEB uses ``passage`` only for the corpus side of asymmetric
                        # retrieval.  Symmetric/classification tasks commonly pass None
                        # and should follow the query/sentence protocol.
                        use_query_template = prompt_type != PromptType.passage
                        template = (
                            self.query_prompt_template
                            if use_query_template
                            else self.document_prompt_template
                        )
                        batch = [
                            format_embedding_text(
                                template,
                                text,
                                task_description=instruction or "",
                            )
                            for text in batch
                        ]
                        prompt = None
                    if self._workers:
                        rank = n % self.world_size
                        self._input_queues[rank].put((n, (batch, max_length, prompt)))
                        if n >= self.world_size:
                            _receive(self._output_queues[rank])
                    else:
                        result_dict[n] = self.model.embed(
                            batch, max_length, prompt, self.device
                        )
                        pbar.update(1)
        if self._workers:
            while len(result_dict) < num_batches:
                for oq in self._output_queues:
                    _receive(oq)

        pbar.close()
        results = [result_dict[n] for n in range(len(result_dict))]
        embeddings = torch.cat(results).float()
        assert embeddings.shape[0] == num_texts
        embeddings = embeddings.cpu().numpy()
        return embeddings

    def start(self):
        self.model.share_memory()
        logger.warning(f"Starting {self.world_size} worker processes.")
        mp_ctx = torch.multiprocessing.get_context("spawn")
        self._input_queues = [
            mp_ctx.Queue(self.mp_qsize) for _ in range(self.world_size)
        ]
        self._output_queues = [
            mp_ctx.Queue(self.mp_qsize) for _ in range(self.world_size)
        ]
        self._workers = list()
        for i, (iq, oq) in enumerate(zip(self._input_queues, self._output_queues)):
            device = torch.device(f"cuda:{i}")
            encode_worker = mp_ctx.Process(
                target=_encode_loop,
                name=f"encode_{i}",
                args=(self.model, iq, oq, device, self.mp_qsize, self.amp_dtype),
            )
            encode_worker.start()
            self._workers.append(encode_worker)
            logger.warning(f"GPU {i} worker initiated.")

    def stop(self):
        [q.put(None) for q in self._input_queues]
        [w.join() for w in self._workers]
        [w.close() for w in self._workers]
        for qs in (self._input_queues, self._output_queues):
            [q.put(None) for q in qs]
