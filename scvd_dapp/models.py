"""HuggingFace model construction (encoder / decoder+LoRA), carried over from scvd.models.

Changes from scvd: the class count and names come from the dataset taxonomy, and
the LoRA ``modules_to_save`` head name follows the backend (``score`` for the
Llama/Qwen/StarCoder decoder heads, ``classifier`` for encoders). The decoder
head is still created by ``from_pretrained`` BEFORE ``get_peft_model``, so it is
trainable -- the fix for the earlier broken-LoRA runs.
"""

from __future__ import annotations

import logging

from .config import BACKEND_HF_DECODER, TrainConfig
from .env import load_with_offline_fallback
from .taxonomy import Taxonomy

LOGGER = logging.getLogger("scvd_dapp")


def build_tokenizer(config: TrainConfig):
    from transformers import AutoTokenizer

    source = config.tokenizer_name or config.backbone_init or config.model_name
    tokenizer = load_with_offline_fallback(AutoTokenizer.from_pretrained, source,
                                           trust_remote_code=True, padding_side="right")
    if source != config.model_name:
        LOGGER.info("Tokenizer from %s (%d tokens)", source, len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("Set pad_token = eos_token (%s)", tokenizer.pad_token)
    return tokenizer


def build_model(config: TrainConfig, tokenizer, tax: Taxonomy):
    from transformers import AutoModelForSequenceClassification

    quant_config = None
    if config.load_in_8bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    LOGGER.info("Loading %s (8bit=%s)", config.model_name, config.load_in_8bit)
    model = load_with_offline_fallback(
        AutoModelForSequenceClassification.from_pretrained,
        config.model_name,
        num_labels=tax.num_classes,
        id2label={i: s for i, s in enumerate(tax.swc_ids)},
        label2id={s: i for i, s in enumerate(tax.swc_ids)},
        problem_type="multi_label_classification",
        quantization_config=quant_config,
        device_map="auto" if config.load_in_8bit else None,
        trust_remote_code=True,
        # a checkpoint that already carries a classification head with a different class
        # count gets a fresh 13-way head instead of a load error (base checkpoints have no head)
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    if config.load_in_8bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    if config.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        head = ["score"] if config.backend == BACKEND_HF_DECODER else ["classifier"]
        lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=config.lora_r, lora_alpha=config.lora_alpha,
                          lora_dropout=config.lora_dropout, target_modules=config.lora_target_modules,
                          bias="none", modules_to_save=head)
        model = get_peft_model(model, lora)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        LOGGER.info("LoRA applied: %d / %d trainable (%.2f%%)", trainable, total, 100 * trainable / total)
    return model
