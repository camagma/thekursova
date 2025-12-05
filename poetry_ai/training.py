"""Training utilities for fine-tuning an autoregressive model on poetry."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str = "facebook/xglm-564M"
    output_dir: str = "poetry-model"
    max_length: int = 256
    train_batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    warmup_steps: int = 300
    eval_steps: int = 200
    save_steps: int = 500
    logging_steps: int = 100
    fp16: bool = False


class PoetryTrainer:
    """Wrapper around Hugging Face Trainer with poetry-focused defaults."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    def _prepare_tokenizer(self) -> AutoTokenizer:
        LOGGER.info("Loading tokenizer %s", self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _tokenize_function(self, batch):
        assert self.tokenizer is not None
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
        )

    def train(self, dataset: DatasetDict):
        if "train" not in dataset:
            raise ValueError("DatasetDict must contain a 'train' split.")

        self.tokenizer = self._prepare_tokenizer()
        LOGGER.info("Loading model %s", self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        tokenized = dataset["train"].map(self._tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
        tokenized_ds = DatasetDict({"train": tokenized})

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            fp16=self.config.fp16,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_ds["train"],
            data_collator=data_collator,
        )

        LOGGER.info("Starting training on %d samples", len(tokenized_ds["train"]))
        trainer.train()
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        LOGGER.info("Model saved to %s", self.config.output_dir)


__all__ = ["TrainingConfig", "PoetryTrainer"]
