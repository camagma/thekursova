"""Inference helpers for poem generation with rhyme checks."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

VOWELS = "аеєиіїоуюяAEЄИІЇОУЮЯ"


def rhyme_suffix(word: str, min_len: int = 2) -> str:
    match = re.findall(f"[{VOWELS}][^{VOWELS}]*$", word.lower())
    if not match:
        return word.lower()[-min_len:]
    suffix = match[-1]
    return suffix if len(suffix) >= min_len else word.lower()[-min_len:]


def count_syllables(line: str) -> int:
    return sum(1 for ch in line if ch.lower() in VOWELS.lower())


@dataclass
class GenerationConfig:
    model_path: str = "poetry-model"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 64
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.9
    num_return_sequences: int = 5
    rhyme_scheme: str = "ABAB"
    expected_syllables: int = 10


class PoemGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        LOGGER.info("Loading generator from %s", config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path).to(config.device)

    def _valid_rhyme(self, lines: List[str]) -> bool:
        rhymes: Dict[str, str] = {}
        for i, line in enumerate(lines):
            label = self.config.rhyme_scheme[i % len(self.config.rhyme_scheme)]
            last_word = line.split()[-1]
            suf = rhyme_suffix(last_word)
            if label in rhymes and rhymes[label] != suf:
                return False
            rhymes[label] = suf
        return True

    def _syllable_ok(self, line: str) -> bool:
        return abs(count_syllables(line) - self.config.expected_syllables) <= 3

    def generate(self, prompt: str, lines: int = 4) -> List[str]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            num_return_sequences=self.config.num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        for seq in outputs:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            poem_lines = [l.strip() for l in text.split("\n") if l.strip()][-lines:]
            if len(poem_lines) < lines:
                continue
            if not all(self._syllable_ok(line) for line in poem_lines):
                continue
            if self._valid_rhyme(poem_lines):
                LOGGER.info("Generated poem with rhyme %s", self.config.rhyme_scheme)
                return poem_lines
        LOGGER.warning("No sequence matched rhyme/metric constraints; returning first candidate.")
        if len(outputs) > 0:
            return [l.strip() for l in self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n") if l.strip()][:lines]
        return []


__all__ = ["PoemGenerator", "GenerationConfig", "rhyme_suffix", "count_syllables"]
