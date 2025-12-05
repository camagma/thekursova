"""Command-line interface for building, training, and using the poetry model.

The module supports two invocation styles:
1) Preferred: ``python -m poetry_ai.cli <command> ...``
2) Direct script call: ``python poetry_ai/cli.py <command> ...``

The fallback import shim below keeps relative imports working when the file is
executed as a script (avoiding ``ImportError: attempted relative import``).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset

try:  # pragma: no cover - import shim for script-style execution
    from .data import DatasetBuilder, PoemScraper, ScraperConfig, SCRAPER_PRESETS
    from .generation import GenerationConfig, PoemGenerator, rhyme_suffix
    from .training import PoetryTrainer, TrainingConfig
except ImportError:  # when __package__ is None (python poetry_ai/cli.py)
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from poetry_ai.data import DatasetBuilder, PoemScraper, ScraperConfig, SCRAPER_PRESETS
    from poetry_ai.generation import GenerationConfig, PoemGenerator, rhyme_suffix
    from poetry_ai.training import PoetryTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


DEFAULT_HF_DATASET = "staliuk/ukrainian-poetry"
DEFAULT_MODEL = "facebook/xglm-1.7B"


def build_dataset(hf_dataset: str, scraped_path: Optional[Path]) -> Dataset:
    builder = DatasetBuilder()
    base = builder.load_hf(hf_dataset)
    scraped_ds: Optional[Dataset] = None
    if scraped_path and scraped_path.exists():
        samples = json.loads(scraped_path.read_text(encoding="utf-8"))
        scraped_ds = builder.from_samples(samples)
        LOGGER.info("Loaded %d scraped poems from %s", len(scraped_ds), scraped_path)
    ds_dict = builder.combine(base, scraped_ds)
    return ds_dict


def build_scraper_config(args) -> ScraperConfig:
    if args.preset:
        base = SCRAPER_PRESETS[args.preset]
        config = ScraperConfig(
            base_url=args.base_url or base.base_url,
            poem_selector=args.poem_selector or base.poem_selector,
            paragraph_selector=args.paragraph_selector or base.paragraph_selector,
            title_selector=args.title_selector or base.title_selector,
            page_param=base.page_param,
            start_page=args.start_page or base.start_page,
            end_page=args.end_page or base.end_page,
            delay_seconds=args.delay if args.delay is not None else base.delay_seconds,
            user_agent=base.user_agent,
            obey_robots=base.obey_robots,
        )
        LOGGER.info("Using preset '%s' with base URL %s", args.preset, config.base_url)
        return config

    if not (args.base_url and args.poem_selector and args.paragraph_selector):
        raise SystemExit("When no preset is specified, base_url and selectors are required.")

    return ScraperConfig(
        base_url=args.base_url,
        poem_selector=args.poem_selector,
        paragraph_selector=args.paragraph_selector,
        title_selector=args.title_selector,
        start_page=args.start_page,
        end_page=args.end_page,
        delay_seconds=args.delay if args.delay is not None else 1.0,
    )


def run_scrape(args):
    config = build_scraper_config(args)
    scraper = PoemScraper(config)
    poems = scraper.scrape()
    output = Path(args.output)
    output.write_text(json.dumps([p.__dict__ for p in poems], ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved %d poems to %s", len(poems), output)


def run_train(args):
    ds_dict = build_dataset(args.dataset, Path(args.scraped) if args.scraped else None)
    trainer = PoetryTrainer(
        TrainingConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_length=args.max_length,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            fp16=args.fp16,
        )
    )
    trainer.train(ds_dict)


def run_generate(args):
    gen = PoemGenerator(
        GenerationConfig(
            model_path=args.model_path,
            rhyme_scheme=args.rhyme_scheme,
            expected_syllables=args.expected_syllables,
            max_new_tokens=args.max_new_tokens,
        )
    )
    poem = gen.generate(args.prompt, lines=args.lines)
    if not poem:
        print("Не вдалося згенерувати вірш із заданими обмеженнями рими/складів.")
        return

    decorated = []
    for idx, line in enumerate(poem):
        label = args.rhyme_scheme[idx % len(args.rhyme_scheme)]
        last_word = line.split()[-1] if line.split() else ""
        suffix = rhyme_suffix(last_word) if last_word else ""
        syllables = sum(ch.lower() in "аеєиіїоуюя" for ch in line.lower())
        decorated.append(f"{idx + 1:>2}. ({label}:{suffix}, {syllables} складів)  {line}")

    frame_width = max(len(line) for line in decorated)
    horizontal = "═" * (frame_width + 2)
    print(f"╔{horizontal}╗")
    for line in decorated:
        padding = " " * (frame_width - len(line))
        print(f"║ {line}{padding} ║")
    print(f"╚{horizontal}╝")


def main():
    parser = argparse.ArgumentParser(description="Ukrainian poetry AI pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    scrape = sub.add_parser("scrape", help="Scrape poems from a website")
    scrape.add_argument("base_url", nargs="?", help="Base URL with paginated poems, e.g. https://example.org/poems")
    scrape.add_argument("poem_selector", nargs="?", help="CSS selector for poem container")
    scrape.add_argument("paragraph_selector", nargs="?", help="CSS selector for paragraphs within a poem container")
    scrape.add_argument("--title-selector", default=None, help="CSS selector for poem title inside container")
    scrape.add_argument("--start-page", type=int)
    scrape.add_argument("--end-page", type=int)
    scrape.add_argument("--delay", type=float, help="Delay between page requests in seconds")
    scrape.add_argument("--preset", choices=sorted(SCRAPER_PRESETS.keys()), help="Use a predefined site profile")
    scrape.add_argument("--output", default="scraped_poems.json", help="Path to save scraped poems")
    scrape.set_defaults(func=run_scrape)

    train = sub.add_parser("train", help="Fine-tune the model on poetry")
    train.add_argument("--dataset", default=DEFAULT_HF_DATASET, help="Hugging Face dataset name")
    train.add_argument("--scraped", help="Optional path to scraped_poems.json")
    train.add_argument("--model-name", default=DEFAULT_MODEL, help="Base model to fine-tune")
    train.add_argument("--output-dir", default="poetry-model", help="Where to save the fine-tuned model")
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--train-batch-size", type=int, default=4)
    train.add_argument("--eval-batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--num-train-epochs", type=int, default=3)
    train.add_argument("--fp16", action="store_true")
    train.set_defaults(func=run_train)

    generate = sub.add_parser("generate", help="Generate a poem")
    generate.add_argument("prompt", help="Prompt to start the poem")
    generate.add_argument("--model-path", default="poetry-model")
    generate.add_argument("--lines", type=int, default=4)
    generate.add_argument("--rhyme-scheme", default="ABAB")
    generate.add_argument("--expected-syllables", type=int, default=10)
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.set_defaults(func=run_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
