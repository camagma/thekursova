"""Data utilities for building a Ukrainian poetry corpus.

The module supports two primary sources:
1. Ready-to-use Hugging Face datasets (for cases where scraping is not possible).
2. Optional scraping of poetry websites that permit crawling.

Scraped and curated texts are normalized and deduplicated before training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional
import logging
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict, load_dataset

LOGGER = logging.getLogger(__name__)


VOWELS = "аеєиіїоуюяАЕЄИІЇОУЮЯ"


def clean_poem_text(text: str) -> str:
    """Normalize poem text by trimming whitespace and collapsing spaces.

    Args:
        text: Raw poem text (potentially with erratic whitespace).

    Returns:
        Cleaned poem text suitable for training and generation.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class PoemSample:
    text: str
    title: str = ""
    author: str = ""
    url: str = ""


@dataclass
class ScraperConfig:
    base_url: str
    poem_selector: str
    paragraph_selector: str
    title_selector: Optional[str] = None
    page_param: str = "page"
    start_page: int = 1
    end_page: int = 3
    delay_seconds: float = 1.0
    user_agent: str = "poetry-research-bot/0.1"
    obey_robots: bool = True


# Pre-configured selectors for popular open catalogs.
SCRAPER_PRESETS = {
    # poetryclub.com.ua provides a catalogue of poems with ?page=N pagination
    # and poem blocks under div.vers with an optional h3 title.
    "poetryclub": ScraperConfig(
        base_url="https://poetryclub.com.ua/listpoems.php",
        poem_selector="div.vers",
        paragraph_selector="div.vers > p",
        title_selector="div.vers > h3",
        start_page=1,
        end_page=5,
        delay_seconds=1.2,
        obey_robots=True,
    ),
    # Another catalogue page that groups poems by meter (fallback example).
    "poetryclub-meter": ScraperConfig(
        base_url="https://poetryclub.com.ua/metrs/index.php",
        poem_selector="div.vers",
        paragraph_selector="div.vers > p",
        title_selector="div.vers > h3",
        start_page=1,
        end_page=3,
        delay_seconds=1.2,
        obey_robots=True,
    ),
}


class PoemScraper:
    """Simple HTML scraper for poetry websites that allow crawling.

    The scraper relies on CSS selectors for portability across sites. It
    intentionally avoids aggressive crawling and respects robots.txt.
    """

    def __init__(self, config: ScraperConfig):
        self.config = config

    def _allowed(self) -> bool:
        if not self.config.obey_robots:
            LOGGER.warning("Robots.txt check disabled by configuration; crawl responsibly.")
            return True
        robots_url = urljoin(self.config.base_url, "/robots.txt")
        headers = {"User-Agent": self.config.user_agent}
        try:
            resp = requests.get(robots_url, headers=headers, timeout=10)
        except requests.RequestException as exc:
            LOGGER.warning("Robots.txt check failed: %s", exc)
            return False
        if resp.status_code != 200:
            LOGGER.warning("Robots.txt returned status %s", resp.status_code)
            return False
        disallow_root = any(
            line.strip().lower().startswith("disallow: /") for line in resp.text.splitlines()
        )
        return not disallow_root

    def scrape(self) -> List[PoemSample]:
        if not self._allowed():
            raise RuntimeError("Scraping is not allowed by robots.txt. Choose another source.")

        poems: List[PoemSample] = []
        headers = {"User-Agent": self.config.user_agent}
        for page in range(self.config.start_page, self.config.end_page + 1):
            url = f"{self.config.base_url}?{self.config.page_param}={page}"
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as exc:
                LOGGER.warning("Failed to fetch %s: %s", url, exc)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            for block in soup.select(self.config.poem_selector):
                paras = [p.get_text(" ", strip=True) for p in block.select(self.config.paragraph_selector)]
                if not paras:
                    continue
                text = clean_poem_text("\n".join(paras))
                title = ""
                if self.config.title_selector:
                    title_el = block.select_one(self.config.title_selector)
                    if title_el:
                        title = title_el.get_text(strip=True)
                poems.append(PoemSample(text=text, title=title, url=url))
            time.sleep(self.config.delay_seconds)
        return poems


class DatasetBuilder:
    """Utility to combine Hugging Face datasets with optional scraped data."""

    def __init__(self, cleaner: Callable[[str], str] = clean_poem_text):
        self.cleaner = cleaner

    def load_hf(self, name: str, split: str = "train") -> Dataset:
        LOGGER.info("Loading dataset %s", name)
        ds = load_dataset(name, split=split)
        return ds.map(lambda ex: {"text": self.cleaner(ex["text"])})

    def from_samples(self, samples: Iterable[PoemSample]) -> Dataset:
        records = [
            {
                "text": self.cleaner(sample.text),
                "title": sample.title,
                "author": sample.author,
                "url": sample.url,
            }
            for sample in samples
            if sample.text.strip()
        ]
        return Dataset.from_list(records) if records else Dataset.from_list([])

    def combine(self, primary: Dataset, extra: Optional[Dataset] = None) -> DatasetDict:
        if extra is not None and len(extra) > 0:
            merged = Dataset.from_dict({
                "text": primary["text"] + extra["text"],
            })
        else:
            merged = primary
        # Simple deduplication by text
        merged = merged.drop_duplicates("text")
        return DatasetDict({"train": merged})


__all__ = [
    "PoemSample",
    "ScraperConfig",
    "PoemScraper",
    "DatasetBuilder",
    "clean_poem_text",
    "SCRAPER_PRESETS",
]
