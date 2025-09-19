from __future__ import annotations

import os
from typing import Any
from cs336_data.filter_common_crawl.extract_data import extract_text
from cs336_data.filter_common_crawl.language_identification import language_identification
from cs336_data.filter_common_crawl.mask_pii import mask_email, mask_phone_number, mask_ip
from cs336_data.filter_common_crawl.harmful_context import toxic_detect, nsfw_detect
from cs336_data.filter_common_crawl.quality_rules import quality_filters, quality_classifier

from cs336_data.deduplication.exact_deduplication import deduplication
from cs336_data.deduplication.minhash_deduplication import minhash_deduplication
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return language_identification(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_email(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_number(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ip(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return nsfw_detect(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return toxic_detect(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return quality_classifier(text)


def run_gopher_quality_filter(text: str) -> bool:
    return quality_filters(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
