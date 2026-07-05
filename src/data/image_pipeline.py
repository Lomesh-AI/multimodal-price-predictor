"""Image downloading and preprocessing.

Every function here degrades gracefully: a single failed URL never crashes
a batch job. Failures are logged and replaced with a black placeholder image
so the row is still usable downstream (matching what the pipeline design
doc calls for in stage 01).
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from PIL import Image, UnidentifiedImageError

from src.utils.exceptions import ImageDownloadError
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S = 10
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 1.5


def _placeholder(size: int) -> Image.Image:
    return Image.new("RGB", (size, size), color=(0, 0, 0))


def download_image(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> Image.Image:
    """Download and decode one image, retrying transient failures.

    Raises:
        ImageDownloadError: if all retries are exhausted. Callers that want
        graceful degradation should use process_image() instead, which
        catches this and returns a placeholder.
    """
    if not url or not isinstance(url, str):
        raise ImageDownloadError(f"Invalid image URL: {url!r}")

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image.load()
            return image.convert("RGB")
        except (requests.RequestException, UnidentifiedImageError, OSError) as e:
            last_error = e
            logger.debug("Attempt %d/%d failed for %s: %s", attempt, retries, url, e)
            if attempt < retries:
                time.sleep(backoff_s * attempt)

    raise ImageDownloadError(
        f"Failed to download image after {retries} attempts: {url} ({last_error})"
    )


def process_image(url: str, size: int = 384) -> Tuple[Image.Image, bool]:
    """Download + resize an image, falling back to a placeholder on failure.

    Returns:
        (image, was_placeholder) — was_placeholder lets callers track data
        quality (e.g. log what fraction of a batch used placeholders).
    """
    try:
        image = download_image(url)
        was_placeholder = False
    except ImageDownloadError as e:
        logger.warning("Using placeholder image: %s", e)
        image = _placeholder(size)
        was_placeholder = True

    try:
        image = image.resize((size, size), Image.BICUBIC)
    except (ValueError, OSError) as e:
        logger.warning("Resize failed (%s), substituting placeholder", e)
        image = _placeholder(size)
        was_placeholder = True

    return image, was_placeholder


def process_batch(
    urls: List[str],
    output_dir: str,
    size: int = 384,
    max_workers: int = 16,
) -> List[dict]:
    """Download + resize a batch of images in parallel, saving each to disk.

    Returns a list of per-row status dicts: {"url", "path", "placeholder"}.
    This function never raises for individual failures — check the returned
    "placeholder" flags to see how many rows degraded.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Optional[dict]] = [None] * len(urls)

    def _worker(index: int, url: str) -> dict:
        image, was_placeholder = process_image(url, size=size)
        file_path = out_dir / f"{index:07d}.jpg"
        try:
            image.save(file_path, format="JPEG", quality=90)
        except OSError as e:
            raise ImageDownloadError(f"Failed to save image to {file_path}: {e}") from e
        return {"url": url, "path": str(file_path), "placeholder": was_placeholder}

    n_failed_save = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, i, url): i for i, url in enumerate(urls)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except ImageDownloadError as e:
                logger.error("Row %d failed permanently: %s", idx, e)
                n_failed_save += 1
                results[idx] = {"url": urls[idx], "path": None, "placeholder": True}

    n_placeholder = sum(1 for r in results if r and r["placeholder"])
    logger.info(
        "Processed %d images: %d placeholders, %d hard failures",
        len(urls), n_placeholder, n_failed_save,
    )
    return results
