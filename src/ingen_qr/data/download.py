import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path

import requests


class TemporaryDownloadError(RuntimeError):
    pass


class Downloader:
    def __init__(
        self,
        cache_dir: str = "data/cache/downloads",
        min_interval: float = 0.5,
        max_attempts: int = 5,
        headers: dict[str, str] | None = None,
    ) -> None:
        if min_interval < 0 or max_attempts < 1:
            raise ValueError("Invalid download settings.")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.last_request = 0.0
        self.cache_hits = 0
        self.session = requests.Session()
        self.session.headers.update(headers or {})

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        self.last_request = time.monotonic()

    def _retry_after(self, error: Exception) -> float:
        if not isinstance(error, requests.HTTPError) or error.response is None:
            return 0.0

        value = error.response.headers.get("Retry-After")
        if not value:
            return 0.0

        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                target = parsedate_to_datetime(value)
                return max(
                    0.0,
                    (target - datetime.now(UTC)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                return 0.0

    def cached(
        self,
        key: str,
        fetch: Callable[[], bytes],
        refresh: bool = False,
    ) -> bytes:
        name = sha256(key.encode("utf-8")).hexdigest()
        path = self.cache_dir / (name + ".bin")

        if path.exists() and not refresh:
            self.cache_hits += 1
            return path.read_bytes()

        for attempt in range(self.max_attempts):
            self._throttle()
            try:
                content = fetch()
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
                TemporaryDownloadError,
            ) as error:
                if isinstance(error, requests.HTTPError) and (
                    error.response is None
                    or error.response.status_code not in (429, 500, 502, 503, 504)
                ):
                    raise

                if attempt + 1 == self.max_attempts:
                    raise

                delay = max(2.0**attempt, self._retry_after(error))
                time.sleep(delay + random.uniform(0.0, 0.25))
            else:
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
                return content

        raise RuntimeError("Download failed.")

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        refresh: bool = False,
    ) -> bytes:
        prepared = requests.Request(
            "GET", url, params=sorted((params or {}).items())
        ).prepare()
        target = prepared.url

        if target is None:
            raise ValueError("Invalid URL.")

        def fetch() -> bytes:
            with self.session.get(target, timeout=(10, 30)) as response:
                response.raise_for_status()
                return response.content

        return self.cached("http:" + target, fetch, refresh)
