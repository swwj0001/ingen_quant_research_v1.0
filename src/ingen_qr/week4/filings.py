"""SEC filing discovery, authoritative timestamps and narrative extraction."""

import json
import os
import random
import re
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter, sleep
from typing import Any
from uuid import uuid4

import pandas as pd
import requests
from dotenv import load_dotenv

from ingen_qr.data.download import Downloader, TemporaryDownloadError

FORMS = {"8-K", "10-K", "10-Q"}
NEW_YORK = "America/New_York"
SUBMISSIONS = "https://data.sec.gov/submissions/"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/"
ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
SEC_FILENAME = re.compile(r"[A-Za-z0-9_.-]+")
LEGACY_HEADER = re.compile(r"<ACCEPTANCE-DATETIME>\s*(\d{14})(?!\d)", re.IGNORECASE)
INDEX_CLOCK = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
SECTIONS = re.compile(
    r"^item\s*(\d{1,2}(?:\.\d{2}|[A-Z])?)\s*[.\-–—:]?\s*(.*)", re.IGNORECASE
)
WORD = re.compile(r"\b[A-Za-z]+(?:['’-][A-Za-z]+)*\b")
TARGET_SECTIONS = {
    "10-K": {"7": ("mdna", "management"), "1A": ("risk_factors", "risk")},
    "10-Q": {"2": ("mdna", "management"), "1A": ("risk_factors", "risk")},
}
SAMPLE_CLOCKS = {
    "0000320193-23-000006": ("2023-02-02 18:01:30", "2023-02-02T23:01:30+00:00"),
    "0000320193-23-000077": ("2023-08-03 18:04:43", "2023-08-03T22:04:43+00:00"),
    "0000320193-23-000106": ("2023-11-02 18:08:27", "2023-11-02T22:08:27+00:00"),
}


SAFE_HARBOR_HEADING = re.compile(
    r"^(?:(?:a |special |important |cautionary )*(?:note|statement|information) "
    r"(?:on|about|regarding|concerning|with respect to) )?"
    r"(?:forward[ -]looking statements?|safe[ -]harbo[u]r(?: statements?| notice)?)"
    r"(?: under (?:the )?private securities litigation reform act(?: of 1995)?)?[.:]?$",
    re.IGNORECASE,
)
SAFE_HARBOR_PARAGRAPH = re.compile(
    r"\bprivate securities litigation reform act\b|"
    r"\bsafe[ -]harbo[u]r (?:provisions?|protections?|statement)"
    r"[^.]{0,120}forward[ -]looking statements?\b|"
    r"\b(?:this|the) (?:report|filing|(?:press |news )?release|document|communication|presentation)"
    r"(?:[^.]{0,70})(?:contains?|includes?|may contain|may include) "
    r"(?:certain |some )?forward[ -]looking statements?\b|"
    r"\b(?:certain|some) statements (?:in|included in) this "
    r"(?:(?:news|press) )?(?:report|filing|release|document|communication) "
    r"(?:are|may be) forward[ -]looking statements?\b|"
    r"\bforward[ -]looking statements? (?:are (?:not guarantees|subject to)|"
    r"should not be relied (?:on|upon)|speak only|include,? without limitation)\b|"
    r"\b(?:do not|does not|disclaim(?:s)?) (?:undertake |assume |any )*"
    r"(?:an? |any )?(?:duty|obligation) to (?:publicly )?(?:update|revise)\b|"
    r"\b(?:actual results|actual outcomes) (?:could|may|might|can) differ materially "
    r"(?:from (?:those|what)|(?:due to|as a result of) (?:various |a number of ))",
    re.IGNORECASE,
)
SAFE_HARBOR_CONTINUATION = re.compile(
    r"^(?:these|such|all) (?:statements|forward[ -]looking statements)|"
    r"^(?:words|phrases) such as |^you (?:should|are cautioned) not to place undue reliance",
    re.IGNORECASE,
)


SEC_MAX_REQUESTS_PER_SECOND = 8.0
SEC_MAX_DOWNLOAD_WORKERS = 8


class RateLimiter:
    """Thread-safe spacing of request starts across every downloader that shares it."""

    def __init__(self, requests_per_second: float) -> None:
        rate = float(requests_per_second)
        if not 0 < rate <= SEC_MAX_REQUESTS_PER_SECOND:
            raise ValueError(
                f"SEC request rate must be above 0 and at most {SEC_MAX_REQUESTS_PER_SECOND:g} per second."
            )
        self.interval = 1.0 / rate
        self._lock = Lock()
        self._next_start = 0.0

    def wait(self) -> float:
        """Block until this thread may send; return the seconds spent waiting.

        A thread claims a send slot only while holding the lock and only once the
        slot is due. Threads that wake late re-check, so a stall cannot release
        several overdue requests at once.
        """
        started = monotonic()
        while True:
            with self._lock:
                now = monotonic()
                if now >= self._next_start:
                    self._next_start = now + self.interval
                    return now - started
                delay = self._next_start - now
            sleep(delay)


def replace_cache_file(temporary: Path, path: Path, attempts: int = 5) -> None:
    """Atomically install a cache file, tolerating Windows sharing conflicts.

    On Windows, os.replace raises PermissionError while another thread is
    replacing or reading the same target. Cache entries for one key hold the
    same URL body, so a target that already exists is an equally valid result.
    """
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if path.exists():
                return
            if attempt + 1 == attempts:
                raise
            sleep(0.05 * (attempt + 1))


class LimitedSession(requests.Session):
    """Session that waits on the shared limiter before every HTTP request it sends.

    `Session.send` is also used for redirects, so followed redirects count against
    the SEC rate cap exactly like the first request.
    """

    def __init__(self, limiter: RateLimiter) -> None:
        super().__init__()
        self.limiter = limiter
        self.waited = 0.0
        self.requests_sent = 0

    def send(  # type: ignore[override]
        self, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        self.waited += self.limiter.wait()
        self.requests_sent += 1
        return super().send(request, **kwargs)


class SecDownloader(Downloader):
    """Week 1 cached downloader with a shared SEC rate limit and network timing."""

    def __init__(
        self, cache_dir: str, limiter: RateLimiter, headers: dict[str, str]
    ) -> None:
        super().__init__(cache_dir, min_interval=0.0, headers=headers)
        self.session.close()
        self.session = LimitedSession(limiter)
        self.session.headers.update(headers)
        self.limiter = limiter
        self.calls = 0
        self.uncached_calls = 0
        self.download_seconds = 0.0

    @property
    def http_requests(self) -> int:
        session = self.session
        return session.requests_sent if isinstance(session, LimitedSession) else 0

    @property
    def limiter_seconds(self) -> float:
        session = self.session
        return session.waited if isinstance(session, LimitedSession) else 0.0

    def _throttle(self) -> None:
        # Rate limiting happens per HTTP request in LimitedSession.send.
        self.last_request = monotonic()

    def cached(
        self,
        key: str,
        fetch: Callable[[], bytes],
        refresh: bool = False,
    ) -> bytes:
        """Week 1 cache and retry rules, with a unique temporary file per write."""
        path = self.cache_dir / (sha256(key.encode("utf-8")).hexdigest() + ".bin")
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
                sleep(delay + random.uniform(0.0, 0.25))
            else:
                temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
                try:
                    temporary.write_bytes(content)
                    replace_cache_file(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                return content
        raise RuntimeError("Download failed.")

    def get(
        self, url: str, params: dict[str, str] | None = None, refresh: bool = False
    ) -> bytes:
        started = perf_counter()
        hits = self.cache_hits
        waited = self.limiter_seconds
        try:
            return super().get(url, params=params, refresh=refresh)
        finally:
            self.calls += 1
            if self.cache_hits == hits:
                self.uncached_calls += 1
                # Network time only: shared rate-limit waits are excluded.
                self.download_seconds += (perf_counter() - started) - (
                    self.limiter_seconds - waited
                )

    def measurements(self) -> tuple[int, int, float]:
        return self.calls, self.uncached_calls, self.download_seconds


@dataclass(frozen=True)
class Filing:
    cik: int
    accession: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str
    submissions_clock: str
    items: str

    @property
    def base(self) -> str:
        return f"{ARCHIVES}{self.cik}/{self.accession.replace('-', '')}/"

    @property
    def index(self) -> str:
        return self.base + self.accession + "-index.htm"

    @property
    def header(self) -> str:
        return self.base + self.accession + "-index-headers.html"


def soup(content: bytes):
    bs4 = import_module("bs4")
    # Inline XBRL filings start with an XML declaration but are XHTML documents.
    warnings.filterwarnings("ignore", category=bs4.XMLParsedAsHTMLWarning)
    return bs4.BeautifulSoup(content, "lxml")


def sec_rate(settings: dict[str, Any]) -> float:
    rate = float(settings["sec_max_requests_per_second"])
    if not 0 < rate <= SEC_MAX_REQUESTS_PER_SECOND:
        raise ValueError(
            f"sec_max_requests_per_second must be above 0 and at most {SEC_MAX_REQUESTS_PER_SECOND:g}."
        )
    return rate


def download_workers(settings: dict[str, Any]) -> int:
    workers = int(settings["sec_download_workers"])
    if not 1 <= workers <= SEC_MAX_DOWNLOAD_WORKERS:
        raise ValueError(
            f"sec_download_workers must be between 1 and {SEC_MAX_DOWNLOAD_WORKERS}."
        )
    return workers


def sec_client(
    settings: dict[str, Any], cache_dir: str, limiter: RateLimiter | None = None
) -> SecDownloader:
    """Return a downloader with its own session. Share `limiter` across threads."""
    load_dotenv()
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in user_agent:
        raise ValueError(
            "Set SEC_USER_AGENT to your name or organization and email (see .env.example)."
        )
    return SecDownloader(
        cache_dir,
        limiter or RateLimiter(sec_rate(settings)),
        headers={"User-Agent": user_agent},
    )


def fetch(client: Downloader, url: str, *, refresh: bool = False) -> bytes | None:
    try:
        return client.get(url, refresh=refresh) if refresh else client.get(url)
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code in {404, 410}:
            return None
        raise RuntimeError(
            f"SEC request failed for {url}. Check SEC_USER_AGENT and resume with the cache."
        ) from error
    except (
        requests.ConnectionError,
        requests.Timeout,
        TemporaryDownloadError,
    ) as error:
        raise RuntimeError(
            f"SEC unavailable at {url}. Re-run to resume from cache."
        ) from error


def filings_for(client: Downloader, cik: int, start: str, end: str) -> Iterator[Filing]:
    content = fetch(client, f"{SUBMISSIONS}CIK{cik:010d}.json")
    if content is None:
        return
    data = json.loads(content)["filings"]
    sets = [data["recent"]]
    for archive in data["files"]:
        if archive["filingTo"] < start or archive["filingFrom"] > end:
            continue
        name = archive["name"]
        if SEC_FILENAME.fullmatch(name) is None:
            raise ValueError(f"Invalid SEC submissions archive filename: {name!r}")
        content = fetch(client, SUBMISSIONS + name)
        if content is None:
            raise RuntimeError(f"SEC history shard is missing: {name}")
        sets.append(json.loads(content))
    seen: set[str] = set()
    fields = (
        "form",
        "filingDate",
        "reportDate",
        "primaryDocument",
        "acceptanceDateTime",
        "items",
    )
    for table in sets:
        accessions = table["accessionNumber"]
        values = [table.get(field, [""] * len(accessions)) for field in fields]
        for accession, form, date, period, primary, accepted, items in zip(
            accessions, *values, strict=True
        ):
            if accession in seen or form not in FORMS or not start <= date <= end:
                continue
            if ACCESSION.fullmatch(accession) is None:
                raise ValueError(f"Invalid SEC accession: {accession!r}")
            seen.add(accession)
            yield Filing(
                cik,
                accession,
                form,
                date,
                period or "",
                primary or "",
                accepted or "",
                items or "",
            )


def index_details(content: bytes) -> tuple[str | None, dict[str, str]]:
    parsed = soup(content)
    details = {}
    for heading in parsed.select("div.infoHead"):
        value = heading.find_next_sibling("div", class_="info")
        if value is not None:
            details[heading.get_text(strip=True)] = value.get_text(" ", strip=True)
    documents: dict[str, str] = {}
    for row in parsed.select("table.tableFile tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 4:
            continue
        form = cells[3].get_text(strip=True)
        link = cells[2].find("a")
        if link is not None:
            name = link.get_text(" ", strip=True).split()
            if name and SEC_FILENAME.fullmatch(name[0]):
                documents.setdefault(form, name[0])
    return details.get("Accepted"), documents


def header_clock(content: bytes) -> str | None:
    match = LEGACY_HEADER.search(soup(content).get_text())
    if match is None:
        match = LEGACY_HEADER.search(content.decode("ascii", errors="ignore")[:30000])
    if match is None:
        return None
    try:
        return datetime.strptime(match[1], "%Y%m%d%H%M%S").isoformat(" ")
    except ValueError:
        return None


def local_clock(value: str | None) -> pd.Timestamp | None:
    if not value or not INDEX_CLOCK.fullmatch(value):
        return None
    try:
        stamp = pd.Timestamp(value).tz_localize(
            NEW_YORK, ambiguous="NaT", nonexistent="NaT"
        )
    except ValueError:
        return None
    return None if pd.isna(stamp) else stamp


def timing(
    index_value: str | None, header_value: str | None, submissions: str, calendar: Any
) -> dict[str, Any]:
    index_time, header_time = local_clock(index_value), local_clock(header_value)
    conflict = (
        index_time is not None and header_time is not None and index_time != header_time
    )
    authoritative = None if conflict else header_time or index_time
    audit: dict[str, Any] = {
        "index_clock": index_value,
        "header_clock": header_value,
        "submissions_clock": submissions,
        "accepted_local": None if authoritative is None else authoritative.isoformat(),
        "accepted_utc": pd.NaT,
        "available_at": pd.NaT,
        "clock_source": None,
        "submissions_check": "unverified",
        "status": (
            "authoritative_clock_conflict" if conflict else "unconfirmed_acceptance"
        ),
    }
    if authoritative is None:
        return audit
    accepted = authoritative.tz_convert("UTC")
    check = "missing" if not submissions else "unparseable"
    if submissions:
        try:
            candidate = pd.Timestamp(submissions)
        except ValueError:
            candidate = None
        if (
            candidate is not None
            and pd.notna(candidate)
            and candidate.tzinfo is not None
        ):
            check = "match" if candidate == accepted else "mismatch"
    next_date = authoritative.tz_localize(None).normalize() + pd.Timedelta(days=1)
    next_session = calendar.date_to_session(next_date, direction="next")
    audit.update(
        accepted_utc=accepted,
        available_at=calendar.session_open(next_session),
        clock_source="sec_header" if header_time is not None else "sec_index",
        submissions_check=check,
        status="ok",
    )
    return audit


def audit_timezone(
    client: Downloader, calendar: Any, *, refresh: bool = False
) -> list[dict[str, Any]]:
    results = []
    report = fetch(client, f"{SUBMISSIONS}CIK0000320193.json", refresh=refresh)
    if report is None:
        raise RuntimeError(
            "SEC Apple submissions were unavailable for clock verification."
        )
    table = json.loads(report)["filings"]["recent"]
    clocks = dict(
        zip(table["accessionNumber"], table["acceptanceDateTime"], strict=True)
    )
    for accession, (expected_wall, expected_utc) in SAMPLE_CLOCKS.items():
        filing = Filing(
            320193, accession, "", "", "", "", clocks.get(accession, ""), ""
        )
        index = fetch(client, filing.index, refresh=refresh)
        header = fetch(client, filing.header, refresh=refresh)
        wall = index_details(index)[0] if index else None
        from_header = header_clock(header) if header else None
        if wall != expected_wall or from_header != expected_wall:
            raise RuntimeError(
                f"SEC clock sample failed for {accession}: index={wall}, header={from_header}."
            )
        result = timing(wall, from_header, filing.submissions_clock, calendar)
        if result["accepted_utc"] != pd.Timestamp(expected_utc):
            raise RuntimeError(f"Unexpected New York clock conversion for {accession}.")
        results.append(
            {
                "accession": accession,
                "index_url": filing.index,
                "header_url": filing.header,
                **result,
            }
        )
    return results


def sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z(])|\n+", text)
        if part.strip()
    ]


def clean_section(lines: list[str]) -> str:
    output = []
    seen = set()
    safe_harbor = False
    for line in lines:
        if SAFE_HARBOR_HEADING.fullmatch(line) or SAFE_HARBOR_PARAGRAPH.search(line):
            safe_harbor = True
            continue
        if safe_harbor and SAFE_HARBOR_CONTINUATION.search(line):
            continue
        safe_harbor = False
        for sentence in sentences(line):
            canonical = re.sub(r"\W+", " ", sentence.lower()).strip()
            if len(WORD.findall(sentence)) < 5 or canonical in seen:
                continue
            if re.search(
                r"^table of contents$|^pursuant to the requirements of the securities exchange act|"
                r"^the information contained in this (?:item|report).*shall not be deemed",
                sentence,
                re.IGNORECASE,
            ):
                continue
            seen.add(canonical)
            output.append(sentence)
    return "\n".join(output)


def narrative_sections(content: bytes, form: str, minimum_words: int) -> dict[str, str]:
    if re.search(rb"<DOCUMENT>\s*<TYPE>", content[:500000], re.IGNORECASE):
        for document in re.split(rb"<DOCUMENT>", content, flags=re.IGNORECASE)[1:]:
            markup = re.match(
                rb"\s*<TYPE>\s*([^\r\n<]+)", document, flags=re.IGNORECASE
            )
            if markup and markup[1].decode("ascii", "ignore").strip() == form:
                content = re.split(
                    rb"<TEXT>\s*", document, maxsplit=1, flags=re.IGNORECASE
                )[-1]
                content = re.split(
                    rb"</TEXT>|</DOCUMENT>", content, maxsplit=1, flags=re.IGNORECASE
                )[0]
                break
        else:
            return {}
    parsed = soup(content)
    for tag in parsed.find_all(
        ["script", "style", "head", "ix:header", "ix:hidden", "noscript"]
    ):
        tag.decompose()
    for tag in parsed.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        if tag.parent is not None:
            tag.decompose()
    for table in parsed.find_all("table"):
        if table.parent is None or table.find("table") is not None:
            continue
        cells = table.find_all("td")
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if (
            len(values) >= 5
            and sum(bool(re.match(r"^[\d$,.(%–—− -]+$", value)) for value in values)
            >= len(values) / 3
        ) or (len(table.find_all("a", href=re.compile("^#"))) >= 3):
            table.decompose()
    for tag in parsed.find_all(
        ["br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "pre"]
    ):
        tag.insert_after("\n")
    lines = [
        " ".join(part.split())
        for part in parsed.get_text(" ").replace("\u00a0", " ").splitlines()
    ]
    candidates: dict[str, list[str]] = {}
    bucket: list[str] = []
    key = ""
    for line in lines:
        match = SECTIONS.match(line)
        if match and len(line) < 240 and (len(match[2]) < 145 or form == "8-K"):
            item, heading = match[1].upper(), match[2].lstrip()
            if form == "8-K":
                target = "current_report" if "." in item and item != "9.01" else ""
            else:
                section, token = TARGET_SECTIONS[form].get(item, ("", ""))
                target = section if not heading or token in heading.lower() else ""
            if key:
                candidates.setdefault(key, []).append(clean_section(bucket))
            bucket = [heading] if form == "8-K" else []
            key = target
        elif re.fullmatch(
            r"(?:part\s+(?:I|II|III|IV)\b.*|signatures?|exhibit index)",
            line,
            flags=re.IGNORECASE,
        ):
            if key:
                candidates.setdefault(key, []).append(clean_section(bucket))
            bucket, key = [], ""
        elif key and line:
            bucket.append(line)
    if key:
        candidates.setdefault(key, []).append(clean_section(bucket))
    result = {}
    for section, texts in candidates.items():
        text = (
            clean_section(texts)
            if form == "8-K"
            else max(texts, key=lambda item: len(WORD.findall(item)))
        )
        if len(WORD.findall(text)) >= minimum_words:
            result[section] = text
    return result


def retrieve(
    filing: Filing,
    client: Downloader,
    calendar: Any,
    minimum: int,
    start: str,
    end: str,
) -> dict[str, Any]:
    index = fetch(client, filing.index)
    wall, documents = index_details(index) if index else (None, {})
    header = fetch(client, filing.header) if local_clock(wall) is None else None
    accepted = timing(
        wall,
        header_clock(header) if header else None,
        filing.submissions_clock,
        calendar,
    )
    record = {
        "cik": filing.cik,
        "accession": filing.accession,
        "form": filing.form,
        "filing_date": filing.filing_date,
        "report_date": filing.report_date,
        "items": ",".join(sorted(set(re.findall(r"\d\.\d{2}", filing.items)))),
        "index_url": filing.index,
        "document_url": None,
        "text": None,
        "sections": None,
        "text_sha256": None,
        **accepted,
    }
    if accepted["status"] != "ok":
        return record
    if not start <= accepted["accepted_local"][:10] <= end:
        return {**record, "status": "outside_acceptance_range"}
    name = documents.get(filing.form, filing.primary_document)
    url = (
        filing.base + name
        if SEC_FILENAME.fullmatch(name) and name not in {".", ".."}
        else None
    )
    content = fetch(client, url) if url else None
    if not content:
        url = filing.base + filing.accession + ".txt"
        content = fetch(client, url)
    record["document_url"] = url
    if not content:
        return {**record, "status": "no_primary_document"}
    sections = narrative_sections(content, filing.form, minimum)
    if not sections:
        return {**record, "status": "no_target_narrative"}
    text = "\n".join(sections.values())
    return {
        **record,
        "sections": json.dumps(sections, ensure_ascii=False),
        "text": text,
        "text_sha256": sha256(text.encode()).hexdigest(),
    }
