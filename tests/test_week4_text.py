from io import StringIO
from types import SimpleNamespace

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
import requests

from ingen_qr.evaluation.week3_metrics import paired_newey_west
from ingen_qr.week4.baseline import MONTHS, annual_parameters
from ingen_qr.week4.common import require_baseline
from ingen_qr.week4.filings import (
    RateLimiter,
    SecDownloader,
    header_clock,
    index_details,
    narrative_sections,
    soup,
    timing,
)
from ingen_qr.week4.incremental import model_matrix
from ingen_qr.week4.paired import paired_monthly
from ingen_qr.week4.sentiment import (
    Scorers,
    dictionary,
    lm_scores,
    nlp_runtime,
    resolve_batch,
    resolve_device,
    sample_sentences,
)
from ingen_qr.week4.text_features import SENTIMENT_CHANGES, asof_features, supervised
from ingen_qr.week4.text_pipeline import (
    download_issuer,
    estimate_runtime,
    select_pilot,
)


def test_frozen_rf_selections():
    selected = annual_parameters()
    assert len(MONTHS) == 95
    assert list(selected) == list(range(2016, 2024))
    assert selected[2016]["n_estimators"] == 200
    assert selected[2023]["min_samples_leaf"] == 43


def test_gate_stops_before_network(tmp_path):
    with pytest.raises(RuntimeError, match="baseline command"):
        require_baseline(tmp_path)


@pytest.fixture
def calendar():
    return xcals.get_calendar("XNYS", start="2023-01-01", end="2024-02-01")


def test_premarket_filing_waits_until_next_session(calendar):
    result = timing("2023-06-30 08:00:00", None, "2023-06-30T12:00:00Z", calendar)
    assert result["accepted_utc"] == pd.Timestamp("2023-06-30 12:00:00Z")
    assert result["available_at"] == pd.Timestamp("2023-07-03 13:30:00Z")
    assert result["submissions_check"] == "match"


def test_holiday_and_clocks(calendar):
    result = timing(None, "2023-07-03 18:00:00", "2023-07-03T18:00:00Z", calendar)
    assert result["available_at"] == pd.Timestamp("2023-07-05 13:30:00Z")
    assert result["submissions_check"] == "mismatch"
    assert (
        timing(None, None, "2023-07-03T22:00:00Z", calendar)["status"]
        == "unconfirmed_acceptance"
    )
    assert (
        timing("2023-07-03 07:00:00", "2023-07-03 08:00:00", "", calendar)["status"]
        == "authoritative_clock_conflict"
    )
    assert (
        header_clock(
            b"<html><pre>&lt;ACCEPTANCE-DATETIME&gt;20230703180000</pre></html>"
        )
        == "2023-07-03 18:00:00"
    )
    assert (
        index_details(
            b'<div class="infoHead">Accepted</div><div class="info">2023-07-03 18:00:00</div>'
        )[0]
        == "2023-07-03 18:00:00"
    )


def test_sections_exclude_exhibits_and_keep_body():
    document = b"""
    <p>Item 1A. Risk Factors</p><p>3</p>
    <p>Item 7. Management's Discussion and Analysis</p><p>8</p>
    <p>Item 1A. Risk Factors</p><p>Supply issues could materially limit our ability to meet customer demand.</p>
    <p>Item 1B. Unresolved Staff Comments</p><p>This should never appear within the risk factors.</p>
    <p>Item 7. Management's Discussion and Analysis</p><p>Costs decreased across several areas because demand weakened during the quarter.</p>
    <p>Item 7A. Quantitative and Qualitative Disclosures</p><p>This should never appear within the financial text.</p>
    """
    result = narrative_sections(document, "10-K", 5)
    assert set(result) == {"risk_factors", "mdna"}
    assert "Supply issues" in result["risk_factors"]
    assert "Costs decreased" in result["mdna"]
    assert "never appear" not in str(result)
    legacy = b"""<DOCUMENT>\n<TYPE>8-K\n<TEXT>
    <p>Item 2.02. Results of Operations</p><p>The company reported lower production costs during the year.</p>
    <p>Item 9.01. Financial Statements and Exhibits</p><p>Exclude the exhibit details here for the current test.</p>
    </TEXT></DOCUMENT><DOCUMENT>\n<TYPE>EX-99\n<TEXT><p>Exclude this exhibit text too.</p></TEXT></DOCUMENT>"""
    assert narrative_sections(legacy, "8-K", 5) == {
        "current_report": "The company reported lower production costs during the year."
    }


def test_lm_scores_and_missing_dictionary(tmp_path):
    path = tmp_path / "dictionary.csv"
    with pytest.raises(FileNotFoundError, match="Loughran"):
        dictionary(path)
    path.write_text(
        StringIO(
            "Word,Positive,Negative,Uncertainty,Litigious\nP,1,0,0,0\nN,0,9,0,0\nU,0,0,1,0\nL,0,0,0,1\n"
        ).getvalue()
    )
    assert lm_scores("p n n u l", dictionary(path)) == {
        "lm_tone": -0.2,
        "lm_uncertainty": 0.2,
        "lm_litigious": 0.2,
    }


def test_paired_matches_week3_on_same_months():
    dates = pd.date_range("2020-01-01", periods=16, freq="MS")
    left = pd.Series(np.linspace(0, 1, 16), index=dates)
    right = pd.Series(np.linspace(0, 0.9, 16) + np.sin(np.arange(16)), index=dates)
    current = paired_monthly(left, right)
    table = (
        pd.concat(
            [
                left.rename("rank_ic").to_frame().assign(model="other"),
                right.rename("rank_ic").to_frame().assign(model="ols3"),
            ]
        )
        .rename_axis("date")
        .reset_index()
    )
    table["ic"] = table["rank_ic"]
    original = paired_newey_west(table, 6).query("metric == 'rank_ic'").iloc[0]
    assert current["nw_t"] == pytest.approx(original["nw_t"])
    assert current["mean_difference"] == pytest.approx(original["mean_difference"])
    assert current["paired_months"] == original["paired_months"]
    assert paired_monthly(left, right.iloc[1:])["paired_months"] == 15


def test_asof_and_head_ignore_future_labels_in_all_historical_rows():
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    features = pd.DataFrame({"date": dates, "ticker": ["A"] * 8, "cik": [1] * 8})
    features["forecast_at"] = features["date"].dt.tz_localize("UTC")
    docs = features[["date", "cik"]].assign(
        accession=list("abcdefgh"),
        report_date=[str(v.date()) for v in dates],
        form="8-K",
        items="2.02",
        lm_tone=np.arange(8, dtype=float),
        lm_uncertainty=0.0,
        lm_litigious=0.0,
        finbert_tone=0.0,
        embedding=[[float(i), 1.0] for i in range(8)],
    )
    docs["available_at"] = docs["date"].dt.tz_localize("UTC") + pd.Timedelta(hours=1)
    docs["accepted_utc"] = docs["available_at"] - pd.Timedelta(days=1)
    joined = asof_features(features, docs, 180)
    assert pd.isna(joined.loc[0, "accession_8k"])
    assert joined.loc[4, "accession_8k"] == "d"
    assert joined.loc[4, "previous_accession_8k"] == "c"
    assert joined.loc[4, "new_8k_count"] == 1
    meta = features[["date", "ticker"]].assign(
        label_excess_return=np.arange(8, dtype=float)
    )
    meta["label_end"] = meta["date"] + pd.Timedelta(days=15)
    meta["label_available_at"] = meta["label_end"].dt.tz_localize("UTC")
    meta.loc[2, "label_available_at"] = pd.Timestamp("2020-08-01", tz="UTC")
    settings = {
        "minimum_head_training_rows": 2,
        "head_embargo_months": 1,
        "embedding_ridge_alpha": 1.0,
    }
    scored, audit = supervised(joined, meta, docs, settings)
    mutated = meta.copy()
    mutated.loc[[2, 4], "label_excess_return"] = 1e6
    alternative, _ = supervised(joined, mutated, docs, settings)
    assert np.isfinite(scored.loc[5, "embedding_score_8k"])
    assert scored.loc[5, "embedding_score_8k"] == pytest.approx(
        alternative.loc[5, "embedding_score_8k"]
    )
    assert audit.query("filing_group == '8k'").iloc[5]["max_training_date"] == dates[3]
    assert audit.query("filing_group == 'periodic'")["training_rows"].eq(0).all()
    assert scored["embedding_score_periodic"].isna().all()
    assert (
        scored.loc[6, "embedding_score_8k"] != alternative.loc[6, "embedding_score_8k"]
    )


def test_pilot_has_20_issuers_for_36_entry_months():
    grid = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=36, freq="MS"), range(25)],
        names=["date", "cik"],
    ).to_frame(index=False)
    result = select_pilot(grid, "2021-01-01", "2023-12-31", 20)
    assert len(result) == 720
    assert result["cik"].nunique() == 20


def test_text_rf_uses_week2_monthly_ranks_and_raw_missing_flags():
    month = pd.to_datetime(["2020-01-02"] * 4 + ["2020-02-03"] * 4)
    base = pd.DataFrame({"base": np.arange(8)})
    sample = SimpleNamespace(
        meta=pd.DataFrame({"date": month, "ticker": list("ABCD") * 2}),
        matrix=base,
    )
    features = pd.DataFrame(
        {"score": [1.0, 3.0, np.nan, 1000.0, 100.0, -500.0, np.inf, 110.0]}
    )
    matrix = model_matrix(sample, features, ["score"])
    pd.testing.assert_series_equal(matrix["base"], base["base"])
    assert matrix["score"].tolist() == [-1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0]
    assert matrix["missing_score"].tolist() == [0, 0, 1, 0, 0, 0, 1, 0]


def test_periodic_change_survives_a_later_8k_in_the_same_month():
    months = pd.to_datetime(["2020-04-01", "2020-05-01"])
    rows = pd.DataFrame({"date": months, "cik": 1, "ticker": "A"})
    rows["forecast_at"] = rows["date"].dt.tz_localize("UTC")
    available = pd.to_datetime(
        ["2020-01-15", "2020-03-11", "2020-04-15", "2020-04-27"], utc=True
    )
    filings = pd.DataFrame(
        {
            "cik": 1,
            "accession": ["q1", "k1", "q2", "k2"],
            "available_at": available,
            "accepted_utc": available - pd.Timedelta(days=1),
            "report_date": ["2020-01-14", "2020-03-10", "2020-04-14", "2020-04-25"],
            "form": ["10-Q", "8-K", "10-Q", "8-K"],
            "items": ["", "2.02", "", "5.02"],
            "lm_tone": [0.1, -0.9, 0.5, -0.8],
            "lm_uncertainty": 0.0,
            "lm_litigious": 0.0,
            "finbert_tone": [0.1, 0.2, 0.3, 0.4],
            "embedding": [[0.0, 1.0]] * 4,
        }
    )
    output = asof_features(rows, filings, 180)
    assert output.loc[1, "accession_periodic"] == "q2"
    assert output.loc[1, "accession_8k"] == "k2"
    assert output.loc[1, "lm_tone_periodic"] == 0.5
    assert output.loc[1, "lm_tone_8k"] == -0.8
    assert output.loc[1, "lm_tone_change_periodic"] == pytest.approx(0.4)
    assert pd.isna(output.loc[1, "lm_tone_change_8k"])
    assert set(SENTIMENT_CHANGES) == {
        f"{metric}_{group}"
        for metric in ("lm_tone_change", "finbert_tone_change", "embedding_change")
        for group in ("periodic", "8k")
    }
    assert asof_features(rows, filings.iloc[:0], 180)["accession_periodic"].isna().all()


def test_safe_harbor_paragraphs_removed_before_lm_scoring():
    result = narrative_sections(
        b"""
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>Our new factory produced ten thousand units during the quarter.</p>
    <p>Cautionary Note Regarding Forward-Looking Statements</p>
    <p>This report contains certain forward-looking statements that involve uncertainty.</p>
    <p>These statements depend on numerous uncertainties and assumptions.</p>
    <p>We do not undertake any obligation to update any of these statements.</p>
    <p>Our only supplier of battery cells presents a measurable operational risk.</p>
    <p>We applied existing tax safe harbor provisions to calculate deductible costs.</p>
    <p>Item 7A. Quantitative and Qualitative Disclosures</p>
    """,
        "10-K",
        5,
    )
    assert result["mdna"] == (
        "Our new factory produced ten thousand units during the quarter.\n"
        "Our only supplier of battery cells presents a measurable operational risk.\n"
        "We applied existing tax safe harbor provisions to calculate deductible costs."
    )


def test_nlp_caps_both_scorers_and_lm_sees_full_text(monkeypatch):
    texts = [f"Sentence {i} identifies the latest shipment risks." for i in range(150)]
    sampled = sample_sentences(texts, 128)
    assert len(sampled) == 128
    assert sampled == sample_sentences(texts, 128)
    assert sampled == [s for s in texts if s in set(sampled)]
    removed = texts.index(next(s for s in texts if s not in sampled))
    texts[removed] = texts[removed].replace("risks", "uncertain")
    scorer = Scorers.__new__(Scorers)
    scorer.max_sentences = 128
    scorer.lexicon = {
        "positive": set(),
        "negative": set(),
        "uncertainty": {"UNCERTAIN"},
        "litigious": set(),
    }
    monkeypatch.setattr(
        scorer,
        "finbert_scores",
        lambda lines: {"finbert_tone": 0.0} if lines == sampled else {},
    )
    monkeypatch.setattr(
        scorer, "embeddings", lambda lines: [1.0] if lines == sampled else []
    )
    result = scorer.score(" ".join(texts))
    assert result["lm_uncertainty"] > 0.0
    assert "finbert_tone" in result
    assert result["embedding"] == [1.0]
    assert result["nlp_sampled_sentences"] == 128


@pytest.mark.parametrize(
    ("uncached", "http", "rate", "workers", "earlier", "sec_seconds"),
    [
        (100, 100, 8.0, 1, (0, 0.0, 0), 5000),  # serial: 0.25 s network per call
        (100, 100, 8.0, 4, (0, 0.0, 0), 2500),  # four workers hit the 8/s cap
        (100, 200, 8.0, 4, (0, 0.0, 0), 5000),  # each call also follows a redirect
        (100, 100, 2.0, 4, (0, 0.0, 0), 10000),  # the rate cap binds first
        (0, 0, 8.0, 1, (0, 0.0, 0), 5000),  # cached pilot uses the clock check
        (0, 0, 8.0, 1, (50, 25.0, 50), 10000),  # earlier first-download timing
        (0, 0, 8.0, 4, (50, 5.0, 150), 7500),  # earlier redirect ratio of three
    ],
)
def test_runtime_forecast_uses_measured_stage_times(
    uncached, http, rate, workers, earlier, sec_seconds
):
    issuers = pd.DataFrame(
        {
            "sec_calls": [200],
            "sec_uncached_calls": [uncached],
            "sec_http_requests": [http],
            "sec_download_seconds": [25.0 if uncached else 0.0],
        }
    )
    filings = pd.DataFrame({"finbert_seconds": [16.0], "embedding_seconds": [8.0]})
    result = estimate_runtime(
        issuers, filings, 60, 6000, (7, 4, 1.0), rate, workers, earlier
    )
    hours = result.set_index("stage")["estimated_full_hours"]
    assert hours.to_dict() == pytest.approx(
        {
            "SEC download": sec_seconds / 3600,
            "FinBERT": 1600 / 3600,
            "Sentence embeddings": 800 / 3600,
        }
    )


def test_limiter_does_not_burst_after_threads_resume_together(monkeypatch):
    """Every sleeping thread is held until one shared resume time, like a pause."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from ingen_qr.week4 import filings

    resume_at = time.monotonic() + 0.5

    def stalled_sleep(delay: float) -> None:
        time.sleep(max(delay, resume_at - time.monotonic()))

    monkeypatch.setattr(filings, "sleep", stalled_sleep)
    limiter = RateLimiter(8.0)
    barrier = Barrier(6)
    released: list[float] = []

    def worker() -> None:
        barrier.wait(timeout=5)
        for _ in range(2):
            limiter.wait()
            released.append(time.monotonic())

    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in [pool.submit(worker) for _ in range(6)]:
            future.result()
    gaps = np.diff(sorted(released))
    assert len(released) == 12
    assert gaps.min() >= 0.125 - 0.01
    # No one-second window may hold more than eight releases.
    times = np.array(sorted(released))
    assert max(int(((times >= t) & (times < t + 1.0)).sum()) for t in times) <= 8


def test_shared_limiter_spaces_requests_across_threads():
    from concurrent.futures import ThreadPoolExecutor
    from time import monotonic

    limiter = RateLimiter(8.0)
    starts: list[float] = []

    def worker() -> None:
        for _ in range(3):
            limiter.wait()
            starts.append(monotonic())

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(worker) for _ in range(4)]:
            future.result()
    gaps = np.diff(sorted(starts))
    assert len(starts) == 12
    assert gaps.min() >= 0.125 - 0.01
    with pytest.raises(ValueError):
        RateLimiter(9.0)


class FakeSecAdapter(requests.adapters.BaseAdapter):
    """Offline adapter: records every HTTP request and answers redirects first."""

    def __init__(self, redirects: dict[str, str] | None = None) -> None:
        super().__init__()
        self.redirects = redirects or {}
        self.sent: list[str] = []

    def send(self, request, **kwargs):
        self.sent.append(str(request.url))
        response = requests.Response()
        response.request = request
        response.url = str(request.url)
        response.raw = None
        response._content_consumed = True
        if request.url in self.redirects:
            response.status_code = 302
            response.headers["Location"] = self.redirects[str(request.url)]
            response._content = b""
        else:
            response.status_code = 200
            response._content = b"body"
        return response

    def close(self) -> None:
        return None


def offline_client(tmp_path, rate: float, redirects=None):
    client = SecDownloader(str(tmp_path), RateLimiter(rate), headers={})
    adapter = FakeSecAdapter(redirects)
    client.session.mount("https://", adapter)
    return client, adapter


def test_sec_downloader_counts_network_time_without_limiter_waits(tmp_path):
    client, adapter = offline_client(tmp_path, 2.0)
    for url in ("https://example.test/a", "https://example.test/b"):
        assert client.get(url) == b"body"
    assert client.get("https://example.test/a") == b"body"
    calls, misses, seconds = client.measurements()
    assert (calls, misses) == (3, 2)
    assert len(adapter.sent) == 2
    assert client.limiter_seconds >= 0.4
    assert seconds < 0.2


def test_followed_redirects_wait_on_the_shared_limiter(tmp_path):
    from time import monotonic

    redirects = {
        f"https://example.test/{i}": f"https://example.test/final/{i}" for i in range(4)
    }
    client, adapter = offline_client(tmp_path, 8.0, redirects)
    started = monotonic()
    for i in range(4):
        assert client.get(f"https://example.test/{i}") == b"body"
    elapsed = monotonic() - started
    assert len(adapter.sent) == 8
    assert client.session.requests_sent == 8
    assert client.http_requests == 8
    # Eight real HTTP requests at 8/s need at least seven full gaps.
    assert elapsed >= 7 * 0.125 - 0.02


def test_concurrent_writes_to_one_cache_key_do_not_collide(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)
    clients = [
        SecDownloader(str(tmp_path), RateLimiter(8.0), headers={}) for _ in range(2)
    ]

    def slow_fetch() -> bytes:
        barrier.wait(timeout=5)
        return b"same body"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result()
            for future in [
                pool.submit(client.cached, "http:https://example.test/x", slow_fetch)
                for client in clients
            ]
        ]
    assert results == [b"same body", b"same body"]
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
    assert len(list(tmp_path.glob("*.bin"))) == 1


def test_windows_sharing_conflict_on_an_existing_cache_file_is_accepted(
    tmp_path, monkeypatch
):
    """Simulate Windows: replacing a target another thread holds raises WinError 5."""
    from pathlib import Path

    client = SecDownloader(str(tmp_path), RateLimiter(8.0), headers={})
    key = "http:https://example.test/x"
    assert client.cached(key, lambda: b"same body") == b"same body"

    def locked(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", locked)
    assert client.cached(key, lambda: b"same body", refresh=True) == b"same body"
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
    assert len(list(tmp_path.glob("*.bin"))) == 1


def test_sharing_conflict_without_a_target_still_fails(tmp_path, monkeypatch):
    from pathlib import Path

    from ingen_qr.week4 import filings

    monkeypatch.setattr(filings, "sleep", lambda seconds: None)

    def locked(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", locked)
    client = SecDownloader(str(tmp_path), RateLimiter(8.0), headers={})
    with pytest.raises(PermissionError):
        client.cached("http:https://example.test/y", lambda: b"body")
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_download_issuer_writes_corpus_before_completion_marker(tmp_path, monkeypatch):
    from ingen_qr.week4 import text_pipeline

    order: list[str] = []
    real_parquet, real_json = text_pipeline.parquet, text_pipeline.write_json
    monkeypatch.setenv("SEC_USER_AGENT", "Test Researcher test@example.com")
    monkeypatch.setattr(
        text_pipeline, "filings_for", lambda client, cik, start, end: iter([1])
    )
    monkeypatch.setattr(
        text_pipeline,
        "retrieve",
        lambda *args: dict.fromkeys(text_pipeline.DOCUMENT_COLUMNS, None)
        | {"status": "no_target_narrative", "report_date": ""},
    )
    monkeypatch.setattr(
        text_pipeline,
        "parquet",
        lambda frame, path: (order.append("corpus"), real_parquet(frame, path)),
    )
    monkeypatch.setattr(
        text_pipeline,
        "write_json",
        lambda data, path: (order.append("timing"), real_json(data, path)),
    )
    settings = {"sec_max_requests_per_second": 8, "minimum_section_words": 40}
    doc, marker = tmp_path / "1_filings.parquet", tmp_path / "1_download.json"
    download_issuer(
        1,
        settings,
        str(tmp_path / "cache"),
        RateLimiter(8.0),
        None,
        "2021-01-01",
        "2021-12-31",
        doc,
        marker,
    )
    assert order == ["corpus", "timing"]
    assert len(pd.read_parquet(doc)) == 1 and marker.is_file()


def test_nlp_device_auto_uses_cuda_only_when_available():
    def fake_torch(available: bool) -> SimpleNamespace:
        cuda = SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda index: "NVIDIA GeForce RTX 5060 Ti",
        )
        return SimpleNamespace(cuda=cuda, __version__="2.7.1")

    assert resolve_device("auto", fake_torch(True)) == (
        "cuda",
        "NVIDIA GeForce RTX 5060 Ti",
    )
    assert resolve_device("auto", fake_torch(False)) == ("cpu", "CPU")
    with pytest.raises(RuntimeError, match="cannot see a CUDA GPU"):
        resolve_device("cuda", fake_torch(False))
    with pytest.raises(ValueError):
        resolve_device("tpu", fake_torch(True))
    assert resolve_batch("auto", "cuda") == 128
    assert resolve_batch("auto", "cpu") == 32
    assert resolve_batch(64, "cuda") == 64


@pytest.mark.parametrize(
    ("available", "expected"),
    [(True, ("cuda", "NVIDIA GeForce RTX 5060 Ti", 128)), (False, ("cpu", "CPU", 32))],
)
def test_nlp_runtime_resolves_auto_before_the_cache_key(
    monkeypatch, available, expected
):
    import sys

    cuda = SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda index: "NVIDIA GeForce RTX 5060 Ti",
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__="2.7.1")
    )
    settings = {"nlp_device": "auto", "nlp_batch_size": "auto"}
    assert nlp_runtime(settings) == expected


def test_ixbrl_xml_declaration_does_not_warn():
    import warnings

    content = (
        b'<?xml version="1.0" encoding="utf-8"?>'
        b"<html><body><p>Item 7. Management</p></body></html>"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed = soup(content)
    assert "Management" in parsed.get_text()
    assert not [w for w in caught if "XMLParsedAsHTML" in type(w.message).__name__]
