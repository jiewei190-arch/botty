from __future__ import annotations

import json

import pytest

from trading_bot.config.settings import ResearchSettings
from trading_bot.research.stock_analyst import ResearchError, _parse_response, analyze_stock


def _payload() -> dict:
    return {
        "steps": [
            {
                "type": "google_search_call",
                "arguments": {"queries": ["Tesla latest investor relations"]},
            },
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "VERDICT: SIT OUT\nCONFIDENCE: 61%\n\n### Bottom line\nWait.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "Tesla Investor Relations",
                                "url": "https://ir.tesla.com/",
                            },
                            {
                                "type": "url_citation",
                                "title": "duplicate",
                                "url": "https://ir.tesla.com/",
                            },
                        ],
                    }
                ],
            },
        ]
    }


def test_parse_grounded_response():
    report = _parse_response(_payload(), "Tesla")
    assert report.verdict == "SIT OUT"
    assert report.search_queries == ("Tesla latest investor relations",)
    assert len(report.sources) == 1
    assert report.sources[0].url == "https://ir.tesla.com/"


def test_missing_verdict_fails_closed_to_sit_out():
    payload = _payload()
    payload["steps"][1]["content"][0]["text"] = "Ambiguous evidence."
    assert _parse_response(payload, "TSLA").verdict == "SIT OUT"


@pytest.mark.parametrize("verdict", ["LONG CALL SWING", "LONG PUT SWING", "SIT OUT"])
def test_options_swing_verdicts(verdict):
    payload = _payload()
    payload["steps"][1]["content"][0]["text"] = f"VERDICT: {verdict}"
    payload["steps"][1]["content"][0]["annotations"].extend(
        [
            {"type": "url_citation", "title": "SEC", "url": "https://sec.gov/a"},
            {"type": "url_citation", "title": "News", "url": "https://example.com/b"},
        ]
    )
    assert _parse_response(payload, "TSLA").verdict == verdict


def test_fewer_than_three_sources_forces_sit_out():
    payload = _payload()
    payload["steps"][1]["content"][0]["text"] = "VERDICT: LONG CALL SWING"
    assert _parse_response(payload, "TSLA").verdict == "SIT OUT"


def test_empty_response_is_rejected():
    with pytest.raises(ResearchError, match="no analysis"):
        _parse_response({"steps": []}, "TSLA")


def test_analysis_requires_key():
    with pytest.raises(ResearchError, match="not configured"):
        analyze_stock("TSLA", ResearchSettings())


def test_analysis_calls_grounded_interactions_api(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(_payload()).encode()

    def fake_open(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    settings = ResearchSettings(gemini_api_key="secret", model="gemini-test")
    report = analyze_stock(" Tesla ", settings)

    body = json.loads(captured["request"].data)
    assert body["model"] == "gemini-test"
    assert body["tools"] == [{"type": "google_search"}]
    assert "7-to-60" in body["input"]
    assert "buying a call" in body["input"]
    assert "buying a put" in body["input"]
    assert captured["request"].headers["X-goog-api-key"] == "secret"
    assert report.verdict == "SIT OUT"
