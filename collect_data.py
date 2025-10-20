"""Модуль для збору даних з відкритих джерел.

Приклад використання
--------------------
>>> from collect_data import collect
>>> collect("AI in military education")
PosixPath('data/results.json')
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from dotenv import load_dotenv


@dataclass
class SourceResult:
    """Структурований опис знайденого джерела."""

    url: str
    summary: str
    relevance: float
    metadata: dict[str, object] = field(default_factory=dict)


class BaseSearchClient:
    """Базовий клас для клієнтів пошукових сервісів."""

    def search(self, query: str) -> List[SourceResult]:  # pragma: no cover - метод інтерфейсу
        raise NotImplementedError


class TavilyClient(BaseSearchClient):
    """Обгортка над Tavily API."""

    def __init__(self, api_key: Optional[str]) -> None:
        self.api_key = api_key
        self.endpoint = "https://api.tavily.com/search"

    def search(self, query: str) -> List[SourceResult]:
        if not self.api_key:
            return []
        payload = {"api_key": self.api_key, "query": query, "include_images": False}
        response = requests.post(self.endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("results", []):
            results.append(
                SourceResult(
                    url=item.get("url", ""),
                    summary=item.get("content", ""),
                    relevance=float(item.get("score", 0.0)),
                    metadata={"provider": "tavily"},
                )
            )
        return results


class GeminiResearchClient(BaseSearchClient):
    """Клієнт для Gemini Deep Research API.

    Формат відповіді узагальнений, оскільки офіційна документація недоступна.
    """

    def __init__(self, api_key: Optional[str]) -> None:
        self.api_key = api_key
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:research"
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-pro-exp")

    def search(self, query: str) -> List[SourceResult]:
        if not self.api_key:
            return []
        url = self.endpoint.format(model=self.model)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"query": query, "max_output_documents": 5}
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("documents", []):
            metadata = item.get("metadata", {})
            results.append(
                SourceResult(
                    url=metadata.get("source", ""),
                    summary=item.get("summary", ""),
                    relevance=float(metadata.get("confidence", 0.5)),
                    metadata={"provider": "gemini", "raw": metadata},
                )
            )
        return results


class GoogleADKClient(BaseSearchClient):
    """Спрощена інтеграція з Google ADK Search Tools."""

    def __init__(self, api_key: Optional[str], graph_path: Optional[str]) -> None:
        self.api_key = api_key
        self.graph_path = graph_path
        self.endpoint = "https://adk.googleapis.com/v1/search"

    def search(self, query: str) -> List[SourceResult]:
        if not self.api_key:
            return []
        graph = None
        if self.graph_path and Path(self.graph_path).exists():
            graph = json.loads(Path(self.graph_path).read_text(encoding="utf-8"))
        payload = {"query": query, "graph": graph}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("items", []):
            results.append(
                SourceResult(
                    url=item.get("link", ""),
                    summary=item.get("snippet", ""),
                    relevance=float(item.get("score", 0.5)),
                    metadata={"provider": "google_adk"},
                )
            )
        return results


class ExaClient(BaseSearchClient):
    """Fallback через Exa API."""

    def __init__(self, api_key: Optional[str]) -> None:
        self.api_key = api_key
        self.endpoint = "https://api.exa.ai/search"

    def search(self, query: str) -> List[SourceResult]:
        if not self.api_key:
            return []
        headers = {"x-api-key": self.api_key}
        payload = {"query": query, "numResults": 5}
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("results", []):
            results.append(
                SourceResult(
                    url=item.get("url", ""),
                    summary=item.get("text", ""),
                    relevance=float(item.get("score", 0.5)),
                    metadata={"provider": "exa"},
                )
            )
        return results


class DuckDuckGoClient(BaseSearchClient):
    """Безключовий резервний пошук через DuckDuckGo Instant Answer API."""

    def __init__(self, proxy: Optional[str] = None) -> None:
        self.proxy = proxy
        self.endpoint = "https://api.duckduckgo.com/"

    def search(self, query: str) -> List[SourceResult]:
        params = {"q": query, "format": "json", "no_html": 1, "no_redirect": 1}
        proxies = {"https": self.proxy, "http": self.proxy} if self.proxy else None
        response = requests.get(self.endpoint, params=params, proxies=proxies, timeout=15)
        response.raise_for_status()
        data = response.json()
        topics = data.get("RelatedTopics", [])
        results = []
        for item in topics:
            if "FirstURL" in item and "Text" in item:
                results.append(
                    SourceResult(
                        url=item["FirstURL"],
                        summary=item["Text"],
                        relevance=0.3,
                        metadata={"provider": "duckduckgo"},
                    )
                )
        if not results and data.get("AbstractURL"):
            results.append(
                SourceResult(
                    url=data.get("AbstractURL", ""),
                    summary=data.get("AbstractText", ""),
                    relevance=0.2,
                    metadata={"provider": "duckduckgo"},
                )
            )
        return results


class SearchOrchestrator:
    """Керує послідовним опитуванням доступних сервісів."""

    def __init__(self, clients: Iterable[BaseSearchClient]) -> None:
        self.clients = list(clients)

    def run(self, query: str) -> List[SourceResult]:
        aggregated: List[SourceResult] = []
        for client in self.clients:
            try:
                results = client.search(query)
            except requests.RequestException as exc:
                aggregated.append(
                    SourceResult(
                        url="",
                        summary=f"Помилка клієнта {client.__class__.__name__}: {exc}",
                        relevance=0.0,
                        metadata={"provider": client.__class__.__name__, "error": True},
                    )
                )
                continue
            aggregated.extend(results)
            time.sleep(0.3)
        unique = _deduplicate(aggregated)
        return sorted(unique, key=lambda item: item.relevance, reverse=True)


def _deduplicate(results: Iterable[SourceResult]) -> List[SourceResult]:
    seen: set[str] = set()
    deduped: List[SourceResult] = []
    for item in results:
        key = item.url or item.summary
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def collect(query: str, output_path: Optional[Path] = None) -> Path:
    """Зібрати інформацію за темою та записати до JSON."""

    load_dotenv()

    clients: List[BaseSearchClient] = [
        TavilyClient(os.getenv("TAVILY_API_KEY")),
        GeminiResearchClient(os.getenv("GEMINI_API_KEY")),
        GoogleADKClient(os.getenv("GOOGLE_ADK_API_KEY"), os.getenv("GOOGLE_ADK_SEARCH_GRAPH")),
        ExaClient(os.getenv("EXA_API_KEY")),
    ]
    clients.append(DuckDuckGoClient(os.getenv("DUCKDUCKGO_PROXY")))

    orchestrator = SearchOrchestrator(clients)
    results = orchestrator.run(query)
    keywords = _extract_keywords(results)

    payload = {
        "query": query,
        "sources": [
            {
                "url": item.url,
                "summary": item.summary,
                "relevance": item.relevance,
                "metadata": item.metadata,
            }
            for item in results
        ],
        "keywords": keywords,
    }

    output_dir = Path(output_path or Path("data/results.json"))
    if output_dir.is_dir():
        output_file = output_dir / "results.json"
    else:
        output_file = output_dir
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def _extract_keywords(results: Iterable[SourceResult], limit: int = 10) -> List[str]:
    bucket: dict[str, int] = {}
    for item in results:
        words = item.summary.split()
        for word in words:
            normalized = word.strip().lower().strip(",.?!")
            if len(normalized) < 4 or "://" in normalized:
                continue
            bucket[normalized] = bucket.get(normalized, 0) + 1
    sorted_words = sorted(bucket.items(), key=lambda pair: pair[1], reverse=True)
    return [word for word, _ in sorted_words[:limit]]


__all__ = ["collect", "SourceResult"]
