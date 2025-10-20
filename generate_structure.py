"""Модуль генерації структури презентації.

Приклад використання
--------------------
>>> from generate_structure import generate
>>> generate()
PosixPath('output/structure.json')
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

from collect_data import SourceResult


@dataclass
class Slide:
    num: int
    title: str
    text: str
    image_prompt: Optional[str] = None


class LLMProvider:
    """Абстракція для різних LLM-постачальників."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - інтерфейс
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], model: str) -> None:
        self.api_key = api_key
        self.model = model or "gemini-2.0-pro-exp"
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY не налаштовано")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini не повернув кандидатів")
        return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")


class GroqProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], model: str) -> None:
        self.api_key = api_key
        self.model = model or "llama3-70b-8192"
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY не налаштовано")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Відповідай JSON-структурою."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class OllamaProvider(LLMProvider):
    def __init__(self, model: str) -> None:
        self.model = model or "llama3"
        self.endpoint = "http://localhost:11434/api/chat"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Поверни лише JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = requests.post(self.endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


def select_provider() -> LLMProvider:
    load_dotenv()
    preferred = os.getenv("LLM_PROVIDER")
    if preferred == "groq":
        return GroqProvider(os.getenv("GROQ_API_KEY"), os.getenv("GROQ_MODEL"))
    if preferred == "ollama":
        return OllamaProvider(os.getenv("OLLAMA_MODEL"))
    return GeminiProvider(os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL"))


def generate(
    results_path: Path | str = Path("data/results.json"),
    output_path: Path | str = Path("output/structure.json"),
) -> Path:
    """Побудувати структуру презентації на основі зібраних результатів."""

    provider = select_provider()
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError("Спочатку потрібно виконати collect_data.collect")
    data = json.loads(results_file.read_text(encoding="utf-8"))
    sources = [
        SourceResult(
            url=item.get("url", ""),
            summary=item.get("summary", ""),
            relevance=float(item.get("relevance", 0.0)),
            metadata=item.get("metadata", {}),
        )
        for item in data.get("sources", [])
    ]

    presentation_type = _detect_presentation_type(data.get("keywords", []))
    prompt = _build_prompt(data["query"], sources, presentation_type)

    try:
        response_text = provider.generate(prompt)
        slides_payload = _parse_response(response_text)
    except Exception as exc:  # noqa: BLE001
        slides_payload = _fallback_structure(data["query"], sources, presentation_type, error=str(exc))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(slides_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def _build_prompt(query: str, sources: List[SourceResult], presentation_type: str) -> str:
    context_lines = [f"- {source.summary[:300]} ({source.url})" for source in sources[:10]]
    context = "\n".join(context_lines)
    return (
        "Згенеруй структуру презентації у форматі JSON зі списком slides.\n"
        "Кожен слайд має поля num, title, text, image_prompt.\n"
        "Текст українською мовою.\n"
        f"Тип презентації: {presentation_type}.\n"
        f"Тема: {query}.\n"
        f"Контекст:\n{context}\n"
    )


def _parse_response(response_text: str) -> Dict[str, List[Dict[str, object]]]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:  # noqa: F841
        raise ValueError("Відповідь не є валідним JSON")
    slides = payload.get("slides")
    if not isinstance(slides, list):
        raise ValueError("Очікується ключ 'slides'")
    return {"slides": slides}


def _fallback_structure(
    query: str,
    sources: List[SourceResult],
    presentation_type: str,
    error: Optional[str] = None,
) -> Dict[str, List[Dict[str, object]]]:
    bullets = [source.summary for source in sources[:3]] or ["Недостатньо даних"]
    slides = [
        {
            "num": 1,
            "title": f"Вступ: {query}",
            "text": bullets[0],
            "image_prompt": f"Інфографіка на тему {query}",
        },
        {
            "num": 2,
            "title": "Ключові висновки",
            "text": "\n".join(bullets),
            "image_prompt": f"Ключові тренди у форматі {presentation_type}",
        },
        {
            "num": 3,
            "title": "Рекомендації",
            "text": "Сформулюй 3-5 рекомендацій на основі зібраних фактів.",
            "image_prompt": f"Ілюстрація рекомендацій щодо {query}",
        },
    ]
    if error:
        slides.append(
            {
                "num": 4,
                "title": "Технічна примітка",
                "text": f"LLM провайдер повернув помилку: {error}",
                "image_prompt": None,
            }
        )
    return {"slides": slides}


def _detect_presentation_type(keywords: List[str]) -> str:
    training = {"training", "education", "course", "навч"}
    analytics = {"analysis", "trend", "data", "аналіти"}
    management = {"management", "strategy", "policy", "управл"}
    lower_keywords = {word.lower() for word in keywords}
    if lower_keywords & training:
        return "освітня"
    if lower_keywords & analytics:
        return "аналітична"
    if lower_keywords & management:
        return "управлінська"
    return "змішана"


__all__ = ["generate", "Slide"]
