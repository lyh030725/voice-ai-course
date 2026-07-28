"""M2 KINGO VOICE TA — low-latency memory and source dispatcher.

The single-student agent automatically stores weak concepts, recalls them on
later turns, searches local course PDFs first, and uses trusted web search only
when the course material is insufficient. External answers are accepted only
when source URLs are present and are also written to an audit log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

from moss_memory import MossMemoryStore

if os.environ.get("VOICE_AI_SKIP_DOTENV") != "1":
    load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dispatcher")

MOSS_MEMORY = MossMemoryStore()
BACKGROUND_TASKS: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup_tasks = [asyncio.to_thread(_pdf_pages)]
    if MOSS_MEMORY.is_configured:
        startup_tasks.append(MOSS_MEMORY.initialize())
    else:
        log.warning(
            "Moss memory is not configured; set MOSS_PROJECT_ID and MOSS_PROJECT_KEY"
        )
    await asyncio.gather(*startup_tasks)
    try:
        yield
    finally:
        if BACKGROUND_TASKS:
            await asyncio.gather(*BACKGROUND_TASKS, return_exceptions=True)
        await MOSS_MEMORY.close()


app = FastAPI(title="M2 KINGO VOICE TA Dispatcher", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent
WEB_SEARCH_LOG = BASE_DIR / "sources" / "web-search.jsonl"
COURSE_SRCS_DIR = BASE_DIR.parents[1] / "srcs"
TRUSTED_WEB_DOMAINS = (
    "skku.edu",
    "arxiv.org",
    "aclanthology.org",
    "proceedings.neurips.cc",
    "jmlr.org",
)
PDF_MAX_RESULTS = 3
PDF_PAGE_CACHE: list[dict] | None = None
MAX_HISTORY_MESSAGES = 12


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in; "
            f"model names and voices live at https://docs.x.ai."
        )
    return value


def xai_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"),
        api_key=require_env("XAI_API_KEY"),
    )


class StageTimer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = round((time.perf_counter() - t0) * 1000)
            self.timings_ms[name] = ms
            log.info("stage %-5s %5d ms", name, ms)

    def record(self, name: str, started_at: float) -> None:
        ms = round((time.perf_counter() - started_at) * 1000)
        self.timings_ms[name] = ms
        log.info("stage %-5s %5d ms", name, ms)


@dataclass
class AgentReply:
    audio: bytes
    mime: str
    transcript: str | None = None
    reply_text: str | None = None
    tools_used: list[str] = field(default_factory=list)


class TextQuestion(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class WeakConceptCapture(BaseModel):
    course: str
    concept: str
    original_question: str
    difficulty_note: str


class WeakConceptReview(BaseModel):
    memory_id: str
    correct: bool


class AgentDecision(BaseModel):
    answer: str
    memory: WeakConceptCapture | None
    review: WeakConceptReview | None
    needs_web_search: bool
    web_search_query: str | None


HISTORY: list[dict] = []

SYSTEM_PROMPT = (
    "You are KINGO VOICE TA, a Socratic voice teaching assistant for one "
    "Sungkyunkwan University student. Speak in Korean unless asked otherwise. "
    "Use one to three short conversational sentences with no markdown lists. "
    "The application has already retrieved the student's relevant weak concepts "
    "and course evidence. Use only that supplied evidence for factual claims. "
    "When local PDF results are used, state the PDF filename and page. When web "
    "results are used, include at least one supplied source URL and say it is an "
    "external source. Never end an answer by saying the course material does not "
    "contain or verify an explanation when a trusted web search could answer it. "
    "Instead set needs_web_search=true and provide a focused web_search_query. "
    "Set needs_web_search=false after web evidence has been supplied. Use recalled "
    "weaknesses to personalize hints. If a weakness repeats, check a prerequisite "
    "before repeating the same explanation. Set memory only when the student says "
    "they are confused, gives an incorrect/incomplete explanation, or clearly fails "
    "to understand; ordinary learning questions are not weaknesses. Set review to "
    "the recalled memory_id and whether the student's explanation is correct when "
    "their message answers a recall question; otherwise set review=null. When review "
    "is non-null, set memory=null because review already updates that weakness."
)


# Retrieval and persistence helpers.

# --------------------------------------------------------------------------
# HOMEWORK 2 — the tool implementations. Fast, terse, validated.
# --------------------------------------------------------------------------

def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


async def save_weak_concept(
    course: str,
    concept: str,
    original_question: str,
    difficulty_note: str,
) -> str:
    """Embed one weak concept in the hot Moss session and queue cloud sync."""
    return _json(
        await MOSS_MEMORY.save(
            course=course,
            concept=concept,
            original_question=original_question,
            difficulty_note=difficulty_note,
        )
    )


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-z가-힣_]{2,}", text)
        if token.casefold() not in {"대해서", "설명", "질문", "무엇", "어떻게"}
    }


async def recall_weak_concepts(topic: str) -> str:
    """Return semantically relevant memories from the hot Moss index."""
    return _json(await MOSS_MEMORY.recall(topic))


def _pdf_pages() -> list[dict]:
    global PDF_PAGE_CACHE
    if PDF_PAGE_CACHE is not None:
        return PDF_PAGE_CACHE

    pages = []
    for pdf_path in sorted(COURSE_SRCS_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(
                        {
                            "file": pdf_path.name,
                            "page": page_number,
                            "text": re.sub(r"\s+", " ", text),
                        }
                    )
        except Exception:
            log.exception("failed to index PDF: %s", pdf_path)
    PDF_PAGE_CACHE = pages
    log.info("indexed %d PDF page(s) from %s", len(pages), COURSE_SRCS_DIR)
    return pages


def _excerpt(text: str, terms: set[str], limit: int = 1000) -> str:
    lowered = text.casefold()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - 250)
    return text[start : start + limit].strip()


def search_course_materials(query: str) -> str:
    """Search local PDFs with simple page-level lexical ranking."""
    query = query.strip()
    terms = _terms(query)
    if not terms:
        return _json({"error": "a focused PDF search query is required"})

    ranked = []
    for page in _pdf_pages():
        lowered = page["text"].casefold()
        score = sum(lowered.count(term) for term in terms)
        if score:
            ranked.append((score, page))
    ranked.sort(key=lambda item: item[0], reverse=True)

    results = [
        {
            "source": f"{page['file']} p.{page['page']}",
            "excerpt": _excerpt(page["text"], terms),
        }
        for _, page in ranked[:PDF_MAX_RESULTS]
    ]
    return _json(
        {
            "found": bool(results),
            "query": query,
            "results": results,
            "instruction": (
                "Use filename and page in the answer."
                if results
                else "No PDF evidence found; trusted web search is now allowed."
            ),
        }
    )


def _trusted_urls(response) -> list[str]:
    payload = response.model_dump() if hasattr(response, "model_dump") else {}
    candidates = set(re.findall(r"https?://[^\s\]\)\"']+", json.dumps(payload)))
    candidates.update(re.findall(r"https?://[^\s\]\)\"']+", response.output_text or ""))

    trusted = []
    for url in sorted(candidates):
        host = urlparse(url.rstrip(".,;")).hostname or ""
        if any(host == domain or host.endswith("." + domain) for domain in TRUSTED_WEB_DOMAINS):
            trusted.append(url.rstrip(".,;"))
    return trusted


def search_trusted_web(
    query: str,
    pdf_evidence_insufficient: bool = False,
    reason: str = "",
) -> str:
    """Use xAI server-side web search, require citations, and retain an audit log."""
    query = query.strip()
    reason = reason.strip()
    if not query:
        return _json({"error": "web search query is required"})
    if not reason:
        return _json({"error": "PDF fallback reason is required"})

    response = xai_client().responses.create(
        model=os.environ.get("WEB_SEARCH_MODEL", "grok-4.3"),
        input=[
            {
                "role": "system",
                "content": (
                    "Answer in Korean using only the web-search evidence. "
                    "Include inline source URLs. If evidence is insufficient, say so."
                ),
            },
            {"role": "user", "content": query},
        ],
        tools=[
            {
                "type": "web_search",
                "filters": {"allowed_domains": list(TRUSTED_WEB_DOMAINS)},
            }
        ],
    )
    answer = (response.output_text or "").strip()
    sources = _trusted_urls(response)
    if not answer or not sources:
        return _json(
            {
                "error": "trusted web search returned no citable sources",
                "query": query,
            }
        )

    record = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "pdf_evidence_insufficient": pdf_evidence_insufficient,
        "fallback_reason": reason,
        "searched_at": time.time(),
    }
    WEB_SEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WEB_SEARCH_LOG.open("a", encoding="utf-8") as file:
        file.write(_json(record) + "\n")
    return _json(
        {
            "found": True,
            "answer": answer,
            "sources": sources,
            "instruction": "The final answer must include at least one source URL.",
        }
    )


# --------------------------------------------------------------------------
# One-pass retrieval and response pipeline.
# --------------------------------------------------------------------------

CONFUSION_MARKERS = (
    "모르겠", "모르겠어", "잘 모르", "어려워", "어렵", "헷갈",
    "이해가 안", "이해 안", "이해되지", "감이 안", "막혀", "틀린 것 같",
    "don't know", "do not know", "confused", "difficult", "hard to understand",
)


def _explicit_confusion(transcript: str) -> bool:
    normalized = transcript.casefold()
    return any(marker in normalized for marker in CONFUSION_MARKERS)


def _fallback_weak_concept(transcript: str) -> WeakConceptCapture:
    concise_question = re.sub(r"\s+", " ", transcript).strip()
    return WeakConceptCapture(
        course="미지정 과목",
        concept=concise_question[:160],
        original_question=concise_question,
        difficulty_note="학생이 명시적으로 이해 부족, 혼란 또는 어려움을 표현함",
    )


def _append_history(message: dict) -> None:
    HISTORY.append(message)
    if len(HISTORY) > MAX_HISTORY_MESSAGES:
        del HISTORY[:-MAX_HISTORY_MESSAGES]


def _background_done(task: asyncio.Task) -> None:
    BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        log.exception("background weak-concept save failed")


def _schedule_memory_save(memory: WeakConceptCapture) -> None:
    task = asyncio.create_task(save_weak_concept(**memory.model_dump()))
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(_background_done)


def _schedule_memory_review(review: WeakConceptReview) -> None:
    task = asyncio.create_task(MOSS_MEMORY.review(review.memory_id, review.correct))
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(_background_done)


async def _retrieve_context(transcript: str, timer: StageTimer) -> tuple[dict, dict]:
    async def recall() -> dict:
        started_at = time.perf_counter()
        try:
            return json.loads(await recall_weak_concepts(transcript))
        except Exception as exc:
            log.exception("Moss recall failed")
            return {"error": str(exc), "found": False, "memories": []}
        finally:
            timer.record("recall", started_at)

    async def pdf_search() -> dict:
        started_at = time.perf_counter()
        try:
            result = await asyncio.to_thread(search_course_materials, transcript)
            return json.loads(result)
        except Exception as exc:
            log.exception("PDF search failed")
            return {"error": str(exc), "found": False, "results": []}
        finally:
            timer.record("pdf", started_at)

    return await asyncio.gather(recall(), pdf_search())


async def think(transcript: str, timer: StageTimer) -> tuple[str, list[str]]:
    _append_history({"role": "user", "content": transcript})
    memory_context, evidence = await _retrieve_context(transcript, timer)
    tools_used = ["recall_weak_concepts", "search_course_materials"]
    external_sources: list[str] = []
    has_web_evidence = False

    async def add_web_evidence(query: str, reason: str) -> None:
        nonlocal evidence, external_sources, has_web_evidence
        started_at = time.perf_counter()
        try:
            web_result = await asyncio.to_thread(
                search_trusted_web,
                query,
                bool(evidence.get("found")),
                reason,
            )
            web_evidence = json.loads(web_result)
            external_sources = web_evidence.get("sources", [])
            evidence = {"course_pdf": evidence, "trusted_web": web_evidence}
            has_web_evidence = web_evidence.get("found") is True
            if "search_trusted_web" not in tools_used:
                tools_used.append("search_trusted_web")
        except Exception as exc:
            log.exception("trusted web fallback failed")
            evidence = {
                "course_pdf": evidence,
                "trusted_web": {"error": str(exc), "found": False},
            }
        finally:
            timer.record("web", started_at)

    if evidence.get("found") is False:
        await add_web_evidence(
            transcript,
            "No relevant course PDF evidence was found.",
        )

    client = xai_client()

    def complete(retrieval_context: str) -> str:
        response = client.chat.completions.create(
            model=os.environ.get("CHAT_MODEL", "grok-4.3"),
            reasoning_effort=os.environ.get("CHAT_REASONING_EFFORT", "none"),
            max_completion_tokens=int(os.environ.get("CHAT_MAX_TOKENS", "500")),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *HISTORY,
                {
                    "role": "system",
                    "content": "Retrieved context for this turn:\n" + retrieval_context,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "kingo_agent_reply",
                    "strict": True,
                    "schema": AgentDecision.model_json_schema(),
                },
            },
        )
        return response.choices[0].message.content or ""

    async def generate_decision() -> AgentDecision:
        retrieval_context = _json({
            "recalled_weak_concepts": memory_context,
            "evidence": evidence,
        })
        started_at = time.perf_counter()
        raw_decision = await asyncio.to_thread(complete, retrieval_context)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        timer.timings_ms["grok"] = timer.timings_ms.get("grok", 0) + elapsed_ms
        log.info("stage grok  %5d ms", elapsed_ms)
        return AgentDecision.model_validate_json(raw_decision)

    decision = await generate_decision()
    captured_memory = decision.memory
    if decision.needs_web_search and not has_web_evidence:
        await add_web_evidence(
            decision.web_search_query or transcript,
            "The PDF excerpts did not fully explain the student's why/how question.",
        )
        decision = await generate_decision()

    reply_text = decision.answer.strip() or "답변을 생성하지 못했어요. 다시 질문해 주세요."
    if external_sources and not any(url in reply_text for url in external_sources):
        reply_text += " 외부 출처: " + ", ".join(external_sources[:3])

    memory = decision.memory or captured_memory
    if memory is None and _explicit_confusion(transcript):
        memory = _fallback_weak_concept(transcript)
        log.info("explicit confusion detected; using fallback weak-concept capture")
    if memory is not None:
        _schedule_memory_save(memory)
        tools_used.append("save_weak_concept")
    if decision.review is not None:
        _schedule_memory_review(decision.review)
        tools_used.append("review_weak_concept")

    _append_history({"role": "assistant", "content": reply_text})
    return reply_text, tools_used


# --------------------------------------------------------------------------
# Provided: the cascade (native xAI STT/TTS, Grok chat via the SDK).
# --------------------------------------------------------------------------

async def answer(audio: bytes, mime: str, timer: StageTimer) -> AgentReply:
    api_key = require_env("XAI_API_KEY")

    with timer.stage("stt"):
        resp = requests.post(
            "https://api.x.ai/v1/stt",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (f"clip.{ext_for(mime)}", audio, mime)},
        )
        resp.raise_for_status()
        transcript = resp.json()["text"]
    log.info("heard: %r", transcript)

    with timer.stage("llm"):
        reply_text, tools_used = await think(transcript, timer)
    log.info("reply: %r (tools: %s)", reply_text, tools_used or "none")

    with timer.stage("tts"):
        resp = requests.post(
            "https://api.x.ai/v1/tts",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "text": reply_text,
                "voice_id": require_env("TTS_VOICE"),
                "language": "auto",
            },
        )
        resp.raise_for_status()
        reply_audio = resp.content

    return AgentReply(
        audio=reply_audio,
        mime="audio/mpeg",
        transcript=transcript,
        reply_text=reply_text,
        tools_used=tools_used,
    )


def ext_for(mime: str) -> str:
    return mime.split(";")[0].split("/")[-1] or "webm"


@app.post("/answer")
async def answer_endpoint(request: Request) -> Response:
    audio = await request.body()
    mime = request.headers.get("content-type", "audio/webm")
    log.info("clip received: %d bytes, %s", len(audio), mime)

    timer = StageTimer()
    with timer.stage("total"):
        try:
            reply = await answer(audio, mime, timer)
        except Exception as exc:
            log.exception("answer() failed")
            return Response(content=str(exc), status_code=500, media_type="text/plain")

    headers = {
        "X-Timings": json.dumps(timer.timings_ms),
        "X-Tools": json.dumps(reply.tools_used),
    }
    if reply.transcript is not None:
        headers["X-Transcript"] = json.dumps(reply.transcript)
    if reply.reply_text is not None:
        headers["X-Reply"] = json.dumps(reply.reply_text)

    return Response(content=reply.audio, media_type=reply.mime, headers=headers)


@app.post("/answer-text")
async def answer_text_endpoint(question: TextQuestion) -> dict:
    """Run the same agent/tool path without STT or TTS."""
    transcript = question.text.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="text must not be blank")

    log.info("text question received: %r", transcript)
    timer = StageTimer()
    try:
        with timer.stage("total"):
            with timer.stage("llm"):
                reply_text, tools_used = await think(transcript, timer)
    except Exception as exc:
        log.exception("text answer failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "transcript": transcript,
        "reply": reply_text,
        "tools": tools_used,
        "timings": timer.timings_ms,
    }


@app.post("/reset")
async def reset() -> dict:
    HISTORY.clear()
    log.info("conversation reset")
    return {"ok": True}


@app.get("/review")
async def review_prompt() -> dict:
    if not MOSS_MEMORY.is_configured:
        return {"due": False}
    memory = await MOSS_MEMORY.next_review()
    if memory is None:
        return {"due": False}
    question = f"{memory['concept']}을 자신의 말로 설명해 볼까요?"
    _append_history({
        "role": "assistant",
        "content": f"복습 질문 (memory_id={memory['id']}): {question}",
    })
    return {
        "due": True,
        "memory_id": memory["id"],
        "concept": memory["concept"],
        "question": question,
    }


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
