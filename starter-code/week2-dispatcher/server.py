"""M2 KINGO VOICE TA — tool-calling memory and source dispatcher.

The single-student agent automatically stores weak concepts, recalls them on
later turns, searches local course PDFs first, and uses trusted web search only
when the course material is insufficient. External answers are accepted only
when source URLs are present and are also written to an audit log.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pypdf import PdfReader

if os.environ.get("VOICE_AI_SKIP_DOTENV") != "1":
    load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dispatcher")

app = FastAPI(title="M2 KINGO VOICE TA Dispatcher")

BASE_DIR = Path(__file__).resolve().parent
WEAK_CONCEPTS_FILE = BASE_DIR / "memory" / "weak_concepts.jsonl"
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
MAX_TOOL_ROUNDS = 8


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


@dataclass
class AgentReply:
    audio: bytes
    mime: str
    transcript: str | None = None
    reply_text: str | None = None
    tools_used: list[str] = field(default_factory=list)


HISTORY: list[dict] = []

SYSTEM_PROMPT = (
    "You are KINGO VOICE TA, a Socratic voice teaching assistant for one "
    "Sungkyunkwan University student. Speak in Korean unless asked otherwise. "
    "Use one to three short conversational sentences with no markdown lists. "
    "For every substantive course question, first call recall_weak_concepts "
    "with the current topic, then call search_course_materials before answering. "
    "If the student asks a course question, expresses confusion, or gives an "
    "incorrect or incomplete explanation, automatically call save_weak_concept; "
    "do not ask for permission. Avoid duplicate saves when the tool says it is "
    "already saved. Use search_trusted_web ONLY after search_course_materials "
    "returns found=false or its excerpts clearly do not contain enough evidence. "
    "Never answer a factual course question from memory alone. When local PDF "
    "results are used, state the PDF filename and page. When external web results "
    "are used, always include at least one returned source URL in the visible "
    "answer and say that it is an external source. If a tool returns an error or "
    "no trustworthy sources, explain that you could not verify the answer rather "
    "than guessing. Use recalled weak concepts to adjust hints and follow-up "
    "questions, but do not reveal a complete answer immediately."
)


# --------------------------------------------------------------------------
# HOMEWORK 1 — the tool schemas.
# --------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "save_weak_concept",
            "description": (
                "Automatically save a student's weak concept. Call after every "
                "substantive course question, explicit confusion, or incorrect/"
                "incomplete explanation. Student consent is NOT required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "course": {"type": "string", "description": "Course name; use '미지정 과목' if unknown."},
                    "concept": {"type": "string", "description": "Specific concept that needs reinforcement."},
                    "original_question": {"type": "string", "description": "The student's current question in concise form."},
                    "difficulty_note": {"type": "string", "description": "Observed misunderstanding or knowledge gap."},
                },
                "required": ["course", "concept", "original_question", "difficulty_note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_weak_concepts",
            "description": (
                "Recall previously stored weak concepts relevant to the current "
                "topic. Call BEFORE answering every substantive course question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Current concept or question used to find relevant memories."},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_course_materials",
            "description": (
                "Search srcs/*.pdf and return page excerpts with filename/page "
                "citations. Call BEFORE answering any factual course question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused course-material search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trusted_web",
            "description": (
                "Search only configured authoritative academic/official sites. "
                "Call ONLY after search_course_materials returned found=false or "
                "insufficient evidence. Results without source URLs are rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused factual query not answered by the PDFs."},
                    "pdf_evidence_insufficient": {
                        "type": "boolean",
                        "description": (
                            "Set true only when the previous PDF search returned "
                            "results but those excerpts do not answer the question."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Briefly state why the previous PDF search had no "
                            "usable evidence or why its excerpts were insufficient."
                        ),
                    },
                },
                "required": ["query", "pdf_evidence_insufficient", "reason"],
            },
        },
    },
]


# --------------------------------------------------------------------------
# HOMEWORK 2 — the tool implementations. Fast, terse, validated.
# --------------------------------------------------------------------------

def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping malformed JSONL row in %s", path)
    return rows


def new_memory_id() -> str:
    return f"M-{1000 + len(_read_jsonl(WEAK_CONCEPTS_FILE))}"


def save_weak_concept(
    course: str,
    concept: str,
    original_question: str,
    difficulty_note: str,
) -> str:
    """Persist one automatically detected weak concept for the single student."""
    fields = {
        "course": course.strip(),
        "concept": concept.strip(),
        "original_question": original_question.strip(),
        "difficulty_note": difficulty_note.strip(),
    }
    if not all(fields.values()):
        return _json({"error": "all weak-concept fields are required"})

    normalized = (
        fields["concept"].casefold(),
        fields["original_question"].casefold(),
    )
    for memory in _read_jsonl(WEAK_CONCEPTS_FILE):
        existing = (
            str(memory.get("concept", "")).casefold(),
            str(memory.get("original_question", "")).casefold(),
        )
        if existing == normalized:
            return _json(
                {
                    "memory_id": memory["id"],
                    "status": "already_saved",
                    "concept": memory["concept"],
                }
            )

    memory = {
        "id": new_memory_id(),
        **fields,
        "saved_at": time.time(),
    }
    WEAK_CONCEPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with WEAK_CONCEPTS_FILE.open("a", encoding="utf-8") as file:
        file.write(_json(memory) + "\n")
    log.info("weak concept saved: %s", memory)
    return _json(
        {
            "memory_id": memory["id"],
            "status": "saved",
            "concept": memory["concept"],
        }
    )


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-z가-힣_]{2,}", text)
        if token.casefold() not in {"대해서", "설명", "질문", "무엇", "어떻게"}
    }


def recall_weak_concepts(topic: str) -> str:
    """Return relevant saved memories, falling back to the most recent ones."""
    topic = topic.strip()
    if not topic:
        return _json({"error": "topic is required"})
    memories = _read_jsonl(WEAK_CONCEPTS_FILE)
    query_terms = _terms(topic)

    ranked = []
    for index, memory in enumerate(memories):
        searchable = " ".join(
            str(memory.get(key, ""))
            for key in ("course", "concept", "original_question", "difficulty_note")
        ).casefold()
        score = sum(searchable.count(term) for term in query_terms)
        ranked.append((score, index, memory))

    matched = [item for item in ranked if item[0] > 0]
    selected = sorted(matched or ranked, key=lambda item: (item[0], item[1]), reverse=True)[:5]
    return _json(
        {
            "found": bool(selected),
            "topic": topic,
            "memories": [
                {
                    "memory_id": memory["id"],
                    "course": memory["course"],
                    "concept": memory["concept"],
                    "original_question": memory["original_question"],
                    "difficulty_note": memory["difficulty_note"],
                }
                for _, _, memory in selected
            ],
        }
    )


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
        model=os.environ.get("WEB_SEARCH_MODEL", "grok-4.5"),
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


def run_tool(name: str, args: dict) -> str:
    log.info("tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False))
    try:
        if name == "save_weak_concept":
            return save_weak_concept(**args)
        if name == "recall_weak_concepts":
            return recall_weak_concepts(**args)
        if name == "search_course_materials":
            return search_course_materials(**args)
        if name == "search_trusted_web":
            return search_trusted_web(**args)
        return _json({"error": f"unknown tool: {name}"})
    except Exception as exc:
        log.exception("tool %s failed", name)
        return _json({"error": str(exc)})


# --------------------------------------------------------------------------
# HOMEWORK 3 — the tool-call loop.
# --------------------------------------------------------------------------

async def think(transcript: str, timer: StageTimer) -> tuple[str, list[str]]:
    HISTORY.append({"role": "user", "content": transcript})
    client = xai_client()
    tools_used: list[str] = []
    external_sources: list[str] = []
    pdf_search_performed = False
    pdf_results_found: bool | None = None

    for _round in range(MAX_TOOL_ROUNDS):
        msg = client.chat.completions.create(
            model=os.environ.get("CHAT_MODEL", "grok-4"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *HISTORY],
            tools=TOOLS,
        ).choices[0].message

        if not msg.tool_calls:
            reply_text = msg.content or "죄송해요. 잠시 생각의 흐름을 놓쳤어요."
            if external_sources and not any(url in reply_text for url in external_sources):
                reply_text += " 외부 출처: " + ", ".join(external_sources[:3])
            HISTORY.append({"role": "assistant", "content": reply_text})
            return reply_text, tools_used

        HISTORY.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                if not isinstance(args, dict):
                    raise TypeError("tool arguments must decode to an object")
            except (json.JSONDecodeError, TypeError) as exc:
                result = _json({"error": f"invalid tool arguments: {exc}"})
            else:
                if tc.function.name == "search_trusted_web":
                    claims_insufficient = args.get("pdf_evidence_insufficient") is True
                    if not pdf_search_performed:
                        result = _json({
                            "error": (
                                "search_course_materials must run in an earlier "
                                "tool round before trusted web search"
                            )
                        })
                    elif pdf_results_found and not claims_insufficient:
                        result = _json({
                            "error": (
                                "PDF results were found. Set "
                                "pdf_evidence_insufficient=true and explain "
                                "why those excerpts cannot answer the question."
                            )
                        })
                    else:
                        result = run_tool(tc.function.name, args)
                else:
                    result = run_tool(tc.function.name, args)

            if tc.function.name == "search_course_materials":
                pdf_search_performed = True
                try:
                    parsed_result = json.loads(result)
                    found = parsed_result.get("found")
                    pdf_results_found = found if isinstance(found, bool) else None
                except json.JSONDecodeError:
                    pdf_results_found = None
            if tc.function.name == "search_trusted_web":
                try:
                    external_sources.extend(json.loads(result).get("sources", []))
                except json.JSONDecodeError:
                    pass

            tools_used.append(tc.function.name)
            HISTORY.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    reply_text = "도구 사용이 반복되어 처리를 마치지 못했어요. 다시 말씀해 주세요."
    HISTORY.append({"role": "assistant", "content": reply_text})
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


@app.post("/reset")
async def reset() -> dict:
    HISTORY.clear()
    log.info("conversation reset")
    return {"ok": True}


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
