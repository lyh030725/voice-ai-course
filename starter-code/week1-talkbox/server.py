"""M1 "Talkbox" — server scaffold.

You edit exactly ONE function in this file: `answer()` (search for "YOUR HOMEWORK").
Everything else is provided plumbing: an HTTP endpoint that receives the mic
clip from the browser, hands it to `answer()`, and ships the reply audio back
with per-stage timing headers the browser page knows how to display.

Run it:

    cp .env.example .env    # then fill in your key, Grok model, TTS voice
    uvicorn server:app --reload --port 8000

Then open http://localhost:8000 — `localhost` is exempt from the browser's
HTTPS-only microphone rule, which is why we serve on it (Lecture 1, pitfalls).

Architecture note (Slide 10): browser mic -> HTTP POST -> this process is the
training-wheels version of: caller -> telephony provider -> WebSocket -> your
server. A later week swaps the POST for a real continuous stream; the shape of the
server stays the same.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("talkbox")

app = FastAPI(title="M1 Talkbox")


# --------------------------------------------------------------------------
# Provided: xAI client + config helpers.
#
# CHAT (Grok) is OpenAI-SDK compatible (Slide 13): one base URL, one key.
# NOTE: xAI's STT and TTS are NOT OpenAI-SDK compatible — they are native
# xAI REST endpoints (/v1/stt, /v1/tts) you call directly (see answer()).
#
# Config is validated LAZILY so that echo mode works with zero setup:
# you only need a key once you start writing the cascade.
# --------------------------------------------------------------------------

def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in. "
            f"Model names live at https://docs.x.ai — check the live models page, "
            f"don't guess (Slide 13)."
        )
    return value


def xai_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"),
        api_key=require_env("XAI_API_KEY"),
    )


# --------------------------------------------------------------------------
# Provided: per-stage stopwatch.
#
# Slide 14 makes instrumentation part of the milestone: "log the wall-clock
# time of each stage — STT, Grok, TTS — per request." Wrap each stage in
# `with timer.stage("stt"): ...` and this logs it AND sends it to the browser,
# which draws your personal version of Slide 5 (the latency budget).
# --------------------------------------------------------------------------

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
    """What `answer()` returns. transcript/reply_text are optional but showing
    them in the browser makes debugging dramatically easier — 'transcripts for
    free' is the cascade's superpower (Slide 6)."""

    audio: bytes
    mime: str
    transcript: str | None = None   # what STT heard
    reply_text: str | None = None   # what Grok said


# --------------------------------------------------------------------------
# YOUR HOMEWORK: implement the cascade inside answer().
#
# The recommended path (Slide 14):
#
#   Step 0 — ship the echo. The function below already echoes the clip back.
#            Run the server, hold the button, say something, hear yourself.
#            This proves mic -> server -> speaker before any API is involved
#            (~10 minutes, de-risks all the plumbing).
#
#   Step 1 — STT. Send the clip to xAI speech-to-text, get a transcript.
#
#   Step 2 — Grok. Send the transcript (plus a short system prompt) to
#            chat completions, get a reply.
#
#   Step 3 — TTS. Send the reply text to xAI text-to-speech, get audio,
#            return it instead of the echo.
#
# A skeleton of steps 1-3 is sketched in comments below. Notes:
#
#  * Grok (chat) is the ONLY OpenAI-SDK call. xAI's STT and TTS are native
#    REST endpoints (/v1/stt, /v1/tts) — the OpenAI `client.audio.*` methods
#    hit /v1/audio/*, which xAI does not serve (you'll get a 404). Use
#    `requests` for those two; see solution/server.py for the full reference.
#
#  * The three calls run SEQUENTIALLY and BLOCK. That is fine — encouraged,
#    even — this week (Lecture 1, pitfalls: "resist optimizing"). Weeks 2-4
#    stream and overlap these stages; week 5 makes blocking the event loop
#    a firing offense. First make it work, then make it fast — with data.
#
#  * The browser sends whatever compressed format MediaRecorder produced
#    (usually audio/webm with Opus; Safari sends audio/mp4). Check the STT
#    docs page for accepted formats before assuming it takes anything.
#
#  * Keep the system prompt SHORT and tell Grok it is speaking out loud:
#    answers get read by TTS, so two sentences beat two paragraphs.
# ------------=--------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Kingo Voice TA, a friendly voice learning assistant for "
    "Sungkyunkwan University students. Reply in the language the student "
    "uses, including Korean for Korean questions. Since every reply is read "
    "aloud , natural sentences without markdown, "
    "lists, headings, or code blocks. *For conceptual questions, do not "
    "immediately give a long answer; when useful, ask one brief guiding "
    "question that helps the student think*. *If the student clearly asks for "
    "an explanation or does not know the answer, explain the core concept "
    "concisely*. Never claim to have checked lecture notes, professor "
    "materials, or other sources that were not provided. Do not imply that "
    "you have RAG or uploaded course materials. make every response easy to speak aloud."
)


async def answer(audio: bytes, mime: str, timer: StageTimer) -> AgentReply:
    # ---- Steps 1-3: the cascade (implement STT -> Grok -> TTS) -------------
    api_key = require_env("XAI_API_KEY")
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    with timer.stage("stt"):
        stt_response = requests.post(
            "https://api.x.ai/v1/stt",
            headers=auth_headers,
            files={
                "file": (
                    f"audio.{ext_for(mime)}",
                    audio,
                    mime,
                )
            },
        )
        stt_response.raise_for_status()
        transcript = (stt_response.json().get("text") or "").strip()

    if not transcript:
        raise RuntimeError(
            "음성을 인식하지 못했습니다. 버튼을 누른 채 다시 말씀해 주세요."
        )
    log.info("transcript: %s", transcript)

    with timer.stage("llm"):
        completion = xai_client().chat.completions.create(
            model=require_env("CHAT_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
        )
        reply_text = (completion.choices[0].message.content or "").strip()
        if not reply_text:
            reply_text = "죄송해요. 답변을 만들지 못했어요. 질문을 다시 말씀해 주세요."

    with timer.stage("tts"):
        tts_response = requests.post(
            "https://api.x.ai/v1/tts",
            headers=auth_headers,
            json={
                "text": reply_text,
                "voice_id": require_env("TTS_VOICE"),
                "language": "auto",
            },
        )
        tts_response.raise_for_status()
        reply_audio = tts_response.content

    return AgentReply(
        audio=reply_audio,
        mime="audio/mpeg",
        transcript=transcript,
        reply_text=reply_text,
    )


def ext_for(mime: str) -> str:
    """'audio/webm;codecs=opus' -> 'webm', 'audio/mp4' -> 'mp4', etc."""
    return mime.split(";")[0].split("/")[-1] or "webm"


# --------------------------------------------------------------------------
# Provided: the HTTP plumbing. You should read this (it's short) but you
# don't need to change it.
# --------------------------------------------------------------------------

@app.post("/answer")
async def answer_endpoint(request: Request) -> Response:
    audio = await request.body()
    mime = request.headers.get("content-type", "audio/webm")
    log.info("clip received: %d bytes, %s", len(audio), mime)

    timer = StageTimer()
    with timer.stage("total"):
        try:
            reply = await answer(audio, mime, timer)
        except Exception as exc:  # surface errors in the browser, not just the terminal
            log.exception("answer() failed")
            return Response(
                content=str(exc), status_code=500, media_type="text/plain"
            )

    # Timing + text ride back as headers so the page can display them.
    # json.dumps escapes non-ASCII, which HTTP headers require.
    headers = {"X-Timings": json.dumps(timer.timings_ms)}
    if reply.transcript is not None:
        headers["X-Transcript"] = json.dumps(reply.transcript)
    if reply.reply_text is not None:
        headers["X-Reply"] = json.dumps(reply.reply_text)

    return Response(content=reply.audio, media_type=reply.mime, headers=headers)


# Serve static/index.html at / — mounted last so /answer wins the route match.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
