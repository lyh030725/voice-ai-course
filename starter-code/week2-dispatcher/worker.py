"""Optional follow-up worker for weak concepts stored in Moss.

The voice path embeds memories in a Moss session. This worker reads the same
cloud-backed index and creates local Socratic follow-up templates.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from moss_memory import MossMemoryStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")

BASE_DIR = Path(__file__).resolve().parent
MOSS_MEMORY = MossMemoryStore()
PROCESSED_FILE = BASE_DIR / "memory" / "worker-processed.txt"
FOLLOWUP_GUIDES_DIR = BASE_DIR / "memory" / "followups"
POLL_SECONDS = 2


def build_followup_guide(memory: dict) -> str:
    """Build a local Socratic prompt without sending student data externally."""
    return (
        f"과목: {memory['course']}\n"
        f"취약 개념: {memory['concept']}\n"
        f"이전 질문: {memory['original_question']}\n"
        f"관찰된 어려움: {memory['difficulty_note']}\n\n"
        "다음 대화에서 사용할 질문\n"
        f"1. {memory['concept']}을 자신의 말로 먼저 설명해 볼까요?\n"
        "2. 어떤 단계까지는 확실하고, 어느 단계부터 막히나요?\n"
        "3. 간단한 예에 이 개념을 적용하면 어떤 결과를 예상하나요?\n\n"
        "힌트: 이전 답을 바로 알려주지 말고, 학생이 막힌 단계부터 한 단계씩 확인합니다.\n"
    )


def save_followup_guide(memory: dict, body: str) -> Path:
    FOLLOWUP_GUIDES_DIR.mkdir(parents=True, exist_ok=True)
    path = FOLLOWUP_GUIDES_DIR / f"{memory['id']}.txt"
    path.write_text(body, encoding="utf-8")
    log.info("FOLLOW-UP GUIDE READY -> %s", path)
    return path


def load_processed() -> set[str]:
    if PROCESSED_FILE.exists():
        return set(PROCESSED_FILE.read_text(encoding="utf-8").split())
    return set()


def mark_processed(memory_id: str) -> None:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_FILE.open("a", encoding="utf-8") as file:
        file.write(memory_id + "\n")


async def pending_memories() -> list[dict]:
    processed = load_processed()
    memories = await MOSS_MEMORY.all_memories()
    return [memory for memory in memories if memory["id"] not in processed]


async def process_once() -> int:
    handled = 0
    for memory in await pending_memories():
        log.info(
            "new weak concept: %s (%s, %s)",
            memory["id"],
            memory["course"],
            memory["concept"],
        )
        try:
            body = build_followup_guide(memory)
            save_followup_guide(memory, body)
            mark_processed(memory["id"])
            handled += 1
        except Exception:
            log.exception("failed on %s — will retry next poll", memory["id"])
    return handled


async def run_worker(once: bool) -> None:
    if once:
        n = await process_once()
        log.info("done: %d weak concept(s) processed", n)
        return

    log.info("watching Moss index %s (Ctrl-C to stop)", MOSS_MEMORY.index_name)
    while True:
        await process_once()
        await asyncio.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="KINGO Moss weak-concept worker")
    parser.add_argument("--once", action="store_true", help="drain the backlog and exit")
    args = parser.parse_args()
    asyncio.run(run_worker(args.once))


if __name__ == "__main__":
    main()
