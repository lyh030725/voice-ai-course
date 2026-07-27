"""Low-latency weak-concept memory backed by a Moss local session."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


log = logging.getLogger("moss-memory")


def _load_sdk() -> SimpleNamespace:
    """Import Moss lazily so non-memory tools can still be tested in isolation."""
    try:
        from moss import DocumentInfo, GetDocumentsOptions, MossClient, QueryOptions
    except ImportError as exc:  # pragma: no cover - exercised only in misconfigured installs
        raise RuntimeError(
            "Moss SDK is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return SimpleNamespace(
        DocumentInfo=DocumentInfo,
        GetDocumentsOptions=GetDocumentsOptions,
        MossClient=MossClient,
        QueryOptions=QueryOptions,
    )


class MossMemoryStore:
    """Create-or-resume one Moss session and keep its hot index in process."""

    def __init__(
        self,
        project_id: str | None = None,
        project_key: str | None = None,
        *,
        index_name: str | None = None,
        model_id: str | None = None,
        student_id: str | None = None,
        sync_debounce_seconds: float | None = None,
        sdk_loader: Callable[[], Any] = _load_sdk,
    ) -> None:
        self.project_id = (project_id or os.environ.get("MOSS_PROJECT_ID", "")).strip()
        self.project_key = (project_key or os.environ.get("MOSS_PROJECT_KEY", "")).strip()
        self.index_name = (
            index_name
            or os.environ.get("MOSS_MEMORY_INDEX", "kingo-week2-weak-concepts")
        ).strip()
        self.model_id = (
            model_id or os.environ.get("MOSS_MEMORY_MODEL", "moss-minilm")
        ).strip()
        self.student_id = (
            student_id or os.environ.get("MOSS_STUDENT_ID", "default-student")
        ).strip()
        configured_debounce = os.environ.get("MOSS_SYNC_DEBOUNCE_SECONDS", "0.75")
        self.sync_debounce_seconds = (
            float(configured_debounce)
            if sync_debounce_seconds is None
            else sync_debounce_seconds
        )
        self._sdk_loader = sdk_loader
        self._sdk: Any = None
        self._client: Any = None
        self._session: Any = None
        self._initialize_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._push_task: asyncio.Task[None] | None = None
        self._dirty = False

    @property
    def is_configured(self) -> bool:
        return bool(self.project_id and self.project_key)

    def _require_credentials(self) -> None:
        missing = [
            name
            for name, value in (
                ("MOSS_PROJECT_ID", self.project_id),
                ("MOSS_PROJECT_KEY", self.project_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"{', '.join(missing)} is not set. Add it to your environment."
            )

    async def initialize(self) -> None:
        """Hydrate the cloud index once; later reads and writes stay in-process."""
        if self._session is not None:
            return
        async with self._initialize_lock:
            if self._session is not None:
                return
            self._require_credentials()
            self._sdk = self._sdk_loader()
            self._client = self._sdk.MossClient(self.project_id, self.project_key)
            self._session = await self._client.session(
                index_name=self.index_name,
                model_id=self.model_id,
            )
            log.info(
                "Moss memory ready: index=%s docs=%s model=%s",
                self.index_name,
                getattr(self._session, "doc_count", "?"),
                self.model_id,
            )

    def _memory_id(self, concept: str, original_question: str) -> str:
        canonical = "\0".join(
            (
                self.student_id.casefold(),
                concept.strip().casefold(),
                original_question.strip().casefold(),
            )
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"M-{digest}"

    @staticmethod
    def _searchable_text(memory: dict[str, Any]) -> str:
        return (
            f"과목: {memory['course']}\n"
            f"취약 개념: {memory['concept']}\n"
            f"학생 질문: {memory['original_question']}\n"
            f"관찰된 어려움: {memory['difficulty_note']}"
        )

    @staticmethod
    def _memory_from_doc(doc: Any) -> dict[str, Any] | None:
        payload = getattr(doc, "payload", None)
        if not payload:
            return None
        try:
            memory = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            log.warning("ignoring Moss document with malformed payload: %s", doc.id)
            return None
        if not isinstance(memory, dict):
            return None
        return memory

    async def save(
        self,
        course: str,
        concept: str,
        original_question: str,
        difficulty_note: str,
    ) -> dict[str, Any]:
        fields = {
            "course": course.strip(),
            "concept": concept.strip(),
            "original_question": original_question.strip(),
            "difficulty_note": difficulty_note.strip(),
        }
        if not all(fields.values()):
            return {"error": "all weak-concept fields are required"}

        await self.initialize()
        memory_id = self._memory_id(fields["concept"], fields["original_question"])
        async with self._operation_lock:
            existing = await self._session.get_docs(
                self._sdk.GetDocumentsOptions(doc_ids=[memory_id])
            )
            if existing:
                return {
                    "memory_id": memory_id,
                    "status": "already_saved",
                    "concept": fields["concept"],
                }

            memory = {
                "id": memory_id,
                "student_id": self.student_id,
                **fields,
                "saved_at": time.time(),
            }
            document = self._sdk.DocumentInfo(
                id=memory_id,
                text=self._searchable_text(memory),
                metadata={
                    "student_id": self.student_id,
                    "course": fields["course"],
                    "concept": fields["concept"],
                },
                payload=json.dumps(memory, ensure_ascii=False),
            )
            await self._session.add_docs([document])
            self._dirty = True

        self._schedule_push()
        log.info("weak concept embedded in Moss: %s", memory_id)
        return {
            "memory_id": memory_id,
            "status": "saved",
            "concept": fields["concept"],
        }

    async def recall(self, topic: str, *, top_k: int = 5) -> dict[str, Any]:
        topic = topic.strip()
        if not topic:
            return {"error": "topic is required"}
        await self.initialize()
        options = self._sdk.QueryOptions(
            top_k=top_k,
            alpha=0.85,
            filter={
                "field": "student_id",
                "condition": {"$eq": self.student_id},
            },
        )
        async with self._operation_lock:
            result = await self._session.query(topic, options)
        memories = [
            memory
            for doc in result.docs
            if (memory := self._memory_from_doc(doc)) is not None
        ]
        return {
            "found": bool(memories),
            "topic": topic,
            "memories": [
                {
                    "memory_id": memory["id"],
                    "course": memory["course"],
                    "concept": memory["concept"],
                    "original_question": memory["original_question"],
                    "difficulty_note": memory["difficulty_note"],
                }
                for memory in memories
            ],
        }

    async def all_memories(self) -> list[dict[str, Any]]:
        """Return structured records for the optional follow-up worker."""
        await self.initialize()
        async with self._operation_lock:
            docs = await self._session.get_docs()
        return [
            memory
            for doc in docs
            if (memory := self._memory_from_doc(doc)) is not None
            and memory.get("student_id") == self.student_id
        ]

    def _schedule_push(self) -> None:
        if self._push_task is None or self._push_task.done():
            self._push_task = asyncio.create_task(self._debounced_push())

    async def _debounced_push(self) -> None:
        await asyncio.sleep(max(self.sync_debounce_seconds, 0))
        try:
            async with self._operation_lock:
                if not self._dirty:
                    return
                await self._session.push_index()
                self._dirty = False
                log.info("Moss memory synced to cloud: %s", self.index_name)
        except Exception:
            log.exception("Moss memory cloud sync failed; next save will retry")

    async def flush(self) -> None:
        """Wait for a scheduled cloud push, or persist dirty data immediately."""
        task = self._push_task
        if task is not None and not task.done():
            await task
        if self._dirty and self._session is not None:
            async with self._operation_lock:
                await self._session.push_index()
                self._dirty = False

    async def close(self) -> None:
        await self.flush()
