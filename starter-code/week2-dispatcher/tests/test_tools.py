from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["VOICE_AI_SKIP_DOTENV"] = "1"
MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import server
import worker


class ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class FakeChatClient:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self._messages = iter(messages)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self._messages))]
        )


class ServerToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.original_memory_file = server.WEAK_CONCEPTS_FILE
        self.original_web_log = server.WEB_SEARCH_LOG
        self.original_pdf_cache = server.PDF_PAGE_CACHE
        server.WEAK_CONCEPTS_FILE = self.root / "memory" / "weak_concepts.jsonl"
        server.WEB_SEARCH_LOG = self.root / "sources" / "web-search.jsonl"
        server.PDF_PAGE_CACHE = None
        server.HISTORY.clear()

    def tearDown(self) -> None:
        server.WEAK_CONCEPTS_FILE = self.original_memory_file
        server.WEB_SEARCH_LOG = self.original_web_log
        server.PDF_PAGE_CACHE = self.original_pdf_cache
        server.HISTORY.clear()
        self.temp_dir.cleanup()

    def test_save_deduplicates_and_recall_finds_memory(self) -> None:
        first = json.loads(server.save_weak_concept(
            "AI 개론",
            "Self-Attention",
            "Query와 Key를 왜 곱하나요?",
            "유사도 점수의 의미가 불명확함",
        ))
        duplicate = json.loads(server.save_weak_concept(
            "AI 개론",
            "Self-Attention",
            "Query와 Key를 왜 곱하나요?",
            "다른 메모는 중복 판단에 영향을 주지 않음",
        ))
        recalled = json.loads(server.recall_weak_concepts("Self-Attention Query"))

        self.assertEqual(first["status"], "saved")
        self.assertEqual(duplicate["status"], "already_saved")
        self.assertEqual(len(server._read_jsonl(server.WEAK_CONCEPTS_FILE)), 1)
        self.assertEqual(recalled["memories"][0]["memory_id"], first["memory_id"])

    def test_pdf_search_returns_filename_and_page(self) -> None:
        server.PDF_PAGE_CACHE = [
            {"file": "lecture.pdf", "page": 3, "text": "Query와 Key의 내적은 토큰 유사도를 계산한다."},
            {"file": "other.pdf", "page": 8, "text": "회귀 분석 소개"},
        ]

        result = json.loads(server.search_course_materials("Query Key 유사도"))

        self.assertTrue(result["found"])
        self.assertEqual(result["results"][0]["source"], "lecture.pdf p.3")

    def test_trusted_url_filter_rejects_unconfigured_domains(self) -> None:
        response = SimpleNamespace(
            output_text=(
                "https://arxiv.org/abs/1234 and "
                "https://evil.example/not-trusted"
            )
        )

        self.assertEqual(
            server._trusted_urls(response),
            ["https://arxiv.org/abs/1234"],
        )

    def test_web_search_requires_pdf_fallback_reason(self) -> None:
        result = json.loads(server.search_trusted_web("attention", reason=""))
        self.assertIn("error", result)

    def test_think_blocks_web_search_before_pdf_round(self) -> None:
        web_call = ToolCall(
            "call-1",
            "search_trusted_web",
            json.dumps({
                "query": "latest attention research",
                "pdf_evidence_insufficient": True,
                "reason": "not in PDFs",
            }),
        )
        fake_client = FakeChatClient([
            SimpleNamespace(content=None, tool_calls=[web_call]),
            SimpleNamespace(content="PDF를 먼저 확인해야 해요.", tool_calls=[]),
        ])

        with patch.object(server, "xai_client", return_value=fake_client):
            reply, tools = asyncio.run(server.think("최신 연구를 알려줘", server.StageTimer()))

        tool_result = json.loads(
            next(item["content"] for item in server.HISTORY if item["role"] == "tool")
        )
        self.assertIn("earlier tool round", tool_result["error"])
        self.assertEqual(reply, "PDF를 먼저 확인해야 해요.")
        self.assertEqual(tools, ["search_trusted_web"])

    def test_think_returns_tool_argument_error_instead_of_crashing(self) -> None:
        invalid_call = ToolCall("call-2", "recall_weak_concepts", "not-json")
        fake_client = FakeChatClient([
            SimpleNamespace(content=None, tool_calls=[invalid_call]),
            SimpleNamespace(content="질문을 다시 정리해 주세요.", tool_calls=[]),
        ])

        with patch.object(server, "xai_client", return_value=fake_client):
            asyncio.run(server.think("질문", server.StageTimer()))

        tool_result = json.loads(
            next(item["content"] for item in server.HISTORY if item["role"] == "tool")
        )
        self.assertIn("invalid tool arguments", tool_result["error"])


class WorkerTests(unittest.TestCase):
    def test_process_once_creates_local_followup_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory_file = root / "memory" / "weak_concepts.jsonl"
            processed_file = root / "memory" / "worker-processed.txt"
            followups_dir = root / "memory" / "followups"
            memory_file.parent.mkdir(parents=True)
            memory_file.write_text(json.dumps({
                "id": "M-1000",
                "course": "AI 개론",
                "concept": "Attention",
                "original_question": "왜 필요한가요?",
                "difficulty_note": "가중합의 의미가 불명확함",
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with (
                patch.object(worker, "WEAK_CONCEPTS_FILE", memory_file),
                patch.object(worker, "PROCESSED_FILE", processed_file),
                patch.object(worker, "FOLLOWUP_GUIDES_DIR", followups_dir),
            ):
                self.assertEqual(worker.process_once(), 1)
                self.assertEqual(worker.process_once(), 0)

            self.assertTrue((followups_dir / "M-1000.txt").exists())
            self.assertEqual(processed_file.read_text(encoding="utf-8").strip(), "M-1000")


if __name__ == "__main__":
    unittest.main()
