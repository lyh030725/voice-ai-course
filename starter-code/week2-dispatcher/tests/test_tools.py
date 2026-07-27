from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self._messages))]
        )


class ServerToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.original_web_log = server.WEB_SEARCH_LOG
        self.original_pdf_cache = server.PDF_PAGE_CACHE
        server.WEB_SEARCH_LOG = self.root / "sources" / "web-search.jsonl"
        server.PDF_PAGE_CACHE = None
        server.HISTORY.clear()

    def tearDown(self) -> None:
        server.WEB_SEARCH_LOG = self.original_web_log
        server.PDF_PAGE_CACHE = self.original_pdf_cache
        server.HISTORY.clear()
        self.temp_dir.cleanup()

    def test_memory_tools_delegate_to_moss_store(self) -> None:
        memory_id = "M-test"

        async def scenario() -> tuple[dict, dict]:
            with (
                patch.object(
                    server.MOSS_MEMORY,
                    "save",
                    new=AsyncMock(return_value={
                        "memory_id": memory_id,
                        "status": "saved",
                        "concept": "Self-Attention",
                    }),
                ) as save_mock,
                patch.object(
                    server.MOSS_MEMORY,
                    "recall",
                    new=AsyncMock(return_value={
                        "found": True,
                        "topic": "Self-Attention Query",
                        "memories": [{
                            "memory_id": memory_id,
                            "course": "AI 개론",
                            "concept": "Self-Attention",
                            "original_question": "Query와 Key를 왜 곱하나요?",
                            "difficulty_note": "유사도 점수의 의미가 불명확함",
                        }],
                    }),
                ) as recall_mock,
            ):
                saved = json.loads(await server.save_weak_concept(
                    "AI 개론",
                    "Self-Attention",
                    "Query와 Key를 왜 곱하나요?",
                    "유사도 점수의 의미가 불명확함",
                ))
                recalled = json.loads(
                    await server.recall_weak_concepts("Self-Attention Query")
                )
                save_mock.assert_awaited_once()
                recall_mock.assert_awaited_once_with("Self-Attention Query")
                return saved, recalled

        saved, recalled = asyncio.run(scenario())
        self.assertEqual(saved["status"], "saved")
        self.assertEqual(recalled["memories"][0]["memory_id"], memory_id)

    def test_text_endpoint_runs_agent_without_audio(self) -> None:
        async def scenario() -> dict:
            with patch.object(
                server,
                "think",
                new=AsyncMock(return_value=(
                    "Query와 Key의 내적은 관련도를 측정해요.",
                    ["recall_weak_concepts", "search_course_materials"],
                )),
            ) as think_mock:
                result = await server.answer_text_endpoint(
                    server.TextQuestion(text="  Query와 Key를 왜 곱해?  ")
                )
                self.assertEqual(think_mock.await_args.args[0], "Query와 Key를 왜 곱해?")
                return result

        result = asyncio.run(scenario())
        self.assertEqual(result["transcript"], "Query와 Key를 왜 곱해?")
        self.assertIn("내적", result["reply"])
        self.assertEqual(result["tools"][0], "recall_weak_concepts")
        self.assertIn("llm", result["timings"])
        self.assertIn("total", result["timings"])

    def test_text_endpoint_rejects_blank_text(self) -> None:
        with self.assertRaises(server.HTTPException) as raised:
            asyncio.run(
                server.answer_text_endpoint(server.TextQuestion(text="   "))
            )
        self.assertEqual(raised.exception.status_code, 422)

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

    def test_think_uses_one_grok_call_with_retrieved_context(self) -> None:
        decision = json.dumps({
            "answer": "lecture.pdf p.3에 따르면 내적은 관련도를 측정해요.",
            "memory": None,
        }, ensure_ascii=False)
        fake_client = FakeChatClient([
            SimpleNamespace(content=decision, tool_calls=[]),
        ])

        async def scenario():
            with (
                patch.object(
                    server,
                    "recall_weak_concepts",
                    new=AsyncMock(return_value=json.dumps({
                        "found": False,
                        "memories": [],
                    })),
                ),
                patch.object(
                    server,
                    "search_course_materials",
                    return_value=json.dumps({
                        "found": True,
                        "results": [{
                            "source": "lecture.pdf p.3",
                            "excerpt": "내적은 관련도를 측정한다.",
                        }],
                    }),
                ),
                patch.object(server, "xai_client", return_value=fake_client),
            ):
                timer = server.StageTimer()
                reply, tools = await server.think("Query와 Key를 왜 곱해?", timer)
                return reply, tools, timer.timings_ms

        reply, tools, timings = asyncio.run(scenario())
        self.assertEqual(len(fake_client.calls), 1)
        self.assertIn("lecture.pdf p.3", reply)
        self.assertEqual(
            tools,
            ["recall_weak_concepts", "search_course_materials"],
        )
        self.assertIn("recall", timings)
        self.assertIn("pdf", timings)
        self.assertIn("grok", timings)
        call = fake_client.calls[0]
        self.assertEqual(call["reasoning_effort"], "none")
        self.assertEqual(call["response_format"]["type"], "json_schema")

    def test_think_saves_memory_in_background(self) -> None:
        decision = json.dumps({
            "answer": "lecture.pdf p.3을 바탕으로 힌트를 줄게요.",
            "memory": {
                "course": "AI 개론",
                "concept": "Self-Attention",
                "original_question": "Query와 Key를 왜 곱해?",
                "difficulty_note": "내적과 관련도의 관계가 불명확함",
            },
        }, ensure_ascii=False)
        fake_client = FakeChatClient([
            SimpleNamespace(content=decision, tool_calls=[]),
        ])

        async def scenario():
            save_mock = AsyncMock(return_value=json.dumps({"status": "saved"}))
            with (
                patch.object(
                    server,
                    "recall_weak_concepts",
                    new=AsyncMock(return_value=json.dumps({
                        "found": False,
                        "memories": [],
                    })),
                ),
                patch.object(
                    server,
                    "search_course_materials",
                    return_value=json.dumps({"found": True, "results": []}),
                ),
                patch.object(server, "xai_client", return_value=fake_client),
                patch.object(server, "save_weak_concept", new=save_mock),
            ):
                _, tools = await server.think("Query와 Key를 왜 곱해?", server.StageTimer())
                await asyncio.sleep(0)
                save_mock.assert_awaited_once()
                return tools

        tools = asyncio.run(scenario())
        self.assertIn("save_weak_concept", tools)

    def test_think_uses_web_only_when_pdf_has_no_result(self) -> None:
        decision = json.dumps({"answer": "외부 출처 https://arxiv.org/abs/1", "memory": None})
        fake_client = FakeChatClient([
            SimpleNamespace(content=decision, tool_calls=[]),
        ])

        async def scenario():
            with (
                patch.object(
                    server,
                    "recall_weak_concepts",
                    new=AsyncMock(return_value=json.dumps({"found": False, "memories": []})),
                ),
                patch.object(
                    server,
                    "search_course_materials",
                    return_value=json.dumps({"found": False, "results": []}),
                ),
                patch.object(
                    server,
                    "search_trusted_web",
                    return_value=json.dumps({
                        "found": True,
                        "answer": "근거",
                        "sources": ["https://arxiv.org/abs/1"],
                    }),
                ) as web_mock,
                patch.object(server, "xai_client", return_value=fake_client),
            ):
                _, tools = await server.think("PDF에 없는 질문", server.StageTimer())
                web_mock.assert_called_once()
                return tools

        tools = asyncio.run(scenario())
        self.assertIn("search_trusted_web", tools)


class WorkerTests(unittest.TestCase):
    def test_process_once_creates_local_followup_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_file = root / "memory" / "worker-processed.txt"
            followups_dir = root / "memory" / "followups"
            memory = {
                "id": "M-1000",
                "student_id": "student-1",
                "course": "AI 개론",
                "concept": "Attention",
                "original_question": "왜 필요한가요?",
                "difficulty_note": "가중합의 의미가 불명확함",
            }

            with (
                patch.object(
                    worker.MOSS_MEMORY,
                    "all_memories",
                    new=AsyncMock(return_value=[memory]),
                ),
                patch.object(worker, "PROCESSED_FILE", processed_file),
                patch.object(worker, "FOLLOWUP_GUIDES_DIR", followups_dir),
            ):
                self.assertEqual(asyncio.run(worker.process_once()), 1)
                self.assertEqual(asyncio.run(worker.process_once()), 0)

            self.assertTrue((followups_dir / "M-1000.txt").exists())
            self.assertEqual(processed_file.read_text(encoding="utf-8").strip(), "M-1000")


if __name__ == "__main__":
    unittest.main()
