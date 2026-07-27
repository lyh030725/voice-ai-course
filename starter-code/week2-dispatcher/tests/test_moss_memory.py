from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from moss_memory import MossMemoryStore


class FakeDocumentInfo:
    def __init__(self, id, text, metadata=None, embedding=None, payload=None):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.embedding = embedding
        self.payload = payload


class FakeGetDocumentsOptions:
    def __init__(self, doc_ids=None):
        self.doc_ids = doc_ids


class FakeQueryOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeSession:
    def __init__(self):
        self.documents = {}
        self.last_query = None
        self.push_count = 0

    @property
    def doc_count(self):
        return len(self.documents)

    async def get_docs(self, options=None):
        if options and options.doc_ids is not None:
            return [
                self.documents[doc_id]
                for doc_id in options.doc_ids
                if doc_id in self.documents
            ]
        return list(self.documents.values())

    async def add_docs(self, docs):
        added = 0
        updated = 0
        for doc in docs:
            if doc.id in self.documents:
                updated += 1
            else:
                added += 1
            self.documents[doc.id] = doc
        return added, updated

    async def query(self, query, options):
        self.last_query = (query, options)
        return SimpleNamespace(docs=list(self.documents.values())[: options.top_k])

    async def push_index(self):
        self.push_count += 1
        return SimpleNamespace(doc_count=self.doc_count, status="completed")


class FakeClient:
    def __init__(self, project_id, project_key, session):
        self.project_id = project_id
        self.project_key = project_key
        self._session = session
        self.session_args = None

    async def session(self, *, index_name, model_id):
        self.session_args = (index_name, model_id)
        return self._session


class MossMemoryStoreTests(unittest.TestCase):
    def test_embed_deduplicate_recall_and_push(self):
        async def scenario():
            session = FakeSession()
            clients = []

            def make_client(project_id, project_key):
                client = FakeClient(project_id, project_key, session)
                clients.append(client)
                return client

            sdk = SimpleNamespace(
                DocumentInfo=FakeDocumentInfo,
                GetDocumentsOptions=FakeGetDocumentsOptions,
                MossClient=make_client,
                QueryOptions=FakeQueryOptions,
            )
            store = MossMemoryStore(
                "project-1",
                "key-1",
                index_name="student-memory",
                model_id="moss-minilm",
                student_id="student-1",
                sync_debounce_seconds=0,
                sdk_loader=lambda: sdk,
            )

            first = await store.save(
                "AI 개론",
                "Self-Attention",
                "Query와 Key를 왜 곱하나요?",
                "유사도 점수의 의미가 불명확함",
            )
            duplicate = await store.save(
                "AI 개론",
                "Self-Attention",
                "Query와 Key를 왜 곱하나요?",
                "중복 저장되지 않아야 함",
            )
            recalled = await store.recall("attention 유사도")
            await store.flush()

            self.assertEqual(first["status"], "saved")
            self.assertEqual(duplicate["status"], "already_saved")
            self.assertEqual(len(session.documents), 1)
            self.assertEqual(recalled["memories"][0]["memory_id"], first["memory_id"] )
            self.assertEqual(clients[0].session_args, ("student-memory", "moss-minilm"))
            self.assertEqual(session.last_query[0], "attention 유사도")
            self.assertEqual(session.last_query[1].alpha, 0.85)
            self.assertEqual(
                session.last_query[1].filter["condition"],
                {"$eq": "student-1"},
            )
            self.assertEqual(session.push_count, 1)

        asyncio.run(scenario())

    def test_missing_credentials_fail_before_sdk_load(self):
        async def scenario():
            with patch.dict(
                os.environ,
                {"MOSS_PROJECT_ID": "", "MOSS_PROJECT_KEY": ""},
                clear=False,
            ):
                store = MossMemoryStore(
                    project_id="",
                    project_key="",
                    sdk_loader=lambda: self.fail(
                        "SDK must not load without credentials"
                    ),
                )
                with self.assertRaisesRegex(RuntimeError, "MOSS_PROJECT_ID"):
                    await store.initialize()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
