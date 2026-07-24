# M2 — KINGO VOICE TA: memory, PDF-first answers, trusted web fallback

단일 학생용 음성 조교입니다. 대화 중 드러난 취약 개념을 자동 저장하고,
다음 질문에서 다시 불러와 힌트와 후속 질문을 개인화합니다.

## Tool 사용 조건

| Tool | 호출 조건 |
|---|---|
| `save_weak_concept` | 학생이 학습 질문을 하거나 혼란·불완전한 이해를 보이면 동의 없이 자동 호출 |
| `recall_weak_concepts` | 모든 실질적인 학습 질문에 답하기 전에 호출 |
| `search_course_materials` | 사실 기반 학습 질문에 답하기 전에 항상 먼저 호출 |
| `search_trusted_web` | 이전 tool round에서 PDF를 검색했고 근거가 없거나 부족할 때만 호출 |

PDF는 저장소 루트의 `srcs/*.pdf`에서 페이지 단위로 검색합니다. PDF 근거를
사용하면 파일명과 페이지를 답변에 표시합니다.

외부 검색은 다음 신뢰 도메인으로 제한됩니다.

- `skku.edu`
- `arxiv.org`
- `aclanthology.org`
- `proceedings.neurips.cc`
- `jmlr.org`

외부 검색 결과에 신뢰 도메인의 URL이 없으면 답변 근거로 사용하지 않습니다.
사용한 URL은 최종 답변에 자동으로 붙고 `sources/web-search.jsonl`에도 기록됩니다.

## 저장 파일

- `memory/weak_concepts.jsonl`: 자동 저장된 학생 취약 개념
- `memory/followups/*.txt`: 선택적 로컬 워커가 만든 다음 대화용 질문
- `sources/web-search.jsonl`: 외부 검색 질의·답변·출처 감사 로그

대화 문맥 `HISTORY`는 서버 메모리에만 있으므로 서버 재시작 또는 화면의
새 대화 버튼으로 초기화됩니다. 취약 개념 JSONL은 재시작 후에도 유지됩니다.

## 설치 및 실행

```bash
cd /home/student/voice-ai-course/starter-code/week2-dispatcher
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

브라우저에서 <http://localhost:8000>을 엽니다.

API 없이 로컬 tool과 worker를 검증하려면 다음을 실행합니다.

```bash
VOICE_AI_SKIP_DOTENV=1 .venv/bin/python -m unittest discover -s tests -v
```

로컬 후속 질문 파일도 생성하려면 두 번째 터미널에서 실행합니다. 이 워커는
학생 메모리를 외부 서비스로 전송하지 않습니다.

```bash
cd /home/student/voice-ai-course/starter-code/week2-dispatcher
source .venv/bin/activate
python worker.py
```

웹 검색은 xAI Responses API의 서버 측 `web_search`를 사용하며 기본 모델은
`grok-4.5`입니다. 필요하면 환경 변수 `WEB_SEARCH_MODEL`로 변경할 수 있습니다.

## 테스트 예시

1. “Self-Attention에서 Query와 Key를 왜 곱하는지 모르겠어.”
2. “아까 내가 어려워했던 개념과 연결해서 다시 설명해줘.”
3. PDF에 없는 최신 내용을 질문해 PDF 검색 후 신뢰 웹 검색과 URL 표시 확인
