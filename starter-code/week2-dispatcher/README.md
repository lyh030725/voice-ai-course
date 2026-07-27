# Week 2 — KINGO VOICE TA

음성 또는 텍스트 질문을 받아 강의 PDF 근거와 학생별 취약 개념을 활용해 답하는
1인용 AI 학습 조교입니다. 취약 개념은 [Moss](https://www.moss.dev/)에 임베딩해
저장하고, 답변은 xAI Grok으로 생성합니다.

## 주요 기능

- 음성 질문: xAI STT → Grok 답변 → xAI TTS
- 텍스트 질문: STT/TTS 없이 같은 답변 파이프라인을 빠르게 테스트
- 강의 자료 우선 검색: 로컬 `srcs/*.pdf`에서 파일명과 페이지 근거 제공
- 개인화 메모리: Moss에서 관련 취약 개념을 의미 검색하고 자동 저장
- 신뢰 웹 검색: PDF 근거가 없을 때 허용된 공식·학술 도메인만 검색
- 지연 시간 표시: STT, 검색, Grok, TTS 단계를 화면과 API 응답에서 확인

## 처리 흐름

```text
질문
 ├─ Moss 취약 개념 조회 ─┐
 └─ PDF 페이지 검색 ─────┤ 병렬 실행
                         ↓
                Grok 구조화 응답 1회
                  ├─ 학생 답변
                  └─ 저장할 취약 개념
                         ↓
                Moss 백그라운드 동기화
```

PDF 인덱스는 서버 시작 시 미리 생성합니다. 일반적인 PDF 기반 질문은 Grok API를
한 번만 호출하며, 취약 개념 저장은 답변을 지연시키지 않도록 백그라운드에서
처리합니다. PDF 결과가 없을 때만 신뢰 웹 검색 호출이 추가됩니다.

## 설치

```bash
cd /home/student/voice-ai-course/starter-code/week2-dispatcher
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`에 본인의 xAI 및 Moss 값을 입력합니다. 실제 키는 `.env.example`이나 Git에
커밋하지 마세요.

| 환경 변수 | 용도 |
|---|---|
| `XAI_API_KEY` | xAI API 인증 |
| `CHAT_MODEL` | 답변 모델, 기본값 `grok-4.3` |
| `CHAT_REASONING_EFFORT` | 추론 강도, 기본값 `none` |
| `CHAT_MAX_TOKENS` | 답변 최대 토큰 수 |
| `TTS_VOICE` | xAI TTS 음성 ID |
| `MOSS_PROJECT_ID` | Moss 프로젝트 ID |
| `MOSS_PROJECT_KEY` | Moss 프로젝트 키 |
| `MOSS_MEMORY_INDEX` | 학생 메모리 인덱스 이름 |
| `MOSS_MEMORY_MODEL` | Moss 임베딩 모델 |
| `MOSS_STUDENT_ID` | 학생 식별자 |

## 실행

```bash
source .venv/bin/activate
uvicorn server:app --reload --port 8000
```

브라우저에서 <http://localhost:8000>을 열면 음성과 텍스트 입력을 모두 사용할 수
있습니다. 강의 PDF는 저장소 루트의 `srcs/`에 로컬로 넣으세요. PDF 파일은 GitHub에
올라가지 않도록 제외되어 있습니다.

## 텍스트로 빠르게 테스트

화면의 텍스트 입력창을 사용하거나 백엔드 API를 직접 호출할 수 있습니다.

```bash
curl -s http://localhost:8000/answer-text \
  -H 'Content-Type: application/json' \
  -d '{"text":"Self-Attention에서 Query와 Key를 왜 곱해?"}' \
  | python -m json.tool
```

응답에는 `reply`, 수행된 검색·저장 작업, 단계별 `timings`가 포함됩니다.
`POST /reset`은 현재 대화 기록만 초기화하며 Moss의 취약 개념은 유지합니다.

## 신뢰 웹 검색

PDF에서 근거를 찾지 못한 경우에만 다음 도메인을 검색합니다.

- `skku.edu`
- `arxiv.org`
- `aclanthology.org`
- `proceedings.neurips.cc`
- `jmlr.org`

출처 URL이 없는 외부 결과는 사용하지 않습니다. 검색 기록은 로컬
`sources/web-search.jsonl`에 남습니다.

## 워커

선택적 워커는 Moss 메모리를 읽어 다음 학습 대화용 후속 질문 파일을 만듭니다.

```bash
source .venv/bin/activate
python worker.py
```

## 테스트

실제 xAI 또는 Moss 호출 없이 단위 테스트를 실행합니다.

```bash
VOICE_AI_SKIP_DOTENV=1 .venv/bin/python -m unittest discover -s tests -v
```
