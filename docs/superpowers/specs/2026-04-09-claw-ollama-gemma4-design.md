# Claw Code + Ollama Gemma 4 31B 통합 설계

- **작성일:** 2026-04-09
- **상태:** 승인됨
- **환경:** Windows 11, RTX 4090 24GB VRAM, RAM 32GB, Ollama v0.20.4

## 1. 목표

Claw Code CLI의 백엔드를 Ollama에서 실행되는 Gemma 4 31B Dense 모델로 교체하여 완전한 로컬 코드 에이전트를 구현한다. 추가로:

- 사용 중 자동 피드백 수집 + 파인튜닝 파이프라인 구축
- API Gateway를 통한 외부 LLM 서버 기능 제공
- 코딩 품질 최우선, VRAM 24GB 최대 활용

## 2. 모델 선택

| 항목 | 결정 |
|------|------|
| **모델** | Gemma 4 31B Dense |
| **양자화** | Q4_K_M (~20GB) |
| **컨텍스트** | 32,768 토큰 |
| **VRAM 예산** | 모델 ~20GB + KV cache ~3GB + 런타임 ~1GB = ~24GB |
| **Tool calling** | 네이티브 지원 (전용 토큰 6개) |
| **파인튜닝** | QLoRA (4-bit NF4 + LoRA r=32), Unsloth 사용 |

## 3. 전체 아키텍처

```
 ┌─ 외부 PC ──────────────────────┐     ┌─ 내 PC (RTX 4090) ──────────────┐
 │                                │     │                                  │
 │  Claw Code / Claude Code       │     │  ┌────────────────────────────┐  │
 │  또는 아무 LLM 클라이언트       │     │  │      API Gateway (NEW)     │  │
 │                                │     │  │  API Key 인증 + Rate Limit │  │
 │  provider: ollama-remote       │     │  │  + TLS + 경로 필터링       │  │
 └───────────┬────────────────────┘     │  │       :8443 (외부 포트)    │  │
             │ HTTPS + Bearer Token     │  └─────────────┬──────────────┘  │
             └──────────────────────────┼───────────────▶│                  │
                                        │                │ 127.0.0.1:11434  │
                                        │  ┌─────────────▼──────────────┐  │
 ┌─ 내 PC 로컬 사용 ─────────────┐     │  │       Ollama Server        │  │
 │                                │     │  │     Gemma 4 31B Q4KM      │  │
 │  Claw CLI / Runtime            │     │  └────────────────────────────┘  │
 │  ConversationRuntime           │     │                                  │
 │        │                       │     │  ┌────────────────────────────┐  │
 │   OllamaClient (NEW)          │─────┼─▶│  FeedbackPipeline (NEW)   │  │
 │  localhost:11434 직접연결       │     │  │  SessionLogger            │  │
 └────────────────────────────────┘     │  │  → DataExporter            │  │
                                        │  │  → FineTuneRunner          │  │
                                        │  │  → ModelReloader           │  │
                                        │  └────────────────────────────┘  │
                                        └──────────────────────────────────┘
```

**내 PC의 두 가지 역할:**
1. **코드 에이전트** -- Claw Code + OllamaClient로 로컬 개발
2. **LLM 서버** -- 외부에 Gemma 4 추론 API 제공

**외부 PC는 내 PC의 Claw Code가 아닌 Ollama LLM 서버에만 접근한다.**

## 4. OllamaClient 상세 설계

### 4.1 구조체

```rust
// crates/api/src/providers/ollama.rs

pub struct OllamaClient {
    http: reqwest::Client,
    base_url: String,               // default: http://localhost:11434
    model: String,                  // default: gemma4:31b
    default_options: OllamaOptions,
    manager: OllamaManager,
}

pub struct OllamaOptions {
    pub num_ctx: u32,               // 컨텍스트 윈도우 (default: 32768)
    pub num_gpu: Option<i32>,       // GPU 레이어 수 (-1 = 전부)
    pub temperature: f64,
    pub top_p: f64,
    pub repeat_penalty: f64,
    pub keep_alive: String,         // 모델 메모리 유지 ("-1" = 영구)
}

pub struct OllamaManager {
    http: reqwest::Client,
    base_url: String,
}
```

### 4.2 Ollama 네이티브 API 매핑

| Claw 동작 | Ollama API | 엔드포인트 |
|-----------|-----------|-----------|
| `send_message()` | `/api/chat` (stream: false) | POST |
| `stream_message()` | `/api/chat` (stream: true) | POST, NDJSON |
| Tool calling | `/api/chat` tools 파라미터 | Gemma 4 네이티브 지원 |
| 모델 정보 조회 | `/api/show` | POST |
| 로드된 모델 확인 | `/api/ps` | GET |
| 모델 로드/언로드 | `/api/chat` keep_alive 제어 | POST |

### 4.3 요청 변환

```
Claw MessageRequest                    Ollama /api/chat Request
─────────────────────                  ─────────────────────────
model                ──────────────▶  model
messages[]           ──── 변환 ────▶  messages[]
  role: user                           role: user
  role: assistant                      role: assistant
system               ──────────────▶  messages[0] role: system
tools[]              ──── 변환 ────▶  tools[]
max_tokens           ──────────────▶  options.num_predict
temperature          ──────────────▶  options.temperature
stream               ──────────────▶  stream
```

### 4.4 응답 변환

```
Ollama /api/chat Response              Claw MessageResponse
─────────────────────────              ─────────────────────
message.content        ──────────────▶  content[Text]
message.tool_calls[]   ──── 변환 ────▶  content[ToolUse]
done_reason            ──── 매핑 ────▶  stop_reason
eval_count             ──────────────▶  usage.output_tokens
prompt_eval_count      ──────────────▶  usage.input_tokens
```

### 4.5 스트리밍 처리

Ollama는 SSE가 아닌 NDJSON (줄바꿈 구분 JSON) 스트리밍:

```
{"message":{"role":"assistant","content":"Hello"},"done":false}
{"message":{"role":"assistant","content":" world"},"done":false}
{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}
```

Claw StreamEvent 시퀀스로 변환:
```
ContentBlockStart → ContentBlockDelta("Hello") → ContentBlockDelta(" world")
→ ContentBlockStop → MessageStop
```

### 4.6 ProviderKind 확장

```rust
pub enum ProviderKind {
    Anthropic,
    Xai,
    OpenAi,
    Ollama,       // NEW
}
```

모델 alias 등록:

| alias | ProviderKind | model |
|-------|-------------|-------|
| `gemma4` | Ollama | `gemma4:31b` |
| `gemma4-31b` | Ollama | `gemma4:31b` |
| `gemma4-26b` | Ollama | `gemma4:26b-a4b` |
| `local` | Ollama | 환경변수 `OLLAMA_MODEL`에서 읽기 |

### 4.7 환경변수

| 변수 | 용도 | 기본값 |
|------|------|--------|
| `OLLAMA_BASE_URL` | Ollama 서버 주소 | `http://localhost:11434` |
| `OLLAMA_MODEL` | 기본 모델 | `gemma4:31b` |
| `OLLAMA_NUM_CTX` | 컨텍스트 윈도우 | `32768` |
| `OLLAMA_KEEP_ALIVE` | 모델 메모리 유지 | `-1` (영구) |
| `OLLAMA_API_KEY` | Gateway 인증 (외부 접근 시만 사용, 로컬은 불필요) | 없음 |

## 5. API Gateway (외부 접근)

### 5.1 구성

Rust로 경량 게이트웨이를 별도 바이너리(`claw-gateway`)로 구현한다. Nginx 같은 외부 의존성 없이 단일 바이너리로 배포 가능.

```bash
claw-gateway serve --port 8443 --ollama http://127.0.0.1:11434
```

### 5.2 처리 파이프라인

```
외부 요청 → TLS 종단 (rustls) → API Key 인증 → Rate Limiter → 경로 필터링 → 프록시 → Ollama
```

1. **TLS 종단** -- rustls, 자체서명 또는 Let's Encrypt
2. **API Key 인증** -- `Authorization: Bearer <key>` 헤더 검증
3. **Rate Limiter** -- 키별 분당 요청 제한, 동시 요청 제한 (GPU 1장)
4. **경로 필터링** -- 안전한 엔드포인트만 허용
5. **리버스 프록시** -- `127.0.0.1:11434`로 전달

### 5.3 API Key 관리

```json
// ~/.claw/gateway/keys.json
{
  "keys": [
    {
      "id": "key_01",
      "name": "내 노트북",
      "secret": "claw-sk-abc123...",
      "rate_limit": 30,
      "enabled": true
    }
  ]
}
```

CLI:
```bash
claw-gateway key create --name "내 노트북"    # claw-sk-xxxx 출력
claw-gateway key list
claw-gateway key revoke key_02
```

### 5.4 경로 허용/차단 정책

| 엔드포인트 | 허용 | 이유 |
|-----------|------|------|
| `POST /api/chat` | O | 추론 |
| `POST /api/generate` | O | 추론 |
| `POST /v1/chat/completions` | O | OpenAI 호환 추론 |
| `POST /api/show` | O | 모델 정보 (읽기 전용) |
| `GET /api/ps` | O | 상태 확인 |
| `GET /api/tags` | O | 모델 목록 |
| `DELETE /api/delete` | X | 모델 삭제 방지 |
| `POST /api/create` | X | 모델 생성은 로컬만 |
| `POST /api/pull` | X | 모델 다운로드 방지 |
| `POST /api/push` | X | 모델 업로드 방지 |

### 5.5 외부 클라이언트 사용법

```bash
# 외부 PC에서
export OLLAMA_BASE_URL="https://my-pc.duckdns.org:8443"
export OLLAMA_API_KEY="claw-sk-abc123..."
claw --model gemma4 prompt "이 코드를 리뷰해줘"
```

OpenAI 호환 엔드포인트도 제공하므로 Aider, Continue 등에서도 사용 가능.

### 5.6 네트워크 설정

- 공유기 포트포워딩: 외부 8443 → 내부 192.168.x.x:8443
- DuckDNS (무료 DDNS): my-pc.duckdns.org → 공인 IP 자동 갱신
- Let's Encrypt: HTTPS 인증서 자동 발급/갱신

### 5.7 핵심 의존성

| crate | 용도 |
|-------|------|
| `axum` | HTTP 서버 프레임워크 |
| `rustls` | TLS |
| `tower` | 미들웨어 (rate limit, auth) |
| `hyper` | 리버스 프록시 |

## 6. FeedbackPipeline (자동 파인튜닝 루프)

### 6.1 전체 흐름

```
Claw Code 대화 세션
    │
    ▼
SessionLogger (Hook) → 세션 로그 저장 (~/.claw/logs/)
    │
    ▼ (트리거 조건 충족 시)
DataExporter → 세션 필터링 + JSONL 변환
    │
    ▼
FineTuneRunner → Unsloth subprocess 실행
    │
    ▼
ModelReloader → GGUF 검증 → ollama create → sanity test → hot-swap
```

### 6.2 세션 로그 → 학습 데이터 변환

**입력 (Claw 세션 로그):**
```json
{
  "session_id": "sess_001",
  "messages": [
    {"role": "user", "content": "정렬 알고리즘 구현해줘"},
    {"role": "assistant", "content": "```python\ndef quicksort(arr)...\n```",
     "tool_calls": [{"name": "Write", "input": {"path": "sort.py"}}]},
    {"role": "tool_result", "tool_use_id": "...", "content": "파일 작성 완료"},
    {"role": "user", "content": "좋아, 잘 동작해"}
  ],
  "feedback": "accepted"
}
```

**출력 (학습용 JSONL):**
```jsonl
{"conversations": [{"role": "user", "content": "정렬 알고리즘 구현해줘"}, {"role": "assistant", "content": "```python\ndef quicksort(arr)...\n```"}]}
```

### 6.3 피드백 수집 기준

| 신호 | 의미 | 가중치 |
|------|------|--------|
| 유저가 코드를 수정 없이 사용 | 암묵적 긍정 | 중 |
| 유저가 "좋아", "완벽" 등 응답 | 명시적 긍정 | 높음 |
| tool 실행 성공 + 에러 없음 | 자동 긍정 | 낮음 |
| 유저가 같은 요청 재시도 | 암묵적 부정 → 제외 | - |
| 유저가 "아니", "다시" 등 응답 | 명시적 부정 → 제외 | - |

### 6.4 DataExporter 처리

1. 세션 로그 필터링 -- 긍정 피드백 대화만 선별
2. Claw 포맷 → 학습용 JSONL 변환
   - system prompt → `{"role": "system", ...}`
   - user message → `{"role": "user", ...}`
   - assistant + tool_use → `{"role": "assistant", ...}`
   - tool_result → 제거 (학습 불필요)
3. 데이터 검증 -- 중복 제거, 토큰 길이 제한, 포맷 검증

### 6.5 파인튜닝 트리거

**자동 트리거 (`claw finetune auto`):**
- 새 긍정 데이터 100개 이상 누적 AND
- 마지막 파인튜닝 이후 7일 경과
- 두 조건 모두 충족 시 실행

**수동 트리거:**
```bash
claw finetune run                    # 즉시 실행
claw finetune run --epochs 2         # 하이퍼파라미터 지정
claw finetune status                 # 현재 상태 확인
claw finetune data stats             # 수집 데이터 통계
claw finetune rollback               # 이전 모델로 복원
```

### 6.6 ModelReloader

1. 새 GGUF 파일 검증 (크기, 포맷 체크)
2. `ollama create my-gemma4-v{N} -f Modelfile`
3. sanity test -- 기본 코딩 질문 3개로 품질 확인
4. 통과 시: OllamaClient 모델명 교체 (hot-swap)
5. 실패 시: 이전 모델 유지 + 경고 로그
6. 이전 버전 모델 보관 (최근 3개까지)

### 6.7 파인튜닝 설정 (Unsloth QLoRA)

| 항목 | 값 |
|------|-----|
| 기법 | QLoRA (4-bit NF4) |
| LoRA rank | 32 |
| LoRA alpha | 64 (2x rank) |
| 옵티마이저 | adamw_8bit |
| 배치 사이즈 | 1 (gradient_accumulation=8) |
| max_seq_length | 4096 |
| 학습률 | 2e-4 |
| GGUF 양자화 | q4_k_m |
| 학습 VRAM | ~22GB / 24GB |
| 학습 시간 | ~10K 샘플 기준 30-45분 |
| 디스크 필요 | ~130GB (모델+변환 임시파일 포함) |

## 7. CLI 인터페이스

### 7.1 새로운 명령어

```bash
# 기본 사용 (로컬 Gemma 4)
claw --model gemma4 prompt "이 코드를 설명해줘"
claw --model gemma4                          # 인터랙티브 세션
claw --model local                           # OLLAMA_MODEL 환경변수 사용

# Ollama 관리
claw ollama status                           # 서버 상태 + GPU 사용량
claw ollama models                           # 사용 가능한 모델 목록
claw ollama load gemma4:31b                  # 모델 미리 로드
claw ollama unload                           # VRAM 해제

# 게이트웨이
claw gateway serve                           # API 게이트웨이 시작
claw gateway serve --port 8443 --tls auto    # HTTPS + Let's Encrypt
claw gateway key create --name "노트북"       # API 키 생성
claw gateway key list                        # 키 목록
claw gateway key revoke key_02               # 키 폐기

# 파인튜닝
claw finetune status                         # 수집 현황 + 마지막 학습
claw finetune data stats                     # 학습 데이터 통계
claw finetune data export                    # JSONL 수동 내보내기
claw finetune run                            # 즉시 파인튜닝 실행
claw finetune run --epochs 2 --rank 32       # 하이퍼파라미터 지정
claw finetune rollback                       # 이전 모델로 복원
claw finetune auto --enable                  # 자동 파인튜닝 활성화

# 헬스체크 (기존 doctor 확장)
claw doctor                                  # 기존 + Ollama 연결 체크 추가
```

### 7.2 설정 파일

```json
// ~/.claw/config.json
{
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "gemma4:31b",
    "options": {
      "num_ctx": 32768,
      "num_gpu": -1,
      "keep_alive": "-1",
      "temperature": 0.7,
      "top_p": 0.9
    }
  },
  "gateway": {
    "enabled": false,
    "port": 8443,
    "tls": {
      "mode": "self-signed",
      "cert_path": null,
      "key_path": null
    },
    "allowed_endpoints": [
      "/api/chat", "/api/generate", "/v1/chat/completions",
      "/api/show", "/api/ps", "/api/tags"
    ],
    "default_rate_limit": 30
  },
  "finetune": {
    "auto_enabled": false,
    "min_samples": 100,
    "interval_days": 7,
    "python_venv": "~/.claw/finetune/venv",
    "training": {
      "max_seq_length": 4096,
      "lora_rank": 32,
      "lora_alpha": 64,
      "epochs": 1,
      "learning_rate": 2e-4,
      "batch_size": 1,
      "gradient_accumulation": 8
    },
    "export": {
      "quantization": "q4_k_m"
    },
    "model_retention": 3
  }
}
```

## 8. Workspace crate 구조

```
rust/
├── Cargo.toml                          # workspace 정의
├── crates/
│   ├── api/
│   │   └── src/providers/
│   │       ├── mod.rs                  # ProviderKind::Ollama 추가
│   │       ├── anthropic.rs            # (기존)
│   │       ├── openai_compat.rs        # (기존)
│   │       └── ollama.rs               # NEW - OllamaClient + OllamaManager
│   ├── runtime/
│   │   └── src/conversation.rs         # OllamaClient 연결
│   ├── feedback/                       # NEW crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── logger.rs               # SessionLogger
│   │       ├── exporter.rs             # DataExporter
│   │       ├── runner.rs               # FineTuneRunner
│   │       ├── reloader.rs             # ModelReloader
│   │       ├── trigger.rs              # 자동/수동 트리거
│   │       └── quality.rs              # 피드백 품질 분석
│   ├── gateway/                        # NEW crate (별도 바이너리)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs                 # claw-gateway 바이너리
│   │       ├── auth.rs                 # API Key 인증
│   │       ├── proxy.rs                # 리버스 프록시
│   │       ├── ratelimit.rs            # Rate limiter
│   │       ├── tls.rs                  # TLS 설정
│   │       └── keys.rs                 # 키 관리 CRUD
│   ├── cli/
│   │   └── src/commands/
│   │       ├── ollama.rs               # NEW - claw ollama 서브커맨드
│   │       ├── gateway.rs              # NEW - claw gateway 서브커맨드
│   │       └── finetune.rs             # NEW - claw finetune 서브커맨드
```

## 9. 파일 구조 (런타임 데이터)

```
~/.claw/
├── config.json                          # 전체 설정
├── logs/
│   ├── session_2026-04-09_001.jsonl     # 세션 로그
│   └── ...
├── finetune/
│   ├── data/
│   │   ├── train.jsonl                  # 변환된 학습 데이터
│   │   └── data_index.json              # 세션 포함 추적
│   ├── runs/
│   │   └── run_2026-04-15/
│   │       ├── config.json              # 학습 하이퍼파라미터
│   │       ├── train.log                # 학습 로그
│   │       └── lora_adapter/            # LoRA 가중치
│   ├── models/
│   │   ├── gemma4-v1.gguf              # 파인튜닝된 모델 (최근 3개 보관)
│   │   ├── gemma4-v2.gguf
│   │   └── current → gemma4-v2.gguf    # 심볼릭 링크
│   ├── scripts/
│   │   └── finetune.py                  # Unsloth 학습 스크립트 (자동 생성)
│   └── venv/                            # Python 가상환경 (Unsloth)
└── gateway/
    └── keys.json                        # API 키
```

## 10. 에러 처리

| 상황 | 처리 |
|------|------|
| Ollama 서버 미실행 | 연결 실패 시 `claw doctor` 안내, 자동 재시도 3회 |
| 모델 미로드 | 자동으로 `/api/chat` 호출 시 로드 (keep_alive 활용) |
| VRAM 부족 | `num_ctx` 자동 축소 제안, 다른 모델 언로드 안내 |
| 파인튜닝 실패 | 이전 모델 유지, 에러 로그 저장, 수동 재시도 안내 |
| 게이트웨이 인증 실패 | 403 반환, 실패 로그 기록 |
| sanity test 실패 | 파인튜닝 모델 롤백, 이전 버전 유지 |

## 11. 의존성 요약

| crate | 신규 의존성 | 용도 |
|-------|-----------|------|
| `api` | 없음 (reqwest 기존) | OllamaClient |
| `feedback` | `serde_json`, `tokio::process` | 로그, subprocess |
| `gateway` | `axum`, `tower`, `rustls`, `hyper` | API 게이트웨이 |
| `cli` | `clap` (기존) | 서브커맨드 |
