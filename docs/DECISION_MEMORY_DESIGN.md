# Decision Memory Phase 1 — 최종 설계 (합성본)

> kumiho-SDKs issue #43. 3개 독립 설계(A=schema-first 24점, B=capture-first 20.5점, C=query-first 22점)를
> 3-렌즈 판정 결과에 따라 합성했다. **골격은 A(최고점)**, 캡처 파이프라인은 B의 디테일을 이식,
> 쿼리 융합은 C의 사전식 정렬을 이식. 판정단이 지적한 fatal flaw는 전부 해당 절에서
> **[판정 반영]** 마커로 인라인 해소한다.

---

## 0. 설계를 지배하는 코드베이스 제약 (전부 실코드 검증 완료)

1. **Typed 노드는 retrieve tool에 보이지 않는다** (`graph_augmentation.py::_fact_recall_leg`).
   published/latest 태그가 없는 노드는 retrieve tool이 조용히 버린다. → code decision의 query 경로는
   반드시 **direct `kumiho.search` + kind/context filter** (fact-recall leg와 동일 패턴). 태그를 붙여
   retrieve tool에 노출시키는 방식은 채택하지 않는다 (대화 recall 경로 오염 = criterion 1 위반).
2. **Typed-node vector crowding** (ON 게이트 블로커로 실측된 사고): typed 임베딩이 같은 project의
   vector k=10을 점령해 대화 recall을 밀어냈다. → **code 도메인은 별도 project로 물리 격리.**
   env 게이트나 kind filter만으로는 vector 인덱스 공유를 못 막는다.

   **[판정 반영]** B안의 "같은 project + `code-*` space 접두" 배치는 기각한다. B가 근거로 든
   "대화 recall은 space-scoped retrieve"는 실코드와 불일치 — 기본 `tool_memory_retrieve`
   (`mcp_server.py:1083`)는 **project-스코프 + kind 필터**다. 서버 크라우딩 수정이 kind-제외-목록
   방식이라면 신규 `code_decision` kind는 목록 밖이라 크라우딩이 재발하고, 그 수정은 조용한 서버
   변경 요구가 된다(과거 ON-게이트 블로커와 같은 사고 클래스). 별도 project는 서버 수정의 내부
   메커니즘과 무관하게 안전하며, 435개 기존 테스트를 구조적으로 보장한다.
3. **entity anchor 허브 패턴** (`entity_promotion.py`): 결정적 slug + get-or-create + 단일 anchor
   revision + 모든 edge가 anchor를 향함. → git anchor는 이 패턴의 직계 재사용.
4. **`create_edge`의 edge_type은 자유 문자열** (UPPERCASE+digits+`_`, ≤50자 —
   `kumiho/edge.py::validate_edge_type`). `MOTIVATED_BY`/`IMPLEMENTED_IN`은 서버 변경 0으로 사용 가능.
5. **임베딩 메커니즘**: `item.create_revision(metadata, number)`에는 `embedding_text` 파라미터가
   **없다** (`item.py:151`). embedding_text 미지정 시 서버는 **모든 metadata를 연결해 자동
   임베딩한다** (`client.py:961-964`) — anchors_json/전체 경로/해시가 임베딩에 유입돼 조성이
   노이즈화된다.

   **[판정 반영]** (A의 fatal: 모듈-레벨 `kumiho.create_revision`은 실존하지 않음 / B·C의 fatal:
   "summary가 임베딩 본문"이라는 전제 오류) — 해소: **공개 액세서 `kumiho.get_client()`** 를 통해
   `get_client().create_revision(item.kref, metadata, embedding_text=...)`를 호출한다.
   `client.py:954`의 시그니처에 `embedding_text` 파라미터가 실존하고, `sdk.get_client()`는
   `dream_state.py:846`/`_graph_walk.py:78`에서 이미 쓰는 코드베이스 관례다 (private `_Client`
   직접 참조 아님). 이 호출을 `code_decisions.py`의 단일 헬퍼
   `_write_revision(item, metadata, embedding_text)`로 감싸 write 경로 전체가 공유한다.
   §1.6의 embedding_text 조성은 이 경로가 있어야만 성립한다 — 채택하는 설계가 무엇이든 필수.
6. `kumiho.search`는 metadata 필터가 없다 (`context_filter`, `kind_filter`만 — `client.py::search`).
   → file→decision 역조회는 metadata 검색이 아니라 **결정적 slug의 anchor 노드 + 엣지 역순회**.
7. LLM은 `LLMAdapter.chat(messages, model, system, max_tokens, json_mode=schema)` —
   summarizer의 어댑터/`light_model` 재사용, 신규 의존성 0.
8. 서버 `FULLTEXT_EXCLUDED_KINDS`는 명시적 목록이므로 신규 kind는 기본 **포함** —
   `code_decision`을 렉시컬 검색 가능하게 두는 것이 서버 무변경으로 달성된다 (1급 콘텐츠).

---

## 1. Node taxonomy

### 1.1 프로젝트/스페이스 레이아웃 — 전용 project (A·C 합의)

```
project: "{agent_project}-code"      # 기본값. CodeMemoryConfig.project / env로 오버라이드
├── /decisions      ← code_decision (1차 콘텐츠, 렉시컬+벡터 검색 대상)
├── /anchors        ← code_anchor   (파일 허브, waypoint 전용)
├── /commits        ← code_commit   (provenance + 멱등 마커)
└── /evidence       ← code_evidence (근거 원자)
```

- **별도 project인 이유**: §0-2. 대안(같은 project + kind 제외 서버 수정)은 서버 변경 요구로 기각.
- **멀티 repo**: Phase 1은 spaces를 repo별로 쪼개지 않는다. repo 정체성은 anchor slug와
  metadata `repo` 키에 인코딩, repo 필터는 클라이언트 사이드.

### 1.2 기존 대화-`decision`과의 화해

기존 ontology의 `decision` kind(대화 유래, `/{project}/decisions`)는 1바이트도 건드리지 않는다.
코드 결정은 **다른 kind(`code_decision`) + 다른 project**로 이중 네임스페이스 — kind_filter 검색과
context_filter 검색 어느 쪽으로도 섞이지 않는다. provenance 모델 자체가 다르다(대화 결정 = "사용자가
말한 것", 코드 결정 = "커밋이 증언하는 것"). 대화↔코드 브리지는 Phase 2 (§8).

### 1.3 `code_decision` (kind: `code_decision`)

**identity** — **[판정 반영]** 두 fatal의 동시 해소:
(i) A의 title-only slug는 반복 문구 리포("fix(memory): ...")에서 무관한 결정 둘이 같은 slug로
조용히 병합될 수 있고, (ii) B·C의 hash12-in-slug는 rebase/squash 후 재채굴 시 전량 중복된다.
해소: **sha-free + 시대 구분자**:

```python
slug = slugify(f"{title} {author_date:%Y%m%d}", hash_on_truncate=True)
```

- **author date는 rebase/squash에서 보존**되므로 rewrite 후 재채굴이 같은 slug로 수렴한다
  (sha-free — A의 §3.4 생존표 성질 유지).
- 날짜 접미가 "다른 시대의 같은 제목" 병합을 차단한다. 같은 날 동일 제목의 무관한 결정이라는
  잔여 케이스는 get-or-create 시 **충돌 가드**로 방어: 기존 item의 `decision` metadata와 신규
  decision 텍스트의 token Jaccard < 0.3이면 별개 결정으로 판정하고 `-2` 접미로 재시도.

**revision metadata** (전부 str — 서버 metadata는 Dict[str, str]):

| key | 내용 |
|---|---|
| `title` | 결정 한 줄 (≤80자, `_title_of` 관례) |
| `summary` | decision + rationale 합성 (graph reader 표시용) |
| `decision` | 무엇을 하기로 했나 |
| `rationale` | 왜 (LLM이 커밋 메시지/디프/주석에서 추출) |
| `why_question` | 이 결정이 답하는 질문의 자연어 형태 |
| `symbols` | diff hunk 헤더/메시지에서 추출한 식별자, 쉼표 조인 (C 이식) |
| `repo` | repo 식별자 (origin URL slug 또는 dir명) |
| `commit_hash` | 대표 커밋 full sha (표시·필터용 — identity 아님) |
| `files` | 앵커 파일 경로 쉼표 조인 (`_search_text` 렉시컬 집계용) |
| `line_ranges` | `path:start-end` 세미콜론 조인 (표시용; 권위는 edge metadata) |
| `author`, `decided_at` | git author / author-date ISO |
| `confidence` | `high\|medium\|low` |
| `status` | `active\|superseded` — SUPERSEDES 기록 시 구 결정에 갱신 (C 이식) |
| `schema_version` | `"kumiho.code_memory.v1"` |

### 1.4 `code_anchor` (kind: `code_anchor`)

entity anchor 패턴의 직계. **identity = (repo, file path)** — commit이 아니다. 휘발성 정보
(commit, line range)는 전부 edge metadata로 (§3).

- `slug = slugify(f"{repo}::{norm_path}", hash_on_truncate=True)` — 결정적, 역조회의 키.
- metadata: `{repo, path, display_name}`만. 콘텐츠 없음, 검색 결과로 반환하지 않음.
- anchor revision 1개, 모든 IMPLEMENTED_IN edge가 이 rev를 향함. get-or-create + per-slug lock은
  entity_promotion과 같은 패턴의 지역 구현 (모듈 의존 엣지를 만들지 않기 위해 — modular boundary).
- 디렉토리 앵커는 Phase 1에 만들지 않는다.

### 1.5 `code_commit` / `code_evidence`

**`code_commit`** — provenance + **멱등 마커** 겸용:
- `slug = slugify(f"{repo}-{hash12}")`. metadata: `{repo, hash, subject, author, committed_at,
  decisions_count, capture_version}`.
- embedding_text: `subject` 한 줄만.
- **커밋 처리 완료 후 마지막에 생성** (§4.6) — 재실행 시 존재 = skip, 부분 실패 커밋 자동 재시도.

**`code_evidence`** — 근거 원자 (도그푸드 (a) "adversarial-review evidence", (c) "displacement
measurement"가 이 노드로 서피스):
- LLM이 커밋당 0..N개: `{statement, evidence_kind}` (kind ∈ `measurement | review_finding |
  incident | benchmark | constraint`).
- **statement는 커밋 메시지/diff 주석에서 verbatim 인용** (B 이식) — 환각 억제 + 도그푸드 판정을
  문자열 부분매치로 기계화. `source_ref` 필수 (`commit:cfec845` | `comment:file:line`) (C 이식).
- `slug = slugify(statement, hash_on_truncate=True)` — 재채굴 수렴.
- embedding_text = statement 그대로 (fact 노드의 승리 공식: answer-shaped atomic claim).
- 기존 `fact` kind 재사용 안 함: 대화 도메인 스키마·recall leg에 물려 있고 evidence_kind 축이 없다.

### 1.6 embedding_text 조성 (§0-5의 클라이언트 경로로 주입)

```
{why_question} {decision}. Rationale: {rationale}.
Anchored: {file basenames} ({symbols}). Commit: "{commit subject}".
```

- **why_question 선두** (A — doc2query 역방향): 사용자 질의는 거의 항상 why-형. 질의-문서 유사도를
  문서 쪽에서 미리 접합. LLM 호출 1회 안에서 공짜.
- **symbols + 파일 basename + 커밋 subject 포함** (C): `rerank_async`, `KUMIHO_MEMORY_ONTOLOGY` 같은
  식별자가 도그푸드 질의의 핵심 신호. basename만 넣고 전체 경로는 제외 (경로 노이즈로 벡터 오염 방지 — B).
- 영어로 통일 (Korean tokenizer가 혼합 텍스트에서 식별자를 파편화하는 서버 이슈 — PR#20 실측).
- `code_decision`은 서버 `FULLTEXT_EXCLUDED_KINDS`에 넣지 않는다 — 1차 콘텐츠 (§0-8).

---

## 2. Edge taxonomy

| edge | 방향 | 신규? | metadata | 의미 |
|---|---|---|---|---|
| `IMPLEMENTED_IN` | code_decision → code_anchor | 신규 (자유 문자열, 서버 무변경) | `{commit_hash, line_start, line_end, role}` | 결정이 이 파일(의 이 범위)에 구현됨 |
| `MOTIVATED_BY` | code_decision → code_evidence | 신규 | `{commit_hash}` | 결정의 근거 |
| `DERIVED_FROM` | code_decision → code_commit / code_evidence → code_commit | **재사용** (ontology provenance 관례) | — | provenance |
| `SUPERSEDES` | code_decision(신) → code_decision(구) | **재사용** (relations.py belief-update) | `{reason, overlap}` | 신 결정이 구 결정을 대체 |

- 재사용 원칙: 의미가 기존과 정확히 같으면 재사용, 새로우면 새 문자열. 서버 변경 0 확인됨.
- `role` ∈ `primary | touched` (A) — 다파일 커밋에서 본체/부수 파일 구분.
- 대화 recall의 `GraphAugmentationConfig.edge_types` 기본셋에 MOTIVATED_BY/IMPLEMENTED_IN이 없고,
  대화 recall의 시드가 code project에 도달할 경로 자체가 없으므로 격리는 공짜 (B의 검증 재사용).
- **엣지 멱등**: 서버가 (src,dst,type) 중복을 dedupe한다고 **가정하지 않는다** — 쓰기 전
  `get_edges` 존재 확인 1회 (C 이식; A의 open question을 안전 쪽으로 확정).

### 2.1 SUPERSEDES — anchor-scoped 후보 + 3-신호 합류

**[판정 반영]** (A의 fatal: 시간 순서 조건 부재로 ingest 순서에 따라 방향이 뒤집힐 수 있고, 누적
튜닝-시리즈가 위양성 후보) — B의 3-신호 합류를 A의 anchor-scoped 후보 발견 위에 얹어 해소:

1. **후보 발견 (A)**: corpus-global fulltext가 아니라 **같은 anchor의 INCOMING IMPLEMENTED_IN
   엣지 목록** — 결정적, BM25 corpus 위생 무의존. LLM `supersedes_hint`(§4.5)가 있으면 그 서술로
   후보를 좁힌다.
2. **판정 (B)**: 아래 **세 조건 전부** 만족 시에만 링크:
   (a) 앵커 파일 교집합 ≥ 1 (후보 발견에서 구조적으로 보장),
   (b) title+decision 토큰 Jaccard ≥ **0.35 (hint 있을 때) / 0.5 (blind)** —
       `relations._jaccard`/`_tokens` import 재사용,
   (c) **구 결정의 decided_at < 신 결정의 decided_at** (author-date) — ingest 순서와 무관하게
       방향 보장.
3. 누적 튜닝-시리즈(같은 파일 4연작 커밋) 방어: 시리즈 커밋은 hint가 나오지 않으므로 blind 임계
   0.5가 적용되고, `reason` metadata에 `reversal | belief update`를 기록해 v2의 REFINES 분리 여지를
   남긴다 (B risk #5의 결론 채택).
4. 링크 성립 시 구 결정의 `status`를 `superseded`로 갱신 (C — §5.4의 강등에 사용).

---

## 3. Anchor model 상세

### 3.1 인코딩 요약

```
결정 → 파일:   IMPLEMENTED_IN edge, target = code_anchor(repo, path),
               edge.metadata = {commit_hash: "cfec845…", line_start: "118", line_end: "142", role: "primary"}
파일 identity:  slug = slugify("{repo}::{norm_path}")   ← 역조회 O(1) 키
커밋 휘발분:    전부 edge/decision metadata              ← rewrite 시 노드 identity 무손상
```

### 3.2 경로 정규화 — 단일 소스 계약

`normalize_path()`를 `code_decisions.py`(스키마 모듈)에 **단일 소스**로 두고 write·query 양쪽이
공유한다 (C 이식): repo-루트 상대화, `\` → `/`, 선행 `./` 제거.

**[판정 반영]** (C의 fatal: 자칭 최대 리스크였던 win32 대소문자 casefold-재시도는 오진단) —
`slugify`는 **이미 casefold한다** (`_text.py:21`). 대소문자 차이는 slug 수준에서 자동 수렴하므로
C가 제안한 casefold-재시도 폴백은 죽은 코드가 된다 — 채택하지 않는다. normalize_path는 구분자·상대화만
책임진다. (같은 디렉토리에 대소문자만 다른 두 파일이 공존하는 병리 케이스는 hash_on_truncate가
아니라 casefold로 인한 slug 충돌인데, win32/NTFS에서는 애초에 공존 불가 — 리스크 표에서 제외.)

### 3.3 역조회 (anchor → decisions) — 검색 0회

1. `slug = slugify(f"{repo}::{normalize_path(file)}")`
2. `project.get_item(slug, "code_anchor", parent_path=".../anchors")` — miss = "기록된 결정 없음"
   확정 (fuzzy 폴백 없음 — semantic leg가 별도로 있으므로 오염 금지).
3. `anchor_rev.get_edges(edge_type_filter="IMPLEMENTED_IN", direction=INCOMING)` → source가 결정.
4. line 인자가 있으면 edge metadata `[line_start, line_end]` (±slack 20)와 교차 판정.
   **라인 범위는 필터가 아니라 부스트** (3안 만장일치 합의): 교차 hit는 상위 정렬, miss여도 같은
   파일이면 포함 — 라인은 가장 먼저 썩는 정보이므로 hard filter는 위양성 누락을 낳는다.
5. `commit` 인자는 edge/decision metadata의 commit_hash prefix 매치 부스트 (동률 타이브레이크).

### 3.4 다파일 커밋 + 앵커 검증

- LLM이 결정별 anchor를 명시 배정: `primary` 1..3 + `touched` 0..N, **결정당 상한 8개**
  (**[판정 반영]** C의 fatal "앵커 무상한 → 메가-리팩터 커밋이 수십 파일 오염" 해소 — A·B의 캡 채택).
- **[판정 반영]** (A의 fatal: LLM 앵커 환각 무방어 → 유령 anchor 허브가 역조회를 영구 오염) —
  B의 검증 이식: **anchors.file은 해당 커밋의 실제 changed-file 목록(`--stat` ground truth)에 대해
  검증**, 목록에 없는 파일은 그 앵커만 드랍(결정은 유지)하고 stat 기반 파일-수준 앵커로 폴백.

### 3.5 History rewrite 생존표 (A §3.4 — 필수 설계 산출물)

| 사건 | 생존? | 근거 |
|---|---|---|
| rebase/squash로 sha 전면 교체 | 노드·엣지 전부 생존, 재채굴 수렴 | identity에 sha가 없음 (decision slug는 title+author-date — §1.3). edge metadata의 sha는 "당시 증거 좌표"로 잔존 |
| 파일 리네임 | 구 anchor 잔존 + 신 anchor 생성 | blame-follow는 non-goal. 구 결정은 "이 경로였던 시절의 결정"으로 정직 강등 |
| 라인 이동 | anchor 레벨 recall 무손상 | 라인은 boost-only (§3.3) |
| 파일 삭제 | anchor 잔존 | "왜 X를 지웠나"도 결정이다 |

**[판정 반영]** (B의 fatal: sha-in-slug + sha-키 마커로 rewrite 후 재인제스트 시 전량 중복 + 중복끼리
SUPERSEDES 오링크로 가짜 belief-update 날조) — §1.3의 sha-free identity로 해소. 마커(`code_commit`)는
sha-키가 맞지만 마커 중복은 무해(스킵 판단용일 뿐)하고, 결정 노드가 수렴하므로 SUPERSEDES 날조가
성립하지 않는다 (Jaccard 후보가 자기 자신 slug로 수렴 → self-link 방지 조건으로 차단).

---

## 4. Capture adapter v1

### 4.1 파이프라인 (A 골격 + B 스테이지)

```
[1] enumerate   git log --format=%H%x00%an%x00%aI%x00%s%x00%b {range}   (subprocess, 신규 dep 0)
[2] prefilter   결정적 휴리스틱 — LLM 비용 0으로 노이즈 30-50% 컷 (B)
[3] evidence    커밋별 diff 요약 패킷 조립 (토큰 예산 내, 결정적)
[4] structure   LLM 배치 콜 (6 커밋/콜, json_mode 스키마)
[5] validate    앵커↔changed-file 검증(§3.4) + confidence 컷 (B)
[6] write       get-or-create 노드 + 엣지 (run_bounded_in_thread, embedding_text 주입 §0-5)
[7] supersede   anchor-scoped 3-신호 SUPERSEDES 패스 (§2.1)
[8] marker      code_commit 노드 생성 = 커밋 완료 마커 (마지막!)
```

**[판정 반영]** (C의 fatal: commit 원장을 **먼저** 쓰는 순서 → 중간 크래시 시 커밋이 mined로 찍혀
재실행이 영구 스킵, 결정 소실) — 마커는 **[8] 마지막** (A·B 합의). 부분 실패 커밋은 재실행 시
자동 재시도된다.

**[판정 반영]** (C의 fatal: 프리필터 없음 + 커밋당 1 LLM 콜 = 30콜 $0.1–0.4, A·B의 ~10배) —
B의 프리필터 + 6커밋/콜 배치 채택. C가 배칭 대비 우려한 앵커 정밀도는 배치 프롬프트에 커밋별
구획을 명시하고 changed-file 검증(§3.4)이 잡는다.

### 4.2 프리필터 (B — 결정적, false-negative 비대칭 원칙)

확실한 노이즈만 자른다 — 애매하면 통과시켜 LLM이 판단:
- merge인데 body 없음 (**"parents≥2 AND body 비어있음"만 skip** — 스쿼시-머지 워크플로에서
  body 있는 머지는 유일한 rationale 캐리어이므로 통과)
- diff가 lockfile/생성물뿐 (denylist glob: `package-lock.json`, `*.pb.go`, `__pycache__` 등)
- subject가 순수 버전 범프 패턴 **이고** body 없음
- subject ≤ 3단어 **이고** body 없음 **이고** diff < 5줄

**통과 기준이 아닌 것**: conventional-commit 타입. `chore:`도 결정을 실을 수 있다
(실례: `a35db47 chore(memory): align __version__ ... + note reformulate-draws knob`).

### 4.3 diff 요약 패킷 (커밋당 예산 ~1,800 tokens, 결정적)

message-first, diff-as-evidence (이 리포는 rationale이 body에 사는 표본):
- subject + body: 사실상 무제한 / `--stat` 파일 목록: 전체 (앵커 검증 ground truth)
- diff: 파일당 40줄 × 최대 6파일, 커밋 상한 4,000 chars
- 절삭 우선순위 (B): (1) hunk 헤더 + 함수 시그니처 항상 보존, (2) 추가 > 삭제,
  (3) **주석/도크스트링 변경 라인 우선 보존** — rationale이 주석에 사는 코드베이스다.
  절삭 시 `[...truncated N lines]` 마커로 고지.
- **코드 주석은 1급 rationale 소스** (C): `recall_rerank.py`의 "ONE worker on purpose" 주석이
  도그푸드 (a)의 정답 그 자체다.

### 4.4 LLM structuring — decision-worthy 정의 (B, 이 리포 실커밋으로 캘리브레이션)

배치 6커밋/콜, `summarizer.adapter` + `light_model` 재사용, json_mode strict 스키마:

```json
{ "commits": [ { "hash": "...", "decisions": [ {
    "title": "...", "decision": "...", "rationale": "...", "why_question": "...",
    "symbols": ["rerank_async", "_RERANK_EXECUTOR"],
    "evidence": [ {"kind": "measurement|review_finding|incident|benchmark|constraint",
                   "text": "커밋 메시지/주석에서 verbatim 인용", "source_ref": "commit:<sha>"} ],
    "anchors": [ {"file": "...", "line_start": 118, "line_end": 142, "role": "primary"} ],
    "supersedes_hint": "이 결정이 뒤집는 과거 결정의 서술 (없으면 빈 문자열)",
    "confidence": "high|medium|low"
} ] } ] }
```

프롬프트 핵심 (B의 4-분류 정의):
> 결정이란: (a) 대안 중 선택 — "X 대신 Y" / (b) 기본값·정책 설정 — "default ON, opt-out" /
> (c) 이전 행동의 반전 / (d) 측정에 근거한 트레이드오프.
> 결정이 아닌 것: 무엇이 바뀌었는지의 재서술(그건 git이 이미 안다), 기계적 리네임, 버그의 존재
> 자체. **0개 결정도 정답이다. 지어내지 마라.** 커밋당 0–3개.
> evidence.text는 반드시 verbatim 인용. anchors.file은 제공된 changed-file 목록에서만.

검증 (stage 5): 환각 앵커 드랍+stat 폴백 (§3.4), `confidence==low && evidence 비어있음` → 결정 드랍.

### 4.5 진입점 (one command) + 증분

- API: `await ingest_repo(repo_path=".", rev_range=None, config=None) -> IngestStats`
- MCP: `kumiho_code_ingest {repo_path, rev_range?, max_commits?}`
- CLI: `python -m kumiho_memory code-ingest --range HEAD~30..HEAD`
- **`rev_range` 생략 시 증분 모드** (B): `git log` 열거 후 `code_commit` 마커 있는 커밋 스킵 —
  별도 커서 상태 없음, 그래프 자체가 장부.

### 4.6 멱등성 (3중) + `--force` 재캡처

1. **커밋 레벨**: `code_commit` 존재 = skip. 마커는 [8] 마지막에 쓰므로 부분 실패는 재시도.
2. **노드 레벨**: 전부 get-or-create (ALREADY_EXISTS 흡수). slug가 sha-free라 rewrite 후에도 수렴.
3. **엣지 레벨**: 쓰기 전 `get_edges` 존재 확인 (§2 — dedupe 가정 안 함).

**[판정 반영]** (C의 fatal: `mined_by` 버전-스탬프 재채굴이 구 노드를 deprecate하지 않아 프롬프트
업그레이드마다 고아 중복 세대 발생) — B의 `--force` 흐름 채택: force 재처리 시 해당 커밋의 기존
`DERIVED_FROM` 결정들을 `deprecate_item` 처리한 뒤 새로 쓴다.

**[Phase 2 구현 완료, 2026-07-11]** Phase 1 리뷰의 "클라이언트 API 부재" 판정은 과했다 —
`Item.set_deprecated(True)`가 아이템 레벨에 실존한다. `--force`는 이제 원안대로
**deprecate-then-rewrite**로 동작한다: (1) 프리패스가 해당 커밋 마커의 INCOMING `DERIVED_FROM`
소스 중 **결정 노드만** deprecate (evidence 원자는 verbatim-slug로 커밋 간 공유되므로 제외) +
해당 리비전에 `status=deprecated`를 in-place 기록 (검색 필터 전파 전에도 쿼리 랭킹이 강등);
(2) 재채굴이 같은 slug로 수렴하면 `set_deprecated(False)`로 복원하고 **항상 새 리비전**을 쓴다
(재캡처의 목적은 낡은 추출 내용의 교체다); (3) 마커도 갱신된 decisions_count/capture_version으로
새 리비전을 쓴다. 쿼리의 active-티어 강등은 `superseded`와 `deprecated`를 동일하게 취급한다.

### 4.7 비용 모델 (30 커밋)

| 항목 | 추정 |
|---|---|
| 프리필터 후 잔존 | ~20–24 커밋 (이 리포 최근 35 커밋 실측: merge 9, bump/chore 3) |
| LLM 콜 | 4–5 (6커밋/배치, asyncio.gather 동시 3) |
| 토큰 | ~40–48k in / ~7k out |
| **비용** | **$0.01–0.04** (light_model 기준) |
| 쓰기 RPC | ~150–250, 로컬 CE ~10–20s |
| **총 소요** | < 90s |

capture는 사용자가 명시 호출하는 포그라운드 작업 — 통계 dict 반환 + 실패 커밋 목록 보고
(consolidation처럼 조용히 삼키지 않음).

---

## 5. Query API

### 5.1 매니저/모듈 API

```python
async def why(
    query: Optional[str] = None,      # 자연어 질문 (없으면 anchor-only)
    *,
    file: Optional[str] = None,       # repo-상대 경로
    line: Optional[int] = None,
    commit: Optional[str] = None,
    repo: Optional[str] = None,
    limit: int = 5,
) -> WhyResult   # {"decisions": [DecisionAnswer...], "context": str}
```

`DecisionAnswer` = `{kref, title, decision, rationale, why_question, confidence, status,
anchors: [{file, lines, commit}], evidence: [{statement, kind, source_ref}],
commits: [{sha, subject}], supersedes: [...], superseded_by: Optional[...],
match: "anchor+line"|"anchor"|"semantic", score}`.

manager 통합은 위임 한 줄 (env 게이트 안에서만 lazy import 바인딩) — 대화 경로 diff 0줄.

### 5.2 3-leg 검색 (A) + 사전식 융합 (C)

1. **anchor leg (결정적)** — file 인자가 있을 때: §3.3 역조회. 검색 인프라 완전 우회.
2. **semantic leg** — query가 있을 때: `kumiho.search(query, context="{code_project}/decisions",
   kind="code_decision", include_revision_metadata=True)` 직접 호출 (§0-1).
3. **evidence-bridge leg** (A) — 같은 질의로 evidence space 1회 검색, hit의 INCOMING
   `MOTIVATED_BY`를 타고 결정으로 승격 — 질문이 결정문보다 **측정문에 가까운** 케이스
   ("displacement measurement" 류, 도그푸드 (c))를 잡는다.

**융합 — [판정 반영]** (A의 fatal: anchor leg 내부 rerank 부재 → 허브 파일(`memory_manager.py`처럼
수십 결정이 앵커된 파일)에서 semantic top-k가 빗나가면 정답을 끌어올릴 방법이 없음 / B의 fatal 동일)
— C의 **사전식 정렬 키** 채택, 레그 간 raw score 가산 금지:

```
sort key = (anchor_line_hit, anchor_hit, ce_score)     # 사전식
```

- anchor 매치는 "이 파일에 대한 결정"이라는 사실 증거, CE는 확률 신호 — 층위를 섞지 않는다
  (fact-leg의 축-분리 교훈을 구조 prior로 승격).
- **CE 재랭크**: 후보 ≥ 2이고 question이 있으면 cross-encoder로 `question` vs
  `title + summary` 스코어링. **[판정 반영]** (판정단 지적: C의 "rerank_async 재사용 = 신규 코드
  0줄"은 memory-dict 셰이핑과 recency/MMR 의미론이 끌려옴을 간과) — `rerank_async` 전체가 아니라
  `recall_rerank`의 **하위 프리미티브만 재사용**: `try_fastembed_reranker()`/`resolve_reranker_from_env()`
  로 얻은 reranker callable을 `_rerank_executor()`(단일-워커 offload 경로, cfec845의 결정 그 자체)
  위에서 실행하는 얇은 `_ce_scores(question, texts)` 헬퍼를 `code_query.py`에 둔다. recency/MMR/
  memory-dict 의미론은 유입되지 않는다.
- CE 미설정 시: ce_score 자리는 semantic leg 내부에서만 서버 점수, anchor leg는 decided_at 최신순.
- anchor-hit ∩ semantic-hit(양쪽 leg 보증)는 같은 티어 안에서 추가 타이브레이크 부스트.

**[판정 반영]** (A의 fatal: superseded 결정이 강등 없이 최상위로 나갈 수 있음) — C 채택:
`status == superseded`인 결정은 같은 티어 내 **최하위로 강등**하고 `superseded_by`를 항상 채운다 —
에이전트가 번복된 결정을 정답으로 받지 않는 것이 안전성의 핵심.

### 5.3 증거 체인 전개

limit 컷 이후 결정별 1-hop `get_edges` (결정적, LLM 무개입):
- `MOTIVATED_BY` → evidence (verbatim statement + kind + source_ref, 최대 6)
- `DERIVED_FROM` → commit ({sha, subject, date})
- `SUPERSEDES` 양방향: OUT = 이 결정이 대체한 과거 결정, IN = `superseded_by` 채움.
- fan-out 가드: 결정당 edge 스캔 ≤ 32, 전체 fetch ≤ limit×3. `run_bounded_in_thread` 바운딩,
  실패 시 decision만 반환 (graceful).

### 5.4 `compose_why_context` (C 이식 — 즉시 사용 가능한 렌더)

구조화 JSON과 **함께** 마크다운 블록 반환 (대화용 compose_context는 재사용하지 않음 —
revision-경쟁/sibling 모델이 이 도메인과 무관). 순수 함수:

```markdown
### [D1] Offload the fastembed cross-encoder rerank off the event loop  (cfec845, 2026-07-10)
files: kumiho_memory/recall_rerank.py:413-430
decision: Run CE reranks on a dedicated single-worker ThreadPoolExecutor.
why: Single worker keeps inferences serialized while freeing the event loop.
evidence:
- (measurement) "inline CE collapsed a concurrency-4 harness to ~1 effective"  [commit:cfec845]
supersedes: none / superseded_by: none
```

char_limit 기본 4000, 초과 시 하위 decision부터 절단 (additive 원칙 — evidence는 자기 블록 안에서만
소비, 다른 decision을 밀어내지 않음).

### 5.5 MCP tools (2개, mcp_tools.py 관례)

```json
{ "name": "kumiho_code_why",
  "description": "Ask why code is the way it is — recall captured decisions anchored to a file/line/commit, with their evidence chain.",
  "inputSchema": { "type": "object", "properties": {
      "question": {"type": "string"},
      "file":     {"type": "string", "description": "repo-relative path (forward slashes)"},
      "line":     {"type": "integer"},
      "commit":   {"type": "string"},
      "repo":     {"type": "string"},
      "limit":    {"type": "integer", "default": 5}},
    "anyOf": [{"required": ["file"]}, {"required": ["question"]}] } }

{ "name": "kumiho_code_ingest",
  "description": "Mine a git commit range into decision nodes (LLM-structured, idempotent).",
  "inputSchema": { "type": "object", "properties": {
      "repo_path":  {"type": "string", "default": "."},
      "rev_range":  {"type": "string", "description": "e.g. HEAD~30..HEAD; omit = incremental"},
      "max_commits":{"type": "integer", "default": 50}},
    "required": ["repo_path"] } }
```

- `anyOf`로 file|question 최소 입력 강제 (C).
- 핸들러는 기존 관례대로 sync + 내부 `asyncio.run`. `KUMIHO_MEMORY_DECISIONS=1`일 때만 등록
  (기본 OFF — opt-in; 대화 도메인의 default-ON과 달리 아직 paired evidence가 없다.
  `KUMIHO_MEMORY_ONTOLOGY`와 완전 독립 — LoCoMo 게이트 0 영향이 릴리즈 안전핀).

---

## 6. Module layout / config / env

**[판정 반영]** (판정단: B의 flat-module 레이아웃이 기존 스타일과 가장 정합 — `entity_promotion.py`가
바운디드 모듈의 표본) — 서브패키지 대신 flat 3파일:

```
kumiho_memory/
  code_decisions.py     # 스키마 상수 + CodeMemoryConfig + normalize_path/slug 함수(단일 소스)
                        #   + _write_revision(embedding_text 주입 헬퍼, §0-5) + get-or-create 헬퍼
  code_capture.py       # 파이프라인 §4 전부 (git subprocess, prefilter, packet, LLM, validate,
                        #   write, supersede, marker)
  code_query.py         # 3-leg why 엔진 + 사전식 융합 + _ce_scores + evidence chain
                        #   + compose_why_context
  mcp_tools.py          # +2 tools (기존 파일 append, 게이트 내)
  memory_manager.py     # +얇은 위임 메서드 (lazy import; 기존 경로 무접촉)
  __main__.py           # +code-ingest 서브커맨드
```

```python
@dataclass
class CodeMemoryConfig:
    project: str = ""                 # 기본: f"{memory_project}-decisions" 파생
    repo: str = ""                    # 기본: origin URL slug, 없으면 dir명
    decisions_space: str = "decisions"
    anchors_space: str = "anchors"
    commits_space: str = "commits"
    evidence_space: str = "evidence"
    llm_batch_size: int = 6
    max_commits: int = 50
    per_commit_diff_chars: int = 4000
    per_file_diff_lines: int = 40
    max_decisions_per_commit: int = 4
    max_anchors_per_decision: int = 8
    max_evidence_per_decision: int = 6
    line_slack: int = 20
    supersede_jaccard_hinted: float = 0.35
    supersede_jaccard_blind: float = 0.5
    drop_low_confidence_without_evidence: bool = True
    write_timeout: float = 60.0
    schema_version: str = "kumiho.code_memory.v1"
```

- env: `KUMIHO_MEMORY_DECISIONS` (기본 unset=off), `KUMIHO_MEMORY_DECISIONS_PROJECT` 오버라이드.
  레거시 `KUMIHO_MEMORY_CODE*` 이름은 폐지 예정이나 폴백으로 계속 인식(신규 이름 우선).
- **의존성 신규 0** — git은 subprocess (GitPython 불채택), LLM은 기존 어댑터.
  `[code]` optional-extra는 만들지 않는다 (빈 extra는 소음 — 이슈의 extra 항목에 대한 답:
  불필요 확인).
- 기존 모듈 수정: `mcp_tools.py`(+2 tool), `__main__.py`(+1 서브커맨드), `memory_manager.py`
  (+게이트 안 위임). ontology/relations/graph_augmentation **무수정** (relations는
  `_jaccard`/`_tokens` import만, recall_rerank는 executor/reranker 프리미티브 import만).

---

## 7. Test plan

### 7.1 Unit (신규 ~30개; LLM·서버 무접속. 기존 ~435 전부 무접촉 통과가 게이트)

- `test_code_capture.py`
  - **프리필터**: merge/lockfile/bump/typo skip, body 있는 chore 통과, body 있는 스쿼시-머지 통과
  - **패킷**: diff 절삭 상한, hunk-헤더/주석 라인 보존, changed-file 목록 —
    `tmp_path`에 실제 `git init` 합성 리포 (subprocess라 진짜 git으로)
  - **structuring**: canned-JSON 스텁 LLMAdapter → 노드/엣지 수 검증 (SDK는 fake);
    **embedding_text가 `get_client().create_revision`에 전달되는지 assert** (§0-5 경로 고정)
  - **검증**: 환각 앵커 드랍 + stat 폴백, low-confidence+무증거 드랍, 앵커 상한 8
  - **멱등**: 같은 range 2회 = 2회차 LLM 콜 0 + 노드 수 불변; **부분 실패 커밋 재시도**
    (마커가 마지막에 쓰임을 크래시 주입으로 검증); `--force` = deprecate 후 재작성
  - **slug**: title+date 결정성, 같은 날 동일 title 무관 결정의 충돌 가드(-2 접미),
    rewrite 시뮬레이션(sha만 바뀐 재채굴) → 수렴
  - **SUPERSEDES**: 3-신호 각각 단독으로는 미링크, 합류 시 링크, 시간 역행 미링크,
    self-link 차단, 구 결정 status 갱신
- `test_code_query.py`
  - normalize_path (백슬래시/`./`/절대경로, 한글 경로), anchor slug 왕복, get_item NOT_FOUND →
    빈 결과 (검색 폴백 없음)
  - line boost (경계/슬랙/빈 범위 — 필터 아님), commit prefix 타이브레이크
  - **사전식 융합**: anchor_line_hit > anchor_hit > ce_score 순위가 CE 유/무 양쪽에서 유지;
    superseded 강등 + superseded_by 채움; evidence-bridge 승격; fan-out 가드
  - compose_why_context char_limit 절단; "file만/question만/둘 다" 3모드
- `test_code_isolation.py` (B 이식 — 격리를 테스트로 증명)
  - manager를 게이트 on/off로 만들고 `recall()`/`store_conversation()` 경로의 **호출 그래프가
    byte-identical**함을 assert (paired-측정 원칙의 유닛판)
- MCP: 게이트 OFF 시 툴 미등록, ON 시 스키마 validate + anyOf 강제

### 7.2 Live dogfood gate (127.0.0.1:9190 CE — 성공 기준 5)

`python/kumiho-memory/scripts/dogfood_code_memory.py` (수동 실행, CI 제외):

```python
os.environ["KUMIHO_MEMORY_DECISIONS"] = "1"
# 0) Paid-run preflight 원칙: 1커밋 드라이런으로 LLM 발화·JSON 준수 확인 후 발사
# 1) 전용 project get-or-create (끝나면 delete_project로 잔여물 0 — SmokeTest 관례)
stats = await ingest_repo("G:/git/KumihoIO/kumiho-SDKs", "HEAD~30..HEAD")
assert stats.decisions >= 10
CASES = [
    (dict(question="why is rerank_async a single-worker executor?",
          file="python/kumiho-memory/kumiho_memory/recall_rerank.py", line=420),
     expect_commit="cfec845", expect_evidence=["adversarial", "concurrency"]),
    (dict(question="why is KUMIHO_MEMORY_ONTOLOGY default ON?",
          file="python/kumiho-memory/kumiho_memory/memory_manager.py"),
     expect_commit="e52e5df", expect_evidence=["paired", "+0.042"]),
    (dict(question="why is the additive partition unconditional?",
          file="python/kumiho-memory/kumiho_memory/context_compose.py"),
     expect_commit="10f113e", expect_evidence=["displacement"]),
]
# 판정은 사람이 아니라 스크립트 (B — verbatim evidence 덕에 기계화 가능):
#   top-3 안에 expect_commit prefix 유래 결정 존재 AND evidence 문자열 부분매치
```

3질의의 근거 문장이 대상 커밋 메시지(cfec845, e52e5df, 10f113e)에 실존함을 사전 확인했다 —
게이트는 추출·앵커링·검색의 종단 검증이지 요행이 아니다. (b)는 file 인자를 **포함**해서 판정 —
허브 파일(`memory_manager.py`) 위 CE 재랭크가 실제로 동작하는지가 §5.2의 검증 포인트다.

---

## 8. Risks + open questions

| # | 리스크 | 심각도 | 대응 |
|---|---|---|---|
| R1 | LLM title 불안정 → force 재캡처 시 중복 | 중 | 커밋 마커가 1차 방어(재추출 자체가 안 일어남); force는 deprecate-후-재작성(§4.6); title+date slug의 잔여 흔들림은 SUPERSEDES-Jaccard가 연결. 완전 dedup은 비목표 |
| R2 | 커밋 메시지 빈약 리포에서 rationale 품질 붕괴 | 중 | confidence=low 정직 마킹 + low+무증거 드랍; 코드 주석을 근거 소스로 완충(§4.3); Phase 2 PR/세션 채굴이 진짜 해법. "좋은 커밋 메시지 팀에게 가장 좋다"를 문서화 |
| R3 | 허브 파일에서 정답이 top-3 밖 | 중 | 사전식 융합 + anchor 티어 내 CE 재랭크(§5.2)가 구조적 해법; 도그푸드 (b)가 이 케이스를 직접 판정 |
| R4 | squash가 author-date를 바꾸는 엣지 케이스 (`--reset-author` 등) | 저 | slug 불일치 → 신규 노드 + SUPERSEDES-Jaccard 연결로 우아 강등. 생존표(§3.5)의 "재채굴 수렴"은 보편 보증이 아니라 기본 git 동작 기준임을 문서화 |
| R5 | 별도 project로 대화↔코드 브리징이 한 hop 멀어짐 | 저 | crowding 재발 방지가 우선. cross-project edge는 서버가 kref로 이미 지원 |
| R6 | 라인 범위의 빠른 부패 | 저 | boost-only 설계로 완화. blame-follow는 명시적 non-goal |
| R7 | `code_commit` 노드 수 증가 | 저 | 별도 project + 미미한 embedding_text. 필요 시 서버 FULLTEXT_EXCLUDED_KINDS에 `code_commit`만 추가 제안 (decision·evidence는 절대 제외 금지) |
| R8 | CE가 code-domain 텍스트에 약함 | 저 | 구조 prior가 사전식 우선이라 CE 실패가 anchor 매치를 못 뒤집음 |

**Open questions** (구현 전/중 확정):
1. `-code` 접미 project의 자동 생성 권한 — MCP 테넌트에서 create_project 허용 여부를 CE에서 확인
   (도그푸드 스크립트 첫 단계에 포함).
2. repo 식별자 정규화 — origin 없는 로컬 repo의 dir명 충돌. Phase 1은 `config.repo` 명시로 회피
   가능, 기본 휴리스틱만 제공.
3. 프롬프트 회귀 — 골든 커밋 셋(이 리포 10커밋) + 기대 결정 스냅샷 테스트를 둘지. Phase 1에서는
   도그푸드 3문항이 사실상 그 역할.
4. `kumiho_code_why`를 코딩 에이전트 시스템프롬프트/스킬에 어떻게 노출할지 (자동 트리거 vs 명시
   호출) — 도그푸드 결과 보고 결정.
5. Phase 2: 대화-`decision` ↔ `code_decision` 브리지 (`DERIVED_FROM` cross-project), REFINES 엣지
   승격, rename blame-follow — 전부 명시적 비목표 재확인.

---

## 9. 구현 순서 (P1 core → P2 capture → P3 dogfood)

코딩 에이전트가 그대로 따라갈 수 있는 파일 단위 순서. 각 단계는 독립 커밋 가능하고,
검증 기준을 명시한다.

### P1 — core (스키마·앵커·쿼리의 결정적 부분; LLM 무관)

1. **`kumiho_memory/code_decisions.py`** (신규)
   - `CodeMemoryConfig` dataclass (§6) + env 읽기 (`KUMIHO_MEMORY_DECISIONS*`, 레거시 `KUMIHO_MEMORY_CODE*` 폴백)
   - `normalize_path()` — 단일 소스 계약 (§3.2)
   - slug 함수들: `anchor_slug(repo, path)`, `decision_slug(title, author_date)`(충돌 가드 포함),
     `commit_slug`, `evidence_slug`
   - `_write_revision(item, metadata, embedding_text)` — `kumiho.get_client().create_revision`
     경유 (§0-5)
   - get-or-create 헬퍼 (per-slug lock, ALREADY_EXISTS 흡수 — entity_promotion 패턴 지역 구현)
   - kinds/spaces/edge-type 상수 + `schema_version`
   - → 검증: `test_code_query.py`의 normalize_path/slug 파트 + `test_code_capture.py`의
     slug 충돌 가드 파트 통과
2. **`kumiho_memory/code_query.py`** (신규, 캡처보다 먼저 — 쿼리가 fake 데이터로 테스트 가능)
   - anchor 역조회 (§3.3), line boost, commit 타이브레이크
   - semantic leg (direct `kumiho.search`), evidence-bridge leg
   - `_ce_scores()` (recall_rerank 프리미티브 재사용, §5.2) + 사전식 융합 + superseded 강등
   - evidence chain 전개 (fan-out 가드) + `compose_why_context()`
   - `why()` 공개 API
   - → 검증: `test_code_query.py` 전체 (fake project/SDK로)
3. **`kumiho_memory/mcp_tools.py`** (수정) — `kumiho_code_why` 등록 (게이트 내, anyOf 스키마)
   **`kumiho_memory/memory_manager.py`** (수정) — 게이트 안 lazy 위임 메서드
   - → 검증: MCP 게이트 on/off 테스트 + `test_code_isolation.py` (byte-identical 호출 그래프)
   - → **이 시점에 기존 ~435 테스트 전체 통과 확인** (venv pytest — 5개 pre-existing env drift
     제외 기준선 준수)

### P2 — capture (git 채굴 + LLM)

4. **`kumiho_memory/code_capture.py`** (신규)
   - `_git` 헬퍼: log 열거([1]), `--stat`/patch 수집, 결정적 diff 요약 패킷([3], §4.3)
   - 프리필터([2], §4.2)
   - LLM structuring([4], §4.4 — 배치 6, json_mode 스키마, summarizer 어댑터 재사용)
   - 검증([5]): changed-file 앵커 검증 + stat 폴백, confidence 컷
   - write([6]): evidence → anchors → decision(embedding_text 주입) → edges(get_edges 사전 확인)
   - supersede([7], §2.1 3-신호) → marker([8] 마지막)
   - `--force` deprecate 흐름, 증분 모드, `IngestStats` 반환
   - `ingest_repo()` 공개 API
   - → 검증: `test_code_capture.py` 전체 (합성 git repo + 스텁 LLM; 크래시-주입 재시도 테스트 포함)
5. **`kumiho_memory/__main__.py`** (수정) — `code-ingest` 서브커맨드;
   **`kumiho_memory/mcp_tools.py`** (수정) — `kumiho_code_ingest` 등록
   - → 검증: CLI 인자 파싱 유닛 + MCP 스키마 validate

### P3 — dogfood + 마감

6. **`python/kumiho-memory/scripts/dogfood_code_memory.py`** (신규, §7.2)
   - 1커밋 드라이런 → 30커밋 인제스트 → 3질의 기계 판정 → project 삭제 정리
   - → 검증: 라이브 CE(127.0.0.1:9190)에서 3/3 통과. 실패 시 어느 단계(추출/앵커/검색/융합)인지
     stats·match 필드로 분리 진단
7. 마감: `__version__`/CHANGELOG 정리 없이(릴리즈 사이클 별도), adversarial review 요청 → PR.
   PR 본문에 §3.5 생존표와 도그푸드 3/3 로그 첨부.
