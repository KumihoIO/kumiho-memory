# Decision Memory Phase 2 — SESSION MINING 최종 설계 (판정 합성본)

> kumiho-SDKs issue #43 + plugins#10. 독립 설계 A(enrichment-first)와 B(capture-first)를
> 어드버서리얼 판정한 뒤 합성한 구현 기준 문서. **골격은 A의 enrichment 승리-조건 역산**
> (도그푸드 게이트가 기계 판정문이 되도록), **캡처 파이프라인·입력 폴백·쿼리 라우팅 수정은
> B**를 채택했다. 판정 과정에서 실코드 재검증으로 발견된 결함의 해결은 본문에
> **[판정 반영]**으로 표기한다.
>
> Phase-1 규율 전부 승계: `KUMIHO_MEMORY_CODE` 옵트인, 신규 의존성 0, 서버 변경 0,
> sha-free identity, additive-only, 정직한 provenance, 비용 바운드, 모듈 경계
> (`code_session.py` 신규 1파일, identity 규칙은 전부 `code_decisions`에서 import).

---

## 0. 설계를 지배하는 검증 사실 (판정 시 전부 실코드 재확인)

1. **Redis 버퍼는 consolidation이 지운다** — `consolidate_session()`은 store 후
   `redis_buffer.clear_session()` 호출 (`memory_manager.py:1046`). 단, 그 **직전에 전체
   대화 전문을 마크다운 artifact로 영속화**한다 (`_build_conversation_markdown`
   `memory_manager.py:1238-1278`: `### {Role}` 헤더 + `<sub>{timestamp}</sub>`의 결정적
   포맷). → 입력 표면은 "Redis 생존 시 raw, 소멸 시 artifact 역파싱"으로 이원화 (§2.1).
2. **consolidation은 revision kref를 손에 쥔다** — `store_result["revision_kref"]`
   (`memory_manager.py:1015`). 게이트 안 후처리 체인의 선례:
   `if stored_kref and self.ontology_enabled: decompose_and_link(...)`
   (`memory_manager.py:1025-1033`) — 같은 자리·같은 모양으로 세션 마이닝을 옵트인 체인.
3. **`kumiho.search`에 metadata 필터 없음** — session_id로 consolidated revision을 역검색
   불가. 브리지 kref는 호출자 전달 또는 체인 in-band만 지원 경로.
4. **evidence-bridge leg가 세션 evidence를 공짜로 잡는다** —
   `code_query._sync_evidence_bridge_leg`(237-283)는 evidence space hit의 INCOMING
   `MOTIVATED_BY`를 타고 결정으로 승격. 세션 evidence를 `MOTIVATED_BY`로 달면 쿼리
   랭킹/융합 무변경으로 why()에 노출.
5. **[판정 반영] chain 전개의 유령-커밋 버그** — `_sync_expand_chain`(`code_query.py:327-334`)은
   OUTGOING `DERIVED_FROM` 대상을 **무조건 커밋으로 렌더**한다 (`m.get("hash","")`).
   결정→`code_session` 마커의 DERIVED_FROM 엣지를 달면 `chain["commits"]`에
   `{"sha":"", "subject":"", "date":""}` 유령 항목이 생기고 fetch 예산에서 실커밋을
   밀어낼 수 있다 (B가 발견, 코드로 확정). A의 "쿼리 변경 0줄" 주장은 이 지점에서
   거짓 — **DERIVED_FROM 분기에 라우팅 수정이 필수다** (§6).
6. **enrichment의 원자성 교훈** — supersede 패스는 구 결정을 새 revision이 아니라 같은
   revision 위 `set_attribute`로 강등 (`code_capture.py:634-646` 주석: 엣지는
   revision-scoped, 새 revision은 identity를 쪼갠다). → enrichment는 기존 decision
   revision에 **엣지/노드 추가만**, revision 스택·metadata 재작성 절대 금지.
7. **멱등 마커 패턴** — `code_commit` 마커를 마지막에 쓰고 `_marker_complete`
   (`code_capture.py:547-577`)가 `decisions_count` 대비 INCOMING `DERIVED_FROM` 수를
   검증해 마커-쓰기/엣지-쓰기 사이 크래시 윈도를 재시도로 전환. 세션 마커는 직계 재사용.
8. **[판정 반영] true-force는 이미 출하됐다** — 두 초안 모두 "client `deprecate_item`은
   Phase2-1 미래 과제"로 미뤘으나, Phase2-1은 **완료**됐고
   `_force_deprecate_commit_decisions`(`code_capture.py:503-544`) + `item.set_deprecated`가
   현행 코드에 있다. 세션 force는 이 패턴을 지금 미러링한다 (§5.4).
9. **[판정 반영] `reject_credentials`는 예외를 던진다** (`privacy.py:67`,
   `CredentialDetectedError` raise). 세션 원문은 env 덤프·예시 키가 흔하므로 패킷/세션
   전체에 걸면 **그 세션은 영구 마이닝 불능**이 된다 (재시도해도 동일 실패). 두 초안
   모두 이를 간과 — 자격증명 검사는 **원자 단위** 적용 + 해당 원자만 드랍으로 해결 (§5.1 [5]).
10. **PII 선례** — 대화 도메인에서 LLM(summarizer)은 raw messages를 보고, cloud 쓰기
    직전에만 `anonymize_summary`를 통과한다 (`memory_manager.py:809,932`). 단 세션
    마이닝은 verbatim 검증이 파이프라인의 축이므로, **redact를 패킷 단계(LLM 이전)로
    올려 "LLM이 보는 텍스트 = 검증 기준 텍스트 = 저장 텍스트"를 단일화**한다 (A안 채택;
    B안의 쓰기-직전 redact는 검증한 문자열과 저장 문자열이 달라지는 불일치 — **[판정 반영]**).
11. **대화 recall은 브리지에 무영향** — `graph_augmentation.py:54-58`의 기본 edge_types는
    `DERIVED_FROM, DEPENDS_ON, REFERENCED, CONTAINS, CREATED_FROM, SUPERSEDES, SUPPORTS`.
    신규 `DISCUSSED_IN`은 셋 밖 → 대화 revision에 INCOMING으로 달려도 recall 경로 불변
    (판정 시 재검증).
12. **evidence identity = verbatim statement slug** (`code_decisions.py:266`) — 커밋
    메시지 문장이 세션에서 그대로 인용되면 slug 수렴 = dedup 1차 방어가 공짜.
13. **Korean tokenizer 실측 (PR#20)** — 한/영 혼합 임베딩은 식별자를 파편화. verbatim
    보존(원문)과 임베딩 품질(영역)을 분리하는 `statement_en` 장치 필요 (§4.2).
14. **메시지 셰이프** — `redis_memory.get_messages(project, session_id, limit)` →
    `{"messages": [{role, content, timestamp, metadata}], "message_count": N, ...}`;
    ts는 `timestamp` 또는 `metadata.timestamp` (artifact 빌더가 이 순서로 읽음).
    consolidation은 `limit=1000`으로 읽는다 — 마이닝도 동일 상한 (초과분 head 소실은
    consolidation과 같은 수용).
15. **LLM 배선** — `manager.summarizer.adapter` + `light_model`, `json_mode` strict
    (`memory_manager.py:1986-1989`의 `code_ingest` 위임과 동일).

---

## 1. 골격: enrichment 승리 조건에서의 역산

승리 조건 (도그푸드 §7.2의 기계 판정문):

> cfec845 커밋-마이닝 결정에 대해, 커밋 메시지에 **없는** "asyncio.to_thread rejected —
> default executor is shared, 32-thread oversubscription"이 세션 마이닝 후
> `why("why not asyncio.to_thread for the rerank offload?", file=recall_rerank.py)` 결과의
> evidence로 `source_ref=session:*` + `evidence_kind=rejected_alternative`와 함께 반환된다.
> 동시에 커밋 유래 evidence·결정 metadata는 1바이트도 변하지 않는다.

| 승리 조건의 절 | 강제되는 설계 |
|---|---|
| "커밋-마이닝 결정에 대해" | 상관이 결정적 발견 + 신호 합류로만 enrich — 오합병 없음 (§3) |
| "커밋 메시지에 없는" | evidence dedup 3층이 신규 정보만 통과 (§4.2) |
| "why()의 evidence로 반환" | `MOTIVATED_BY` + evidence-bridge leg 재사용 (§6) |
| "session provenance로" | `source_ref=session:<id>#m<n>` + `DERIVED_FROM`→세션 마커 (§4.3) |
| "1바이트도 안 변한다" | enrichment = 추가만; 기존 revision/metadata 무접촉 (§3.4) |

Standalone 캡처(미션 2)는 상관 실패 후보의 경로, 브리지(미션 3)는 kref가 손에 있을 때의
부가 엣지다. 단, standalone의 품질은 공짜가 아니다 — **캡처 파이프라인(salience/packet/
verbatim 검증)은 B안을 채택**한다: 세션은 커밋과 달리 "전부 통과"가 불가능하고, 합의
성립("제안+수락")이 여러 메시지에 걸치는 대화 문법을 가지므로, 이진 컷이 아니라 랭킹+예산
충전이 맞다 (§2.3).

---

## 2. 입력 표면 + 트리거/API + 예산

### 2.1 입력: raw 우선, artifact 폴백 — 요약은 입력이 아니다

| 소스 | 언제 | 비고 |
|---|---|---|
| **raw Redis messages** (`get_messages(..., limit=1000)`) | 세션 생존 중 / 체인 내부(메시지 in-band) | 1차 소스. verbatim 원칙은 원문에서만 성립 |
| **explicit `messages` 인자** | plugins#10 훅이 transcript를 직접 투입 | 같은 파이프라인 |
| **consolidated artifact 역파싱** (B안) | Redis가 비워진/만료된 과거 세션 | `### {Role}` + `<sub>ts</sub>` 결정적 포맷 역파싱. role·ts·순서 복원, metadata 소실 수용. **골든 라운드트립 유닛으로 빌더-파서를 한 계약으로 고정** (포맷 드리프트 방어) |
| `knowledge.decisions` 요약 | **입력 아님** | 압축+redacted라 verbatim 소실 — Phase 2의 존재 이유가 이 갭 |

**[판정 반영]** A안은 artifact 파서를 open question으로 미뤘으나, Redis TTL 만료 +
consolidation 소거를 합치면 **폴백 없는 수동 툴은 "지금 이 순간의 세션"에만 동작**한다 —
"지난주 그 세션을 마이닝해줘"가 Phase 2의 실사용 시나리오이므로 B의 폴백을 v1에 포함한다.
파서는 `code_session.py` 내 ~40줄 순수 함수 + 골든 테스트로 경계를 갚는다.

### 2.2 트리거/API — 3형

**(a) manager 메서드** (Phase-1 `code_ingest`와 같은 lazy/게이트 위임 셰이프):

```python
async def code_mine_session(
    self,
    session_id: str = "",
    *,
    messages: Optional[List[Dict[str, Any]]] = None,   # 훅/체인 직접 투입
    conversation_kref: str = "",   # 브리지 대상 (없으면 브리지 생략)
    repo_path: str = ".",
    ingest_first: bool = True,     # enrichment 신뢰성의 전제 (§3.1)
    force: bool = False,
) -> Dict[str, Any]:               # SessionMineStats.as_dict()
    # 게이트 밖: {"errors": ["code memory is disabled (set KUMIHO_MEMORY_CODE=1)"]}
    # 게이트 안: (1) ingest_first → incremental ingest_repo(repo_path) 선행
    #                (마커-스킵이라 재실행 LLM 비용 0),
    #            (2) messages 미제공 → Redis 로드, 비어있으면 artifact 폴백 시도,
    #            (3) code_session.mine_session(...) 위임.
```

**(b) MCP tool** — `mcp_tools._CODE_MEMORY_TOOLS`에 append (기존 게이트 블록
`mcp_tools.py:1439-1447` 내 등록, 게이트 OFF 시 미등록):

```json
{ "name": "kumiho_code_mine_session",
  "description": "Mine the current agent session into the code-decision graph: enrich commit-mined decisions with conversation-only alternatives/measurements, capture decisions that never reached a commit, and bridge decisions to the consolidated conversation.",
  "inputSchema": { "type": "object", "properties": {
      "session_id":        {"type": "string"},
      "conversation_kref": {"type": "string", "description": "consolidated revision kref for the bridge edge; omit to skip bridging"},
      "repo_path":         {"type": "string", "default": "."},
      "ingest_first":      {"type": "boolean", "default": true},
      "force":             {"type": "boolean", "default": false}},
    "required": ["session_id"] } }
```

**(c) consolidation 체인 (이중 옵트인, 기본 OFF)** — `KUMIHO_MEMORY_CODE=1` **그리고**
`KUMIHO_MEMORY_CODE_AUTOMINE=1`일 때만, `consolidate_session()`의 `decompose_and_link`
블록 직후 · `clear_session()` **직전** (~8줄, lazy import):

```python
if stored_kref and _code_automine_enabled():
    try:
        await self.code_mine_session(
            session_id, messages=messages,            # 이미 메모리에 있음 — Redis 재조회 0
            conversation_kref=stored_kref, ingest_first=True,
        )
    except Exception as exc:                          # 마이닝 실패가 consolidation을 못 죽인다
        logger.warning("code session mining failed (non-fatal): %s", exc)
```

- 위치가 본질: Redis 소거 전 + `stored_kref` in-band (사실 3의 역검색 불가 제약 우회).
  체인 모드가 브리지(미션 3)의 주 경로다.
- 두 env 모두 unset이면 env 체크 1회 외 실행 없음 — 대화 경로 diff는 `if` 1개
  (`decompose_and_link` 체인과 동일 선례·동일 안전 등급, `test_code_isolation.py`
  byte-identical 검증을 consolidation 경로로 확장해 증명).

### 2.3 예산: salience 스코어링 (B) + 패킷 규율

세션은 커밋보다 크고(수백 turn, tool 덤프) 노이즈 비율이 높다. **결정적 salience 랭킹 →
예산 충전 → 청크** 3단 (전 단계 LLM 비용 0):

**[i] salience 스코어 (결정적, false-negative 비대칭)**

```
score(msg) = Σ
  +3  결정/합의 어휘: decided|let's go with|we'll use|going with|opt(ed)? for|
      instead of|rather than|reject(ed)?|revert|기각|채택|결정|선택|가기로|롤백|대신
  +3  대안 구문: "why not X"|considered ... but|"X 대신"|"X 말고"
  +2  측정 패턴: \d+(\.\d+)?\s*(%|x|ms|s\b|tokens?|MB|점) | measured|benchmark|F1|recall|p9\d|실측
  +2  리뷰 어휘: critical|confirmed|defect|blocker|리뷰|지적
  +2  커밋 sha: \b[0-9a-f]{7,40}\b
  +1  파일 경로 regex(확장자 필수), +1 백틱 식별자 (식별자 가산 상한 +3)
  +1  직전 메시지가 고신호(≥4)인 짧은 user 메시지의 assent:
      yes|do it|go ahead|approve|좋아|그래|승인|ㄱㄱ   ← 합의 성립의 증인
  -2  잡음: 스택트레이스(비알파벳 라인 비율>0.8), base64/hex 덩어리,
      500자 초과 순수 코드 페이스트
```

- `score >= session_salience_min(2)` 메시지 + **±1 이웃**(제안과 수락은 다른 메시지에
  산다) 시간순 채택. **프레임 보존**: 첫 2 + 끝 3 메시지 무조건 포함.
- 생략 구간은 `[... N messages elided (low signal)]` 마커 — LLM이 윈도잉을 인지.
- 예산 초과 시 저점수부터 탈락 (이웃 자격만인 메시지 먼저).
- **role별 절삭**: `tool` role/초장문은 head 500 + `[...]` + tail 300
  (`session_per_message_chars=800`) — 결정 문장은 서두/말미에 몰린다 (A+B 병합).

**[ii] 청크** — 시간순 연속 분할, 메시지 중간 비분할, 각 메시지에 `[m41 user
2026-07-10T14:22]` 번호 프레임 부착 (source_ref와 decided_at의 원천). 청크 헤더에
`session {id}, chunk k/N, messages m..n of M`. 청크 간 상태 공유 없음 — **각 청크 독립
마이닝 + slug 수렴 dedup** (크로스-청크 상태 기계는 안 만든다; 중복은 identity 레이어가
흡수).

**[iii] 상한** — `session_chunk_chars=18000`(~4.5k tokens), `session_max_chunks=6` =
세션당 LLM 콜 하드캡 6, 동시 2 (`asyncio.Semaphore`). 초과 시 salience 밀도(청크 내 평균
score) 상위 유지.

**비용 모델**: 전형 200-메시지 세션 = salience 후 40-60 메시지 → 2-3청크 → **$0.005-0.02**
(light_model). 최악 6청크 ~$0.05 — `code_ingest` 30-커밋 예산과 같은 자릿수.
`ingest_first`의 incremental ingest는 마커-스킵으로 LLM 0콜 (RPC 마커 체크 ≤ max_commits회).

---

## 3. 상관 알고리즘: 절대 오합병하지 않는 enrich-or-create

### 3.1 전제: commit-first 순서

enrichment는 기존 `code_decision`이 있어야 성립한다. `ingest_first=True` 기본값이
"대화하고 → 커밋하고 → 세션 마이닝" 워크플로에서 enrichment를 기본 동작으로 만든다
(방금 커밋된 것만 신규 채굴, 나머지 마커-스킵).

### 3.2 후보 발견 — 결정적, 검색 무의존

**벡터/렉시컬 검색은 후보 발견에 쓰지 않는다** — 확률 신호로 병합 대상을 고르는 것
자체가 오합병 벡터다 (supersede 패스와 같은 원칙). LLM도 상관에 무개입 (층위 분리).

- **T1 (sha 경로)**: candidate의 `mentioned_commits`(LLM 추출 + hex-regex 선검증)를
  `git rev-parse --verify --quiet <tok>^{commit}`으로 실재 확인 → full sha →
  `commit_slug(repo, sha)` `get_item` → 마커 revision의 INCOMING `DERIVED_FROM` 소스
  = 그 커밋에서 채굴된 결정들.
- **T2 (anchor 경로)**: candidate의 검증된 파일(§5.1 [4], `git ls-files` 대조) →
  `anchor_slug` → INCOMING `IMPLEMENTED_IN` 소스 결정들 (`_supersede_pass`와 동일 기하).

### 3.3 판정 — 신호 합류(conjunction), 임계는 split 편향

**wrong-merge는 복구 곤란, wrong-split은 나중에 꿰맬 수 있다** (Phase-1 slug 충돌 가드
원칙). 어휘 일치 단독으로는 어떤 경로에서도 병합하지 않는다:

```
lex = jaccard(_tokens(FULL prose), _tokens(FULL prose))
      # [도그푸드 캘리브레이션, 2026-07-11] FULL prose = title+decision+rationale+
      # why_question 양쪽 모두. 라이브 동일-결정 쌍이 title+decision만으로 0.14,
      # 전체 산문으로 0.26 실측 — 세션과 커밋은 같은 결정을 다르게 제목 짓지만
      # rationale 어휘(진짜 why)는 공유한다. (relations._jaccard/_tokens 재사용)

ENRICH(target) iff:
  [S1: sha 경로]    target ∈ T1  AND  lex ≥ 0.20         # sha가 사실을 증언 — lex는 sanity floor
                    # (도그푸드 실측 보정: 정직한 동일-결정 쌍 ~0.26, 오인용-sha
                    #  음성 ~0.1 — 0.20이 여유를 갖고 가른다. 초안의 0.25는 미측정)
  [S2: anchor 경로] target ∈ T2  AND  lex ≥ 0.35
                    AND (symbol_overlap ≥ 1  OR  lex ≥ 0.50)
                    AND |parse_decided_at(target) − session_ts| ≤ 14일
둘 다 실패 → STANDALONE (§3.5)
```

- **[판정 반영]** S2는 A의 symbol 합류(`rerank_async` 같은 식별자는 고정밀 토큰)와 B의
  **14일 창**(세션은 커밋 직전/직후에 산다 — cfec845 시나리오는 같은 날; A의 30일은
  "같은 파일의 다른 시대 결정" 오합병 표면만 넓힌다)을 병합했다. 임계는 미측정 캘리브레이션
  이므로 전부 config 노출 — 도그푸드+초기 실사용이 캘리브레이션 게이트.
- **복수 target 통과 시 lex 최고 1개만** (동률 → decided_at 최신). enrichment fan-out은
  오합병의 완곡한 형태 — 금지.
- **감사 흔적**: enrichment 엣지 metadata에 `{session_id, correlation: "sha"|"anchored",
  overlap: "0.42"}` — 오합병 발생 시 세션 단위 진단·`kumiho_delete_edge`로 수동 복구 가능.

### 3.4 ENRICH 불변식 — 추가만, 재작성 없음

1. 기존 decision revision에 **새 revision을 만들지 않는다** (사실 6).
2. 기존 decision/커밋 evidence의 **metadata를 1키도 갱신하지 않는다** (`set_attribute`
   조차 금지 — enrichment는 상태 전이가 아니라 정보 추가).
3. 추가되는 것 전부 `_create_edge_once` 존재-확인 경유:
   (i) 신규 `code_evidence` 노드(세션 유래) + decision→evidence `MOTIVATED_BY`,
   (ii) 신규 evidence→세션 마커 `DERIVED_FROM`,
   (iii) enriched decision→세션 마커 `DERIVED_FROM` (감사 허브 — force/audit이 세션이
   건드린 결정을 열거하는 유일 경로),
   (iv) decision→대화 revision `DISCUSSED_IN` (kref 있을 때).
4. 최악의 오합병 피해 = 남의 결정에 잘못된 엣지 몇 개 — metadata 오염·identity 훼손
   없음, 엣지 삭제로 복구 가능. (그래도 §3.3 임계로 발생 자체를 억제.)

### 3.5 STANDALONE — 커밋에 닿지 않은 결정 (미션 2)

- `get_or_create_decision_item(title, decided_at, decision_text)` — 같은 slug 규칙·같은
  충돌 가드. `decided_at` = `settled_by_message`의 ts (없으면 세션 마지막 ts; Redis가
  UTC ISO 기록) — 재마이닝 수렴 (sha-free/non-rotting).
- metadata: Phase-1 스키마 + `origin: "session"`, `session_id`,
  `source_ref: "session:<id>#m<n>"`, `status_hint: "committed|uncommitted|unknown"`;
  `commit_hash: ""`.
- **anchor 3단 사다리** (B):
  1. ls-files 검증 통과 경로 → `code_anchor` get-or-create + `IMPLEMENTED_IN`
     **role=`mentioned`** (기존 `primary|touched` enum의 additive 확장; 쿼리는 role로
     필터하지 않으므로 anchor leg가 무변경으로 서피스 — "이 파일에 아직 커밋 안 된 결정이
     있다"는 정확히 에이전트가 알아야 할 정보다). edge metadata
     `{commit_hash: "", session_id}`.
  2. 경로 전멸 → anchor-less semantic-only (anchor 허브 오염 0).
  3. 커밋 실체화 시 → 아래 스티치.
- embedding_text (세션판 `_compose_embedding_text` — **B의 rejected-alternatives 조성 채택**:
  실사용 질의의 상당수가 기각된 쪽 이름으로 온다 — doc2query 역방향):
  ```
  {why_question} {decision}. Rationale: {rationale}.
  Rejected alternatives: {opt1} ({reason1}); {opt2} ({reason2}).
  Anchored: {basenames} ({symbols}). Session: "{첫 user turn 요지 ≤80자}".
  ```
- provenance: decision→세션 마커 `DERIVED_FROM` (커밋 결정의 →`code_commit`과 평행).
- **세션-먼저 → 커밋-나중의 스티치 (커밋 경로 0줄)**: (i) 같은 제목·같은 날이면 slug
  수렴 + 충돌 가드 Jaccard가 "같은 결정" 판정 → 커밋 마이닝이 그 노드에
  IMPLEMENTED_IN/DERIVED_FROM을 보태는 자연-enrichment. (ii) 제목이 달라도 mentioned-role
  엣지가 INCOMING `IMPLEMENTED_IN`이므로 기존 anchor-scoped SUPERSEDES 패스가 세션 결정을
  후보로 자연 발견 — blind 0.5 통과 시 커밋 결정이 세션 결정을 supersede ("구현이 논의를
  대체" — 의미 정직, superseded_by 체인이 세션 alternatives를 계속 노출). 역-꿰매기 코드는
  비목표 (§8).

---

## 4. 스키마 추가분 — 전부 `kumiho.code_memory.v1`의 하위호환 확장

버전 문자열 유지: 신규 kind/키/엣지는 모르는 reader에게 무해 (kind는 별도 space, 키는
무시, 엣지는 기본 walk 셋 밖 — 사실 11).

### 4.1 `code_session` (신규 kind) — 멱등 장부 + provenance 허브

- space `sessions` 신설. `slug = slugify(f"{repo}-session-{session_id}",
  hash_on_truncate=True)` — **[판정 반영]** repo를 slug에 넣어 A의 R7(다른 repo에서
  session_id 재사용 시 마커 오스킵)을 접미사 우회 없이 구조적으로 해소 (B 채택).
- metadata (전부 str): `{session_id, repo, mined_at, message_count, source
  ("redis"|"artifact"|"explicit"), decisions_created, decisions_enriched, evidence_added,
  conversation_kref, capture_version, schema_version}`.
- embedding_text: 첫 user turn 요지 한 줄 (`code_commit`의 subject-only와 같은 최소주의;
  sessions space는 어떤 why() leg도 검색하지 않으므로 벡터 오염 없음).
- **처리 완료 후 마지막에 생성** (§5.2). 완결성 검증 `_session_marker_complete`:
  INCOMING `DERIVED_FROM` 수 ≥ `decisions_created + decisions_enriched + evidence_added`
  (`_marker_complete` 직계 — 마커-먼저-크래시 창을 재시도로 변환).

### 4.2 alternatives 표현 + evidence dedup 3층

**표현**: 신규 노드/엣지 타입 없음. `EVIDENCE_KINDS += ("rejected_alternative",)` +
`code_evidence` metadata 옵션 키:

| key | 내용 |
|---|---|
| `statement` | (기존) **세션 원문 verbatim 인용** — "we considered asyncio.to_thread and rejected it because the default executor is shared" |
| `evidence_kind` | (기존) `rejected_alternative` 포함 확장 |
| `source_ref` | (기존) 신규 관례 `session:<session_id>#m<idx>` |
| `alternative` | (신규 옵션) 기각된 대안의 짧은 이름 — `asyncio.to_thread` |
| `statement_en` | (신규 옵션) 비영어 verbatim의 영역. **임베딩 전용** (embedding_text = `statement_en or statement`) — PR#20 파편화 회피와 verbatim 정직성을 동시 충족 |

- **[판정 반영]** B의 정규형("Rejected: {option} — {reason}")은 기각한다 — 정규형은
  verbatim이 아니어서 트랜스크립트-대조 기계 판정의 기준을 스스로 깬다. statement는
  항상 원문 인용(한국어면 한국어), 대안 이름은 `alternative` 키로. LLM 스키마는 B처럼
  `alternatives`를 별도 배열로 분리해 추출을 강제하되(§5.3), 각 항목에 verbatim `quote`
  필드를 요구하고 그것이 statement가 된다.
- `MOTIVATED_BY`로 연결 — 기각은 의미상 결정의 근거다(그것을 기각했기 **때문에** 이
  결정). evidence 노드 자체의 embedding이 기각 문장이므로 "why not X?" 질의를
  evidence-bridge leg가 잡는다.

**dedup 3층** (커밋 evidence를 밀어내지도, 스팸하지도 않는다):

1. **slug 수렴 (공짜)**: 같은 문장이 커밋/세션 양쪽에 있으면 같은 노드로 수렴. **기존
   노드면 metadata 무접촉** (source_ref가 `commit:*`인 채 유지 — 최초 소스의 정직성),
   빠진 `MOTIVATED_BY` 엣지만 존재-확인 후 추가. "세션에서도 언급됨"은 엣지 metadata의
   `session_id`가 증언.
2. **근사중복 컷**: 신규 생성 전, 대상 결정의 기존 `MOTIVATED_BY` evidence statement들과
   token Jaccard ≥ `evidence_dup_jaccard(0.8)`이면 skip. 미만이면 생성 — enrichment의
   존재 이유가 "커밋이 잃은 것"이므로 컷은 보수적으로.
3. **상한**: 결정당 세션 추가 evidence ≤ `session_max_evidence_per_decision(4)` —
   기존 캡(6)과 합쳐 chain fan-out 가드(MAX_EDGES_PER_DECISION=32) 안.

### 4.3 source_ref / origin 관례 (honest provenance)

| 아티팩트 | 커밋 유래 | 세션 유래 |
|---|---|---|
| evidence `source_ref` | `commit:<sha12>` | `session:<id>#m<idx>` |
| decision provenance | `DERIVED_FROM` → `code_commit` | `DERIVED_FROM` → `code_session` |
| decision metadata | (origin 키 없음 = commit) | `origin: "session"` |
| enrichment 엣지 metadata | — | `{session_id, correlation, overlap}` |

reader 규칙: `origin` 부재 = `"commit"`. 어떤 소스 표기도 사후 재작성 없음.

### 4.4 `DISCUSSED_IN` (신규 edge) — 대화↔코드 브리지 (미션 3)

```
code_decision(rev) --DISCUSSED_IN--> conversation revision (cross-project, kref)
metadata: {session_id}
```

- **방향이 code→conversation인 이유**: (i) 쓰기가 code 도메인에서만 일어난다 — 대화
  project 노드/metadata 무접촉 (게이트 OFF 대화-경로-불변 제약의 그래프판), (ii) why()
  chain 전개는 OUTGOING walk 기존 구조, (iii) 대화 recall 기본 edge_types 밖 (사실 11).
- 대상 revision은 `kumiho.get_revision(kref)`로 획득 — cross-project는 서버가 kref로
  이미 지원, 신규 메커니즘 0. enrich·standalone 양쪽, kref 있을 때만, 존재-확인 후.
- kref는 그 시점의 consolidated revision을 영구히 가리킨다 — "그때 그 대화"가 브리지의
  의미이므로 최신-추적은 비목표.

### 4.5 CodeMemoryConfig 추가 필드 (전부 additive)

```python
sessions_space: str = "sessions"
session_salience_min: int = 2
session_per_message_chars: int = 800
session_chunk_chars: int = 18000
session_max_chunks: int = 6
session_max_decisions: int = 8                 # 세션당 LLM 출력 컷
session_max_evidence_per_decision: int = 4     # 결정당 세션-추가 evidence
session_max_alternatives_per_decision: int = 4
evidence_dup_jaccard: float = 0.8
evidence_containment: float = 0.6              # verbatim 완화 매치 (§5.1 [4])
correlate_jaccard_sha: float = 0.20      # 도그푸드 실측 보정 (§3.3)
correlate_jaccard_anchored: float = 0.35
correlate_jaccard_blind: float = 0.50          # anchored + symbol 0개일 때
correlate_window_days: int = 14
session_remine_message_delta: int = 10         # §5.2 [2]
```

env 추가: `KUMIHO_MEMORY_CODE_AUTOMINE` (기본 unset=off). `KUMIHO_MEMORY_CODE` 마스터
게이트는 그대로 — AUTOMINE은 하위 스위치.

---

## 5. 쓰기 파이프라인 + 멱등 + 크래시 안전

### 5.1 스테이지

```
[1] load        messages 확보 (인자 | Redis | artifact 역파싱) + m<idx>/ts 부여
[2] salience    결정적 스코어링 + 이웃 스티칭 + 프레임 보존 + 절삭/청크/상한 (§2.3)
[3] redact      pii_redactor.anonymize_summary를 **청크 패킷 텍스트에** 적용 —
                LLM이 보는 것 = verbatim 검증 기준 = 저장 텍스트 (단일 텍스트 스트림).
[4] structure   청크별 LLM 콜 (json_mode strict — §5.3), 동시 2
[5] validate    기계 검증 — 전부 결정적:
                (a) verbatim: evidence.text / alternative.quote가 redacted 패킷의
                    공백-정규화+casefold 부분문자열인가 → 실패 시 token containment
                    ≥ 0.6 완화 매치 → 그것도 실패면 그 원자만 드랍 (결정은 유지);
                    드랍 수는 stats.evidence_dropped_verbatim으로 노출 (프롬프트
                    회귀의 조기 신호)
                (b) sha: hex-regex 선검증 + `git rev-parse --verify` → 불통과 폐기
                (c) file: normalize_path 후 `git ls-files` 집합(세션당 1회 캐시) 대조
                    → 불통과 시 **유일-접미사 해석** (세그먼트 경계의 suffix가
                    추적 파일 중 정확히 1개와 일치하면 그 경로로 복원 —
                    라이브 실측: 모델이 full path를 kumiho_memory/recall_rerank.py로
                    축약) → 그래도 불통과면 폐기; 전멸 시 anchor-less 생존
                    [도그푸드 캘리브레이션, 2026-07-11]
                (d) confidence==low && evidence 0 && alternatives 0 → 후보 드랍
                (e) **[판정 반영] credentials — 원자 단위**: 각 statement/decision/
                    rationale/embedding_text에 reject_credentials를 개별 try로 적용,
                    CredentialDetectedError면 **그 원자(또는 그 결정)만 드랍** +
                    stats.credentials_dropped. 세션 전체에 걸면 credential 1개가
                    세션을 영구 마이닝-불능으로 만든다 (raise 함수임 — privacy.py:67)
[6] correlate   T1/T2 발견 + S1/S2 판정 → enrich | standalone (§3)
[7] write       (sync 워커 1개, run_bounded_in_thread + write_timeout — 커밋 쓰기와 동형)
                enrich: evidence get-or-create + MOTIVATED_BY + DERIVED_FROM(evidence→마커,
                        decision→마커) + DISCUSSED_IN          (§3.4 불변식)
                standalone: Phase-1 write 경로 재사용 (decision + anchors(role=mentioned)
                        + evidence + 엣지) + DERIVED_FROM(→마커) + DISCUSSED_IN
[8] marker      code_session 마커 = 마지막 (완결성 카운트 포함, §4.1)
```

### 5.2 멱등성 (Phase-1 3중 + 세션 레벨 2규칙)

1. **세션 레벨**: `_session_marker_complete` 통과 = 전체 skip (LLM 0콜).
2. **재마이닝 트리거 (B)**: 마커 존재 + 현재 가용 메시지 수 > 마커 `message_count` +
   `session_remine_message_delta(10)` → 전체 재마이닝 (증분 파싱 상태 기계 없음 — slug
   수렴+근사중복 컷+엣지 존재-확인이 재실행을 안전·단순하게 만든다). 메시지 소스가
   없으면(체인 후 Redis 소거 등) 이 검사는 자연히 skip.
3. **브리지-only 보정 패스 (B)**: 마커 완결 + 신규 `conversation_kref` 제공(마커의
   `conversation_kref`가 비었거나 다름) → **LLM 0콜**로 마커의 INCOMING `DERIVED_FROM`
   결정들에 `DISCUSSED_IN`만 보충 + 마커 metadata 갱신 (마커 자신의 metadata는 세션
   도메인 소유라 갱신 허용). "수동 마이닝 → 나중에 consolidation" 시퀀스의 지원 경로 —
   A의 open question #2를 흡수.
4. 노드 레벨 get-or-create + sha-free slug / 엣지 레벨 `_create_edge_once` — 전부 재사용.

### 5.3 LLM 스키마 (summarizer adapter + light_model, json_mode strict, `obj()` 빌더 재사용)

```json
{ "decisions": [ {
    "title": "...", "decision": "...", "rationale": "...", "why_question": "...",
    "symbols": ["asyncio.to_thread", "_RERANK_EXECUTOR"],
    "files": ["python/kumiho-memory/kumiho_memory/recall_rerank.py"],
    "mentioned_commits": ["cfec845"],
    "alternatives": [ {
        "option": "asyncio.to_thread",
        "verdict": "rejected|deferred",
        "quote": "세션 원문 verbatim — 기각 사유가 담긴 문장",
        "quote_en": "영역 — quote가 비영어일 때만" } ],
    "evidence": [ {
        "kind": "measurement|review_finding|incident|benchmark|constraint",
        "text": "세션 원문 verbatim 인용",
        "text_en": "영역 — 비영어일 때만",
        "message_index": 41 } ],
    "settled_by_message": 42,
    "status_hint": "committed|uncommitted|unknown",
    "confidence": "high|medium|low"
} ] }
```

`alternatives[].quote`가 `rejected_alternative` evidence의 statement가 되고 `option`이
`alternative` 키가 된다 — **추출은 분리 강제(B), 저장은 verbatim 원칙(A)**.

프롬프트 핵심 (커밋판 4-분류 공유 + 세션 문법 재정의):

> 세션에서 결정이란 **합의가 성립한 선택**이다: (a) 제안 + 명시적 수락, (b) 이유가
> 발화된 대안 기각, (c) 측정 결과가 선택을 강제한 순간, (d) 리뷰어 지적 수용 + 수정
> 방향 확정. 결정이 아닌 것: 결론 없이 흐른 탐색적 가설, 코드가 이미 말하는 것의 재서술,
> TODO 희망사항, 질문 자체. **0개도 정답이다.** `alternatives`는 세션의 고유 화물 —
> 고려되고 명시적으로 기각/보류된 옵션과 그 사유 문장을 verbatim `quote`로 담아라.
> `evidence.text`와 `quote`는 트랜스크립트의 verbatim 인용 — 검증기가 원문 대조로
> 탈락시킨다.

### 5.4 force = deprecate-then-rewrite — **[판정 반영] 지금 구현한다**

두 초안 모두 "true-force는 Phase2-1 이후"로 미뤘으나 Phase2-1은 완료됐고
`_force_deprecate_commit_decisions` 패턴이 현행이다. 세션판
`_force_deprecate_session_decisions`:

- 세션 마커의 INCOMING `DERIVED_FROM` 소스를 열거하고, **metadata에 `origin=="session"`
  AND `session_id`가 이 세션인 decision 노드만** `set_attribute("status","deprecated")` +
  `item.set_deprecated(True)`.
- **가드가 본질**: 마커의 INCOMING에는 이 세션이 **enrich한 커밋-유래 결정**도 매달려
  있다 (§3.4 (iii)) — origin 가드 없이 커밋 패턴을 그대로 복사하면 force가 남의 커밋
  결정을 강등한다. evidence 원자도 (커밋 패턴과 동일하게) 공유 자산이라 deprecate 대상이
  아니다.
- 이전 enrichment 엣지는 남는다 — 재마이닝이 같은 verbatim이면 slug/존재-확인으로 수렴,
  달라졌으면 근사중복 컷이 스팸을 막는다. 잔여 stale 엣지는 감사 metadata(session_id)로
  수동 `kumiho_delete_edge` 가능 — 수용 한계로 문서화.
- force 재캡처가 deprecated item에 수렴하면 `set_deprecated(False)` 복원 + 새 revision
  (커밋 force와 동일, `code_capture.py:729-743`).

### 5.5 크래시 안전

| 크래시 지점 | 결과 |
|---|---|
| validate 이전 | 아무것도 안 씀 — 재실행 무비용 재시도 |
| write 일부 후 중단 | 마커 없음 → 전체 재시도; get-or-create/존재-확인 수렴 |
| 마커 revision 후 · DERIVED_FROM 전 | 완결성 카운트 검증이 미완 판정 → 재시도 |
| Redis 만료/체인 실패로 원문 소실 | artifact 폴백 시도; 그것도 없으면 stats.errors에 명시하고 종료 (조용히 삼키지 않음 — 포그라운드 원칙) |

### 5.6 모듈 배치 (flat 유지, 신규 1파일)

```
kumiho_memory/
  code_session.py       # 신규 — 본 설계 전부: load/salience/redact/structure/validate/
                        #   correlate/write/marker + artifact 파서 + SessionMineStats
                        #   identity·쓰기 프리미티브는 전부 code_decisions에서 import
                        #   (_run_git 류는 code_capture에서 import — code_query가
                        #    derive_repo_id를 가져가는 기존 선례와 동일)
  code_decisions.py     # +KIND_SESSION, +EDGE_DISCUSSED_IN, +EVIDENCE_KINDS 1항,
                        #   +session_slug(), +config 필드 (§4.5) — 전부 additive
  code_query.py         # +DERIVED_FROM 라우팅 수정 + DISCUSSED_IN 전개 + 렌더 (§6)
  code_capture.py       # 무변경 (0줄)
  mcp_tools.py          # +kumiho_code_mine_session (기존 게이트 블록 내)
  memory_manager.py     # +code_mine_session 위임 + AUTOMINE 체인 블록 (§2.2)
```

신규 의존성 0, 서버 변경 0.

---

## 6. 쿼리 통합 — 거의 그대로, 단 라우팅 수정 1건은 필수

- **[판정 반영] DERIVED_FROM 라우팅 수정 (유령-커밋 차단)** — `_sync_expand_chain`의
  `EDGE_DERIVED_FROM and src == me` 분기에서 대상 revision metadata에 `session_id`가
  있으면 `chain["sessions"].append({"session_id": ..., "mined_at": ...})`, 없으면 기존
  `chain["commits"]` 경로. 이 라우팅이 없으면 세션 provenance가
  `commits: [{"sha": ""}]` 유령으로 새고(코드 확정, 사실 5) `compose_why_context`의
  `d["commits"][0]["sha"]` 헤더/fetch 예산을 오염시킨다. 세션 마커 metadata에는
  `session_id`가 항상 있고 커밋 마커에는 없다 — 결정적 판별자.
- **alternatives/세션 evidence: 변경 0줄.** `MOTIVATED_BY` 연결이므로 chain 전개가
  statement/kind/source_ref를 이미 반환 — `rejected_alternative`는 kind 문자열이 그대로
  렌더된다: `- (rejected_alternative) "…default executor is shared…" [session:abc#m41]`.
  "why not X?" 질의는 evidence-bridge leg가 잡는다.
- **standalone 세션 결정: 변경 0줄 + 패스스루 1줄.** 같은 kind·space라 semantic leg
  그대로, mentioned-role 엣지라 anchor leg 그대로. `_answer_from`에
  `origin`/`status_hint` 패스스루 1줄 — 에이전트가 "아직 커밋에 안 닿은 결정"임을 안다.
- **브리지: additive 2곳.** `_sync_expand_chain`에 `elif etype == EDGE_DISCUSSED_IN and
  src == me: chain["conversation"] = {"kref": dst, "session_id": ...}` 1분기 +
  `_answer_from`/`compose_why_context`에 1줄 렌더. `_prio`의 SUPERSEDES-first 정렬
  무접촉 (superseded_by 보증이 계속 예산 1순위).
- **융합/랭킹 무변경**: 사전식 키에 origin 축을 넣지 않는다 — "세션-유래가 커밋-유래보다
  약하다"는 prior는 미검증; 도그푸드가 반례를 보이면 그때 `active` 뒤 티어로 논의 (비목표).
- fan-out: 세션 evidence 상한 4 + MAX_EDGES_PER_DECISION=32로 chain 예산 유지.

---

## 7. 테스트 플랜 + 라이브 도그푸드

### 7.1 Unit (신규 ~30개; LLM·서버·Redis·git 무접속 — 전부 스텁/합성. 기존 스위트 무접촉 통과가 게이트)

`test_code_session.py`:
- **salience/packet**: 렉시콘 축별 점화, assent-이웃 스티칭, 프레임(첫2·끝3) 보존, 생략
  마커, tool 덤프 절삭(head500+tail300), 청크 메시지-비분할, max_chunks 캡, `[m41 ...]`
  번호 프레임, KO+EN 혼합, 스택트레이스 감점, 결정성(같은 입력 = 같은 패킷)
- **artifact 파서**: `_build_conversation_markdown` 실출력과의 골든 라운드트립 (role·ts·순서)
- **redact-then-verify 일관성**: PII 포함 메시지 → 인용 검증이 redacted 기준으로 성립
- **validate**: 비-verbatim 드랍(공백/casefold 정규화 통과 · 완전 창작 탈락 · containment
  0.6 경계), 가짜 sha 폐기, ls-files 밖 파일 폐기 → anchor-less 생존, low+무증거+무대안
  드랍, **credential 원자 드랍이 세션을 죽이지 않음 [판정 반영]**
- **correlate (핵심 매트릭스)**: S1 sha+lex 0.3 → enrich / sha+lex 0.1 → standalone;
  S2 anchor+0.4+symbol1 → enrich / anchor 단독 → standalone / **lex 0.6 단독(anchor
  없음) → standalone** (합류 없이 절대 병합 없음) / 15일 밖 → standalone;
  복수 target → 최고 1개만
- **additive 불변식 (헌법 테스트)**: enrich 후 대상 decision revision 수 불변 + metadata
  byte-identical + 기존 커밋 evidence source_ref 불변; 신규는 엣지·evidence만
- **dedup**: 동일 문장 → slug 수렴 + 기존 노드 metadata 무접촉 + 엣지만 추가; Jaccard
  0.85 → skip; 0.5 → 생성; 결정당 상한 4
- **standalone**: origin=session, decided_at=settled ts, slug 재마이닝 수렴, role=mentioned
  앵커, DERIVED_FROM→마커, embedding_text에 "Rejected alternatives" 포함
- **브리지**: kref 존재 시 DISCUSSED_IN 1회(재실행 무중복), 없으면 생략, 브리지-only
  보정 패스(LLM 0콜)
- **멱등/크래시**: 마커 완결 재실행 = LLM 0콜; message_count 델타 → 재마이닝; 마커-전
  크래시 주입 → 전체 재시도; 마커-후-엣지-전 → 완결성 검증이 재시도 전환;
  **force: origin=session 결정만 deprecate되고 enrich된 커밋-유래 결정은 무접촉
  [판정 반영]**
- **게이트/격리**: `KUMIHO_MEMORY_CODE` off → manager 즉시 반환/MCP 미등록; `AUTOMINE`
  off → consolidate_session 호출 그래프 byte-identical (test_code_isolation 확장)

`test_code_query.py` 확장 (+4): **DERIVED_FROM 세션 라우팅(유령-sha 차단) [판정 반영]**,
DISCUSSED_IN 전개, conversation/origin 렌더, DISCUSSED_IN이 superseded_by 예산 불침범.

### 7.2 Live dogfood — ENRICHMENT을 특정해서 증명 (127.0.0.1:9190 CE, 수동, CI 제외)

`scripts/dogfood_session_memory.py` — Paid-run preflight 원칙: 1청크 드라이런으로
발화·JSON 준수 확인 후 발사; 종료 시 `delete_project` 정리 (SmokeTest 관례).
도그푸드 문자열은 PII/credential-free로 설계 (리댁션과 기계 판정의 충돌 방지).

```python
os.environ["KUMIHO_MEMORY_CODE"] = "1"
# [0] 전용 project + Phase-1 ingest: HEAD~30..HEAD → cfec845 결정 실존 확인
# [1] ENRICHMENT: 합성 세션 A (redis add_message ×8) — 실제 cfec845 논의 재구성:
#     user:      "the CE rerank is blocking the event loop under the locomo harness"
#     assistant: "two options: asyncio.to_thread, or a dedicated executor"
#     user:      "we considered asyncio.to_thread and rejected it because the
#                 default executor is shared — a 32-thread pool oversubscribes
#                 the cross-encoder"                      ← 커밋 메시지에 없는 문장
#     assistant: "agreed — dedicated single-worker ThreadPoolExecutor keeps
#                 inference serialized. committing as cfec845"
#     (+ recall_rerank.py 경로 언급 turn)
#     before-스냅샷: cfec845 결정 revision 수 + metadata dict + 기존 evidence source_ref
# [2] stats = await manager.code_mine_session(session_A, conversation_kref=K)
#     assert stats["decisions_enriched"] >= 1 and stats["decisions_created"] == 0
#     (enrich 대신 create가 나오면 상관 실패 — 오합병·오분열 양방향 동시 판정)
# [3] after-스냅샷 대조: revision 수 불변, metadata byte-identical,
#     commit:* evidence 무변경                            ← additive 불변식 라이브 증명
# [4] r = await manager.code_why("why not asyncio.to_thread for the rerank offload?",
#                                file="python/kumiho-memory/kumiho_memory/recall_rerank.py")
#     [판정 반영] 판정은 top-1 고정이 아니라 top-3 스캔 — anchor leg가
#     recall_rerank.py의 결정을 전부 반환하므로 top-1은 CE 재랭크에 결합돼 flaky:
#     assert any(d에 cfec845 유래 AND evidence 중
#                kind=="rejected_alternative"
#                AND "default executor is shared" in statement   (verbatim 부분매치)
#                AND source_ref.startswith("session:")
#                for d in r["decisions"][:3])
#     AND 그 d["conversation"]["kref"] == K                      (브리지)
#     AND r["decisions"] 어디에도 sha=="" 유령 커밋 없음          (라우팅 수정 증명)
# [5] STANDALONE 대조군: 커밋 무관 결정("defer the bge-m3 embedding migration until
#     after the release cycle") 세션 B → created >= 1, origin=="session",
#     why("why was the bge-m3 migration deferred?") top-3 회수, match=="semantic"
# [6] 같은 세션 재마이닝 → llm_calls == 0, 노드/엣지 수 불변      (멱등 라이브 증명)
# [7] 세션 B consolidate → kref 확보 → 브리지-only 보정 패스 → DISCUSSED_IN resolve
# [8] delete_project 정리
```

성공 기준: [2][3][4][5][6][7] **6/6**. 실패 시 stats(enriched/created/dropped_*)와
match 필드로 스테이지(추출/검증/상관/쓰기/쿼리) 분리 진단.

---

## 8. 리스크 + 비목표

| # | 리스크 | 심각도 | 대응 |
|---|---|---|---|
| R1 | 잔여 오합병 (S2 임계를 우연히 통과하는 어휘+anchor+symbol 일치) | 중 | conjunction 임계 + 14일 창 + fan-out 금지로 억제; 피해가 additive-only(엣지)라 복구 가능; 엣지 metadata session_id로 세션 단위 감사·delete_edge. 임계는 미측정 캘리브레이션 — 도그푸드+초기 실사용이 게이트 |
| R2 | 세션 노이즈에서 가짜 결정 추출 (브레인스토밍 오인) | 중 | "합의 성립 = 결정" 프롬프트 + verbatim 검증 + low+무증거 드랍; standalone 오탐은 커밋 결정을 오염시키지 않음 (별도 노드, origin 정직 마킹) |
| R3 | verbatim 검증이 다중-메시지 조합 rationale에 과엄격 → evidence 기근 | 중 | 2단 완화(정규화 substring → containment 0.6) + evidence는 원자 단위 드랍(결정 생존) + 드랍률 stats — 도그푸드가 회귀 감지 |
| R4 | "말한 것 ≠ 한 것" — 세션 결정 미실행 | 중 | status_hint/origin 정직 마킹 + settled-only 프롬프트 + 커밋 실체화 시 supersede 스티치 (§3.5) |
| R5 | Redis 만료/체인 미가동으로 원문 소실 | 중 | AUTOMINE 체인이 구조 해법 + artifact 폴백이 백스톱 + 훅 explicit messages |
| R6 | artifact 포맷 드리프트가 폴백 파서를 조용히 죽임 | 저 | 골든 라운드트립 유닛이 빌더-파서를 한 계약으로 고정 |
| R7 | 세션 PII/credential 유출 | 중 | 패킷-단계 redact (LLM 이전) + 원자 단위 reject_credentials 드랍 (§5.1 [5]) |
| R8 | 비영어 세션의 검색 품질 | 저 | statement_en/quote_en 임베딩 분리; statement는 원문 유지 |
| R9 | evidence space 성장이 bridge leg 정밀도 희석 | 저 | 결정당 상한 4 + 근사중복 컷 + 세션당 결정 상한 8; leg는 scan_limit 컷 |
| R10 | force 재마이닝의 잔여 stale enrichment 엣지 | 저 | slug/존재-확인 수렴이 대부분 흡수; 잔여는 session_id 감사 → 수동 delete_edge (수용 한계로 문서화) |
| R11 | salience 렉시콘 언어 편향 (KO/EN 외) | 저 | false-negative 비대칭 원칙상 미스는 "덜 캡처"로 그침; 렉시콘은 상수, v2 재보정 |

**비목표 (전부 명시)**:
- 커밋 채굴 시점의 역방향 세션-꿰매기 (`code_capture.py` 0줄 원칙 — 스티치는 slug 수렴
  + 기존 SUPERSEDES 패스가 공짜로 제공, §3.5)
- PR/이슈/리뷰 코멘트 마이닝 (Phase 3 후보)
- conversation ontology의 대화-`decision` 노드와의 노드-노드 정합 — 브리지는
  consolidated revision 단위(kref)만
- 랭킹에 origin 축 추가 (미검증 prior)
- AUTOMINE 기본 ON — paired evidence 없이 기본값을 켜지 않는다 (Plus↔LoCoMo 트레이드오프
  교훈; LoCoMo 게이트 0 영향이 릴리즈 안전핀)
- 브리지의 최신-revision 추적 ("그때 그 대화"가 옳은 semantics)
- 멀티-세션 집계 결정, 신규 관계 타입 승격 (`REALIZED` 등)

**Open questions**:
1. plugins#10 훅의 transcript 포맷 → `messages` 셰이프 어댑터를 훅 쪽/SDK 쪽 어디에 둘지
   (Phase2-2 구현 시 훅 스펙 확정과 함께).
2. `alternative` 키의 decision-metadata 승격 집계(`alternatives_rejected` csv) — 렌더
   편의 vs metadata 비대. 도그푸드 결과 보고 결정.

---

## 9. 구현 순서 (파일 단위 — 코딩 에이전트용, 단계별 검증 기준)

1. **`code_decisions.py`** — `KIND_SESSION`, `EDGE_DISCUSSED_IN`,
   `EVIDENCE_KINDS += ("rejected_alternative",)`, `session_slug(repo, session_id)`,
   §4.5 config 필드. 전부 additive.
   → 검증: 기존 테스트 무접촉 통과 + session_slug/config 유닛.
2. **`code_session.py` 1부 (결정적 코어, LLM 무관)** — messages load(3소스) + artifact
   역파서 + salience 스코어러 + 절삭/청크 packet + redact + validate(verbatim/sha/
   ls-files/credential-원자) 순수 함수들.
   → 검증: salience/packet/파서 골든 라운드트립/validate 유닛 전체 (최다 유닛 구간).
3. **`code_session.py` 2부 (correlate + write + marker)** — T1/T2 발견, S1/S2 판정,
   enrich 쓰기(불변식), standalone 쓰기(Phase-1 경로 재사용), DISCUSSED_IN 브리지,
   `_session_marker_complete`, 재마이닝 델타, 브리지-only 패스,
   `_force_deprecate_session_decisions`(origin 가드), `SessionMineStats`,
   `mine_session()` 오케스트레이션.
   → 검증: correlate 매트릭스 + additive 헌법 테스트 + 멱등/크래시/force 유닛.
4. **`code_query.py`** — DERIVED_FROM 세션 라우팅 수정(**유령-sha 차단이 이 단계의
   핵심 diff**), DISCUSSED_IN 전개, `_answer_from` origin/conversation 패스스루,
   `compose_why_context` 1줄 렌더.
   → 검증: test_code_query 확장 4개 + 기존 쿼리 테스트 무수정 통과.
5. **`memory_manager.py` + `mcp_tools.py`** — `code_mine_session` 위임(게이트 셰이프는
   `code_ingest` 복제) + AUTOMINE 체인 블록(`decompose_and_link` 직후·`clear_session`
   직전) + MCP 툴 등록.
   → 검증: 게이트 on/off 유닛 + AUTOMINE off 시 consolidate 호출 그래프 byte-identical
   + **기존 전체 스위트** (venv pytest; 5개 pre-existing env-drift 실패는 기준선).
6. **`scripts/dogfood_session_memory.py`** — §7.2 시나리오.
   → 검증: 1청크 드라이런 선행 후 라이브 6/6.
7. **어드버서리얼 리뷰 → PR** — 본문에 correlate 매트릭스 표 + 도그푸드 [3]
   before/after 스냅샷 로그 + 유령-sha 라우팅 수정의 전/후 chain 덤프 첨부.
