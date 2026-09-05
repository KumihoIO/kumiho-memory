# Reliability checks

## Non-live regression gate

Every pull request now runs the full non-live suite on Python 3.10, 3.11 and
3.12 in addition to building and importing the wheel. Failures fail CI; JUnit
reports are retained for seven days. Run locally in a separate virtualenv:

```sh
python -m pip install -e '.[dev,all]' pytest-timeout
python -m pytest -m 'not live' --strict-markers --timeout=90
```

Unit tests of session resolution explicitly disable the default SDK retriever.
Passing `None` to the manager constructor means **auto-load**, not disable.
The CLI preference test restores its environment even when the variables did
not exist before the test. Legacy HTTP compatibility tests supply their own
module double rather than depending on transitive provider dependencies.

## Live Cloud contracts

Use the **Cloud contract tests** workflow with the existing repository or
organization-shared `KUMIHO_AUTH_TOKEN` Actions secret. Missing credentials fail
the run; no replacement secret name or token value belongs in source/chat.
After the workflow is merged it supports manual dispatch. Before merge, pushing
an explicitly opted-in same-repository `test/cloud-*` branch runs it. Fork PRs
and `pull_request_target` never receive these credentials.

The tests run separately from the unit suite, which substitutes SDK modules.
They use official Cloud discovery, a temporary authentication/discovery cache,
request-scoped credentials, no LLM provider, and a fresh `memory-ci-<uuid>`
project. They verify:

- Cloud working-memory writes, session isolation, readback and explicit clearing.
- Graph memory storage and manager recall, including space-boundary filtering.
- Revision-pinned SUPERSEDES edges, target demotion, dependent staleness and
  idempotent replay, all checked with fresh server reads.
- Ownership-checked project cleanup in a finalizer. Cleanup archives the exact
  synthetic project with `force=False`; it never force-deletes real data.

For a deliberate local live run, supply `KUMIHO_AUTH_TOKEN` securely in the
process environment, set `KUMIHO_RUN_CLOUD_TESTS=1`, and run:

```sh
python -m pytest integration/test_cloud_memory.py -v --timeout=180 --tb=short
```

Do not enable traceback locals or upload authentication caches. Interrupted
runner/process termination can prevent finalizers: inspect any leftover
`memory-ci-*` project and its `memory_ci_owner` marker before cleaning up.
These tests are a contract check, not a load test or a claim of retrieval quality.

## Belief replacement protocol

Facts, ontology decisions, code-captured decisions and maintenance dedup now
share `supersession.supersede_revision`: confirm/create the edge, demote the
**same target revision**, then run the bounded grounding-staleness ripple.
An edge failure cannot demote the target. Replays repair a prior metadata or
ripple interruption even when the edge already exists. Code capture does not
stamp a complete commit marker after a reported edge/status failure.

CONTRADICTS does not demote a belief. Profile-history revision links remain
separate. Candidate selection is intentionally unchanged: lexical overlap is
still a heuristic, not semantic proof of a contradiction. Agent-declared
replacement targets remain preferable. The grounding ripple retains its
best-effort fanout cap of 20; this is not a multi-write transaction or an
unbounded repair of old graphs.
