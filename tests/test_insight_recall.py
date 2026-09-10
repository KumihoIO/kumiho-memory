import asyncio
import json
from types import SimpleNamespace

import pytest

from kumiho_memory.experience import normalize_experience
from kumiho_memory.insight_patterns import prepare_pattern_request, validate_pattern_candidate
from kumiho_memory.insight_recall import recall_learned_sources


def experience(name='first', **changes):
    record = normalize_experience(dict(experience_id='event-' + name, title='Pilot ' + name,
        situation='Small team', goal='Release', decision='Try pilot', rationale='Observe cost',
        expected_outcome='Know cost', origin='user', decision_state='accepted'))
    record['recorded_at'] = '2026-09-09T00:00:00+00:00'
    return {'kref': f'kref://p/experiences/{name}.experience?r=1',
            'metadata': {'experience_record': json.dumps(record)}, 'tags': [], **changes}


def pattern(rows):
    req = prepare_pattern_request(rows, space_paths=['/p'])
    cand = validate_pattern_candidate(req, {'kind': 'conditional_lesson', 'title': 'Pilot first',
        'hypothesis': 'Pilots may reveal operating cost', 'applicability_conditions': ['Uncertain cost'],
        'counterexamples': [], 'source_krefs': req['source_krefs']})
    return {'kref': 'kref://p/patterns/lesson.pattern_candidate?r=1',
            'metadata': {'pattern_candidate': json.dumps(cand)}, 'tags': []}


def setup(monkeypatch, rows, discovery=None):
    import kumiho
    indexed = {row['kref']: row for row in rows}
    calls = []
    reads = []
    def get_revision(ref):
        reads.append(ref)
        row = indexed[ref]
        return SimpleNamespace(kref=SimpleNamespace(uri=ref), metadata=row['metadata'],
                               tags=row.get('tags', []), deprecated=row.get('deprecated', False))
    def get_item(ref):
        row = next(row for key, row in indexed.items() if key.split('?')[0] == ref)
        return SimpleNamespace(kref=SimpleNamespace(uri=ref), metadata=row.get('item_markers', {}), deprecated=False)
    monkeypatch.setattr(kumiho, 'get_revision', get_revision)
    monkeypatch.setattr(kumiho, 'get_item', get_item)
    def retrieve(*, project, query, limit, space_paths, memory_item_kind, include_revision_metadata):
        calls.append({'project': project, 'query': query, 'limit': limit,
                      'space_paths': space_paths, 'kind': memory_item_kind})
        refs = discovery[memory_item_kind] if discovery is not None else [ref for ref in indexed if ref.split('?')[0].endswith('.' + memory_item_kind)]
        return {'revision_krefs': refs[:limit]}
    return SimpleNamespace(project='p', memory_retrieve=retrieve), calls, reads


def test_two_explicit_kinds_and_canonical_context(monkeypatch):
    first = experience()
    proposal = pattern([first])
    manager, calls, reads = setup(monkeypatch, [first, proposal])
    result = asyncio.run(recall_learned_sources(manager, 'Should we run a pilot?'))
    assert result['status'] == 'ready'
    assert [c['kind'] for c in calls] == ['experience', 'pattern_candidate']
    assert all(c['space_paths'] == ['/p'] for c in calls)
    assert len(result['results']) == 2
    assert result['results'][0]['canonical_record']['rationale'] == 'Observe cost'
    assert result['results'][1]['decision_state'] == 'proposal'
    assert result['results'][1]['source_health']['status'] == 'reviewable'
    assert all(r['evidence_level'] == 'unverified' for r in result['results'])


def test_scope_and_kind_filter_never_fall_back(monkeypatch):
    row = experience()
    outside = experience('other')
    outside['kref'] = 'kref://p/elsewhere/other.experience?r=1'
    manager, calls, reads = setup(monkeypatch, [row, outside], {
        'experience': [outside['kref'], row['kref'].replace('.experience', '.conversation')],
        'pattern_candidate': [row['kref']],
    })
    result = asyncio.run(recall_learned_sources(manager, 'pilot', space_paths=['/p/experiences']))
    assert result['results'] == [] and reads == []
    assert len(calls) == 2


def test_caps_three_selected_records_and_six_health_refs(monkeypatch):
    rows = [experience(str(i)) for i in range(12)]
    proposal = pattern(rows)
    manager, calls, reads = setup(monkeypatch, rows + [proposal])
    result = asyncio.run(recall_learned_sources(manager, 'pilot'))
    assert result['source_reads'] <= 3
    assert result['health_source_reads'] == 6
    assert len(reads) <= 9
    assert len(result['results']) <= 3
    recalled_pattern = next(r for r in result['results'] if r['type'] == 'pattern_candidate')
    assert recalled_pattern['source_health']['status'] == 'unknown'
    assert len(recalled_pattern['source_health']['missing_source_krefs']) == 6
    assert json.loads(recalled_pattern['summary'])['source_health'] == 'unknown'


def test_deprecated_source_marks_pattern_stale(monkeypatch):
    first = experience()
    proposal = pattern([first])
    first['deprecated'] = True
    manager, calls, reads = setup(monkeypatch, [first, proposal])
    result = asyncio.run(recall_learned_sources(manager, 'pilot'))
    assert next(r for r in result['results'] if r['type'] == 'pattern_candidate')['source_health']['status'] == 'stale'


def test_unknown_retrieval_signature_does_not_broaden(monkeypatch):
    manager, calls, reads = setup(monkeypatch, [experience()])
    manager.memory_retrieve = lambda project, query, limit: {'revision_krefs': []}
    result = asyncio.run(recall_learned_sources(manager, 'pilot'))
    assert result['status'] == 'partial'
    assert result['retrieval_calls'] == 2
    assert not result['results'] and not reads


@pytest.mark.parametrize('kwargs', [{'limit': 4}, {'space_paths': []}, {'space_paths': ['/other/private']}, {'space_paths': ['/p/sk-proj-' + 'A' * 24]}])
def test_invalid_scope_or_limits_do_not_retrieve(monkeypatch, kwargs):
    manager, calls, reads = setup(monkeypatch, [experience()])
    assert asyncio.run(recall_learned_sources(manager, 'pilot', **kwargs))['status'] == 'unavailable'
    assert not calls and not reads


def test_actual_installed_sdk_retrieve_dispatch(monkeypatch):
    import kumiho
    import kumiho.mcp_server as sdk
    first = experience()
    proposal = pattern([first])
    manager, _, reads = setup(monkeypatch, [first, proposal])
    indexed = {'experience': first, 'pattern_candidate': proposal}
    kinds = []
    monkeypatch.setattr(sdk, '_ensure_configured', lambda: None)
    monkeypatch.setattr(kumiho, 'get_project', lambda name: SimpleNamespace(name=name))
    def search(query, *, context, kind, include_revision_metadata):
        kinds.append(kind)
        row = indexed[kind]
        revision = SimpleNamespace(kref=SimpleNamespace(uri=row['kref']), metadata=row['metadata'])
        item = SimpleNamespace(kref=SimpleNamespace(uri=row['kref'].split('?')[0]),
            get_revision_by_tag=lambda tag: revision,
            space=SimpleNamespace(path='/p'))
        return [SimpleNamespace(item=item, score=1.0)]
    monkeypatch.setattr(kumiho, 'search', search)
    monkeypatch.setattr(kumiho, 'item_search', lambda **kwargs: pytest.fail('Unexpected broad fallback'))
    manager.memory_retrieve = sdk.tool_memory_retrieve
    result = asyncio.run(recall_learned_sources(manager, 'pilot'))
    assert result['status'] == 'ready'
    assert kinds == ['experience', 'pattern_candidate']
    assert len(result['results']) == 2


def test_canonical_event_identity_survives_learned_synthesis(monkeypatch):
    """Two reports of one event must remain recognizable to host reasoning."""
    first = experience()
    manager, _, _ = setup(monkeypatch, [first])
    result = asyncio.run(recall_learned_sources(manager, 'pilot'))
    packet = result['results'][0]
    canonical = json.loads(first['metadata']['experience_record'])
    assert packet['canonical_record']['experience_id'] == canonical['experience_id']
    assert packet['canonical_record']['record_id'] == canonical['record_id']
    from kumiho_memory.insight_synthesis import prepare_insight_request
    request = prepare_insight_request('pilot', result['results'])
    summary = json.loads(request['sources'][0]['summary'])
    assert summary['record']['experience_id'] == canonical['experience_id']


def test_item_level_stale_warning_survives_synthesis_whitelist(monkeypatch):
    first = experience(item_markers={'grounding_stale': 'true'})
    proposal = pattern([first])
    manager, _, _ = setup(monkeypatch, [first, proposal])
    result = asyncio.run(recall_learned_sources(manager, 'pilot'))
    from kumiho_memory.insight_synthesis import prepare_insight_request
    request = prepare_insight_request('pilot', result['results'])
    recalled = next(row for row in request['sources'] if row['type'] == 'pattern_candidate')
    assert not recalled.get('grounding_stale')
    summary = json.loads(recalled['summary'])
    assert summary['source_health'] == 'stale'
    assert summary['source_health_details']['stale_sources'] == [
        {'kref': first['kref'], 'reason': 'grounding_stale', 'marker_scope': 'item'}]
    # The warning is on the pattern's source item, not the pattern revision.
    own = next(row for row in request['sources'] if row['type'] == 'experience')
    assert own['item_markers']['grounding_stale'] is True
    assert not own.get('grounding_stale')


def test_missing_item_state_cannot_become_reviewable_in_synthesis(monkeypatch):
    first = experience()
    proposal = pattern([first])
    manager, _, _ = setup(monkeypatch, [first, proposal], discovery={
        'experience': [], 'pattern_candidate': [proposal['kref']],
    })
    import kumiho
    get_item = kumiho.get_item
    def missing(ref):
        if ref == first['kref'].split('?')[0]:
            raise ValueError('source item denied')
        return get_item(ref)
    monkeypatch.setattr(kumiho, 'get_item', missing)
    result = asyncio.run(recall_learned_sources(manager, 'pilot'))
    assert result['results'][0]['source_health']['status'] == 'unknown'
    from kumiho_memory.insight_synthesis import prepare_insight_request
    request = prepare_insight_request('pilot', result['results'])
    assert json.loads(request['sources'][0]['summary'])['source_health'] == 'unknown'


def test_secret_query_rejected_before_discovery(monkeypatch):
    manager, calls, reads = setup(monkeypatch, [experience()])
    result = asyncio.run(recall_learned_sources(manager, 'sk-proj-' + 'A' * 24))
    assert result['status'] == 'unavailable'
    assert calls == [] and reads == []


def test_source_counts_and_budget_account_for_oversized_backend_return(monkeypatch):
    from kumiho_memory.insight_recall import MAX_RESULT_CHARS
    rows = [experience(str(i)) for i in range(12)]
    manager, _, reads = setup(monkeypatch, rows)
    # The retrieval callable may ignore its requested limit; the adapter must
    # still cap refs before touching SDK revisions/items.
    manager.memory_retrieve = lambda **kwargs: {'revision_krefs': [r['kref'] for r in rows] * 1000}
    result = asyncio.run(recall_learned_sources(manager, 'pilot'))
    assert result['source_reads'] <= 3
    assert len(reads) <= 3
    assert len(json.dumps(result, ensure_ascii=False)) <= MAX_RESULT_CHARS


def test_positive_threshold_filters_aligned_sdk_scores_before_reads(monkeypatch):
    low, high = experience('low'), experience('high')
    manager, _, reads = setup(monkeypatch, [low, high])
    manager.memory_retrieve = lambda **kwargs: {'revision_krefs': [low['kref'], high['kref']], 'scores': [0.1, 0.9]}
    result = asyncio.run(recall_learned_sources(manager, 'pilot', min_score=0.5))
    assert [r['kref'] for r in result['results']] == [high['kref']]
    assert result['results'][0]['score'] == 0.9
    assert low['kref'] not in reads


@pytest.mark.parametrize('scores', [None, ['0.9'], [float('nan')], [True], []])
def test_positive_threshold_fails_closed_without_valid_scores(monkeypatch, scores):
    first = experience()
    manager, _, reads = setup(monkeypatch, [first])
    manager.memory_retrieve = lambda **kwargs: {'revision_krefs': [first['kref']], 'scores': scores}
    result = asyncio.run(recall_learned_sources(manager, 'pilot', min_score=0.5))
    assert not result['results'] and reads == []
    assert result['status'] == 'partial'


def test_zero_threshold_preserves_unscored_discovery(monkeypatch):
    manager, _, reads = setup(monkeypatch, [experience()])
    result = asyncio.run(recall_learned_sources(manager, 'pilot', min_score=0))
    assert result['status'] == 'ready'
    assert len(result['results']) == 1
    assert 'score' not in result['results'][0]


@pytest.mark.parametrize('types', [[], ['decision'], ['fact']])
def test_disjoint_memory_types_do_not_widen_discovery(monkeypatch, types):
    manager, calls, reads = setup(monkeypatch, [experience()])
    result = asyncio.run(recall_learned_sources(manager, 'pilot', memory_types=types))
    assert result['results'] == []
    assert calls == [] and reads == []


def test_outcome_type_filter_is_applied_to_canonical_record(monkeypatch):
    from kumiho_memory.experience import _normalize_outcome
    first = experience()
    observation = {**_normalize_outcome(first['kref'], {
        'observed_outcome': 'Support cost exceeded budget', 'observed_at': '2026-09-09T00:00:00Z',
        'outcome_status': 'failure'}), 'recorded_at': '2026-09-10T00:00:00Z'}
    outcome = {'kref': 'kref://p/experiences/result.experience?r=1',
               'metadata': {'experience_record': json.dumps(observation)}, 'tags': []}
    manager, calls, _ = setup(monkeypatch, [first, outcome])
    result = asyncio.run(recall_learned_sources(manager, 'pilot', memory_types=['outcome']))
    assert [c['kind'] for c in calls] == ['experience']
    assert len(result['results']) == 1
    assert result['results'][0]['type'] == 'outcome'
    assert result['results'][0]['canonical_record']['experience_kref'] == first['kref']


def test_pattern_only_filter_keeps_pattern_identity(monkeypatch):
    first = experience()
    saved = pattern([first])
    manager, calls, _ = setup(monkeypatch, [first, saved])
    result = asyncio.run(recall_learned_sources(manager, 'pilot', memory_types=['pattern_candidate']))
    assert [c['kind'] for c in calls] == ['pattern_candidate']
    assert len(result['results']) == 1
    assert result['results'][0]['type'] == 'pattern_candidate'
    assert result['results'][0]['canonical_record']['candidate_id']
