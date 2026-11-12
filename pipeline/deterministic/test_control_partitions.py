import pytest

from pipeline.deterministic.control_partitions import (
    bind_joint_components,
    select_opened_parents,
)


def test_opened_selection_reconciles_exclusions_and_distinct_parents_in_pool_order():
    rows = [
        {'source_id': str(source), 'parent_tpf': str(parent)}
        for source, parent in [(1, 10), (2, 20), (3, 10), (4, 30), (5, 40), (6, 50)]
    ]
    result = select_opened_parents(rows, 2, {'2'}, {'30'})
    assert result['status'] == 'NOT_FROZEN'
    assert result['selection'] == [rows[0], rows[4]]
    assert [row['reason'] for row in result['excluded']] == [
        'DECLARED_SOURCE_EXCLUSION', 'SELECTED_PARENT_DUPLICATE',
        'DECLARED_PARENT_EXCLUSION', 'BEYOND_REQUESTED_COUNT',
    ]
    assert len(result['selection']) + len(result['excluded']) == len(rows)
    with pytest.raises(ValueError, match='distinct parents'):
        select_opened_parents(rows, 4, {'2'}, {'30'})
    with pytest.raises(ValueError, match='invalid'):
        select_opened_parents([rows[0], rows[0]], 1, set())


def joint_metadata():
    return {
        'schema': 1,
        'status': 'PORTABLE_METADATA_SCOPE_PREPARATION_ONLY',
        'joint_independence_complete': False,
        'trials': [
            {'trial_id': 'holdout:A', 'scope': 'HOLDOUT', 'full_witness_component_id': 'shared'},
            {'trial_id': 'core:B', 'scope': 'CORE_FINAL', 'full_witness_component_id': 'shared'},
            {'trial_id': 'opened:C', 'scope': 'OPENED', 'full_witness_component_id': 'separate'},
        ],
        'full_witness_components': [
            {'component_id': 'shared', 'trial_ids': ['holdout:A', 'core:B'],
             'identity_map_members': {'historical': ['reserved:D']}, 'historical_stage_exposure': 'UNRESOLVED'},
            {'component_id': 'separate', 'trial_ids': ['opened:C'], 'identity_map_members': {},
             'historical_stage_exposure': 'UNRESOLVED'},
        ],
        'holdout_overlap_witnesses': [{'holdout_trial_id': 'holdout:A', 'shared_identity_edges': ['parent:10']}],
        'qualified_external_connectivity': [{'reserved:D': 'external:E'}],
        'unresolved': {'null_partitions': 'NOT_PROVISIONED'},
        'boundary': 'Metadata only; original local inputs not reverified',
    }


def test_joint_selection_retains_sealed_and_historical_dependency_members():
    metadata = joint_metadata()
    result = bind_joint_components(metadata, ['holdout:A'])
    assert result['selected_trial_ids'] == ['holdout:A']
    assert result['full_witness_components'] == metadata['full_witness_components'][:1]
    assert result['cross_scope_components'] == {'shared': ['CORE_FINAL', 'HOLDOUT']}
    assert result['holdout_overlap_witnesses'] == metadata['holdout_overlap_witnesses']
    assert result['qualified_external_connectivity'] == metadata['qualified_external_connectivity']
    assert result['unresolved'] == metadata['unresolved']
    assert result['joint_independence_complete'] is False
    assert result['status'] == 'METADATA_COMPONENT_BINDINGS_ONLY'


@pytest.mark.parametrize('selected', [[], ['missing'], ['holdout:A', 'holdout:A']])
def test_joint_selection_rejects_missing_and_repeated_trials(selected):
    with pytest.raises(ValueError, match='missing or repeated'):
        bind_joint_components(joint_metadata(), selected)


@pytest.mark.parametrize('mutation', ['duplicate_trial', 'duplicate_component', 'missing_member', 'repeated_member', 'wrong_component'])
def test_joint_selection_rejects_inconsistent_full_dependency_graph(mutation):
    metadata = joint_metadata()
    if mutation == 'duplicate_trial':
        metadata['trials'].append(metadata['trials'][0])
    elif mutation == 'duplicate_component':
        metadata['full_witness_components'].append(metadata['full_witness_components'][0])
    elif mutation == 'missing_member':
        metadata['full_witness_components'][0]['trial_ids'].pop()
    elif mutation == 'repeated_member':
        metadata['full_witness_components'][0]['trial_ids'].append('holdout:A')
    else:
        metadata['trials'][0]['full_witness_component_id'] = 'separate'
    with pytest.raises(ValueError, match='component'):
        bind_joint_components(metadata, ['holdout:A'])


def test_joint_component_bindings_cannot_claim_independence():
    metadata = joint_metadata()
    metadata['joint_independence_complete'] = True
    with pytest.raises(ValueError, match='provisional'):
        bind_joint_components(metadata, ['holdout:A'])


@pytest.mark.parametrize('schema', [99, None, True, '1'])
def test_joint_component_binding_rejects_an_unrecognized_schema(schema):
    metadata = joint_metadata()
    metadata['schema'] = schema
    with pytest.raises(ValueError, match='provisional'):
        bind_joint_components(metadata, ['holdout:A'])


@pytest.mark.parametrize('field', ['trial_id', 'scope', 'full_witness_component_id'])
@pytest.mark.parametrize('value', ['', '  ', None, 99])
def test_joint_component_binding_rejects_blank_or_nonstring_trial_identities(field, value):
    metadata = joint_metadata()
    metadata['trials'][0][field] = value
    with pytest.raises(ValueError, match='identities'):
        bind_joint_components(metadata, ['holdout:A'])


@pytest.mark.parametrize('field', ['component_id', 'trial_ids', 'identity_map_members', 'historical_stage_exposure'])
def test_joint_component_binding_rejects_truncated_full_witnesses(field):
    metadata = joint_metadata()
    del metadata['full_witness_components'][0][field]
    with pytest.raises(ValueError, match='witnesses'):
        bind_joint_components(metadata, ['holdout:A'])


@pytest.mark.parametrize('field,value', [
    ('component_id', ''), ('component_id', None), ('historical_stage_exposure', ''),
    ('identity_map_members', []), ('trial_ids', ['']), ('trial_ids', 'holdout:A'),
])
def test_joint_component_binding_rejects_malformed_witness_values(field, value):
    metadata = joint_metadata()
    metadata['full_witness_components'][0][field] = value
    with pytest.raises(ValueError, match='witnesses'):
        bind_joint_components(metadata, ['holdout:A'])
