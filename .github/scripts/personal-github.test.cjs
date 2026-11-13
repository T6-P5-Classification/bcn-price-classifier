const { test } = require('node:test');
const assert = require('node:assert/strict');
const { issueNumbers, synchronize, startIssue } = require('./personal-github.cjs');
const config = require('../personal-github.json');

// Fake clients: these tests never use credentials or make network requests.
function fixture({ pr = {}, status = 'In progress', reviews = [], issue = {}, archived = false, missing = false } = {}) {
  const writes = [];
  const moves = [];
  const github = {
    rest: {
      pulls: {
        get: async () => ({ data: {
          number: 100, state: 'open', draft: false, requested_reviewers: [],
          body: 'Closes #101', base: { ref: config.integration_branch, repo: { full_name: config.repository } }, ...pr,
        } }),
        listReviews: 'reviews', list: 'pulls',
      },
      issues: {
        get: async () => ({ data: { node_id: 'FAKE_ISSUE', state: 'open', ...issue } }),
        update: async (args) => writes.push(args),
      },
    },
    paginate: async (method) => method === 'reviews' ? reviews : [],
  };
  const projectGithub = { graphql: async (query, args) => {
    if (query.includes('query LifecycleProject')) return { organization: { projectV2: {
      id: 'FAKE_PROJECT', field: { id: 'FAKE_STATUS', options: [
        { id: 'PROGRESS', name: 'In progress' }, { id: 'REVIEW', name: 'In Review' }, { id: 'DONE', name: 'Done' },
      ] },
    } } };
    if (query.includes('query LifecycleItems')) return { node: { projectItems: {
      nodes: missing ? [] : [{ id: 'FAKE_CARD', project: { id: 'FAKE_PROJECT' },
        isArchived: archived, fieldValueByName: { name: status } }],
      pageInfo: { hasNextPage: false },
    } } };
    if (query.includes('mutation LifecycleStatus')) { moves.push(args); return {}; }
    throw new Error('Unexpected fake API call');
  } };
  const [owner, repo] = config.repository.split('/');
  return { config, github, projectGithub, context: { repo: { owner, repo } },
    core: { info() {}, warning() {} }, prNumber: 100, writes, moves };
}

test('configuration points to this repository, dev and Project 3 with exact status names', () => {
  assert.equal(config.repository, 'T6-P5-Classification/bcn-price-classifier');
  assert.equal(config.integration_branch, 'dev');
  assert.deepEqual(config.project, {
    owner: 'T6-P5-Classification', owner_type: 'organization', number: 3,
    status_field: 'Status', review_status: 'In Review', done_status: 'Done', progress_status: 'In progress',
  });
});

test('only standalone local completion directives count, never fenced examples or comments', () => {
  assert.deepEqual(issueNumbers('Closes #101\nFixes #102\nCloses #101\nRefs #103\nCloses other/repo#104\n```\nCloses #105\n```\n<!-- Closes #106 -->'), [101, 102]);
});

test('ready PR moves the issue to In Review without closing it', async () => {
  const f = fixture(); await synchronize(f);
  assert.equal(f.moves[0].option, 'REVIEW');
  assert.equal(f.writes.length, 0);
});

test('merge into dev closes the issue and moves its card to Done', async () => {
  const f = fixture({ pr: { state: 'closed', merged: true } }); await synchronize(f);
  assert.equal(f.writes[0].issue_number, 101);
  assert.equal(f.writes[0].state_reason, 'completed');
  assert.equal(f.moves[0].option, 'DONE');
});

test('draft and unmerged closure return to In progress, without closing the issue', async () => {
  for (const pr of [{ draft: true }, { state: 'closed', merged: false }]) {
    const f = fixture({ pr, status: 'In Review' }); await synchronize(f);
    assert.equal(f.moves[0].option, 'PROGRESS');
    assert.equal(f.writes.length, 0);
  }
});

test('blocking review remains effective after a comment and clears on explicit re-request', async () => {
  const reviews = [
    { id: 1, state: 'CHANGES_REQUESTED', user: { login: 'reviewer' } },
    { id: 2, state: 'COMMENTED', user: { login: 'reviewer' } },
  ];
  const blocked = fixture({ status: 'In Review', reviews }); await synchronize(blocked);
  assert.equal(blocked.moves[0].option, 'PROGRESS');
  const requested = fixture({ reviews, pr: { requested_reviewers: [{ login: 'reviewer' }] } });
  await synchronize(requested);
  assert.equal(requested.moves[0].option, 'REVIEW');
});

test('closed, archived and Done issues/cards never regress', async () => {
  for (const state of [{ issue: { state: 'closed' } }, { archived: true }, { status: 'Done' }]) {
    const f = fixture(state); await synchronize(f);
    assert.equal(f.moves.length + f.writes.length, 0);
  }
});

test('another base branch and partial references do not mutate anything', async () => {
  for (const pr of [{ base: { ref: 'other' } }, { body: 'Refs #101' }]) {
    const f = fixture({ pr }); await synchronize(f);
    assert.equal(f.moves.length + f.writes.length, 0);
  }
});

test('missing credentials or denied Project access fail before mutations', async () => {
  const missing = fixture(); missing.projectGithub = null;
  await assert.rejects(synchronize(missing), /credential/);
  const denied = fixture(); denied.projectGithub.graphql = async () => { throw new Error('Access denied'); };
  await assert.rejects(synchronize(denied), /Access denied/);
  assert.equal(missing.moves.length + missing.writes.length + denied.moves.length + denied.writes.length, 0);
});

test('missing card is reported instead of silently adding it', async () => {
  const f = fixture({ missing: true });
  await assert.rejects(synchronize(f), /Missing project cards/);
  assert.equal(f.moves.length + f.writes.length, 0);
});

test('manual start moves Todo to In progress but preserves review and done', async () => {
  for (const status of ['Todo', 'In Review', 'Done']) {
    const f = fixture({ status }); await startIssue({ ...f, issueNumber: 101 });
    assert.equal(f.moves.length, status === 'Todo' ? 1 : 0);
    if (f.moves.length) assert.equal(f.moves[0].option, 'PROGRESS');
    assert.equal(f.writes.length, 0);
  }
});
