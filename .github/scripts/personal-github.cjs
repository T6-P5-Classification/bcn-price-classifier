function issueNumbers(body = '') {
  const result = new Set();
  let fence = null;
  // Comments and fenced examples are documentation, not completion directives.
  for (const line of body.replace(/<!--[\s\S]*?-->/g, '').split(/\r?\n/)) {
    const marker = line.match(/^\s*(`{3,}|~{3,})/);
    if (marker) {
      if (!fence) fence = marker[1];
      else if (marker[1][0] === fence[0] && marker[1].length >= fence.length) fence = null;
      continue;
    }
    if (fence) continue;
    const match = line.match(/^\s*(?:[-*]\s+)?(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#([1-9]\d*)\s*$/i);
    if (match && Number.isSafeInteger(Number(match[1]))) result.add(Number(match[1]));
  }
  return [...result];
}

function validateConfig(config, repository) {
  const p = config.project;
  if (config.repository !== repository || !config.integration_branch ||
      !p || !['organization', 'user'].includes(p.owner_type) || !p.owner ||
      !Number.isSafeInteger(p.number) || p.number < 1 ||
      !p.status_field || !p.progress_status || !p.review_status || !p.done_status ||
      new Set([p.progress_status, p.review_status, p.done_status]).size !== 3) {
    throw new Error('Invalid lifecycle configuration or repository mismatch.');
  }
}

async function projectMetadata(client, p) {
  const result = await client.graphql(`query LifecycleProject($owner: String!, $number: Int!, $field: String!) {
    ${p.owner_type}(login: $owner) {
      projectV2(number: $number) { id field(name: $field) {
        ... on ProjectV2SingleSelectField { id options { id name } }
      } }
    }
  }`, { owner: p.owner, number: p.number, field: p.status_field });
  const project = result[p.owner_type]?.projectV2;
  const options = project?.field?.options || [];
  const review = options.find((o) => o.name === p.review_status);
  const done = options.find((o) => o.name === p.done_status);
  const progress = options.find((o) => o.name === p.progress_status);
  if (!project?.id || !project.field?.id || !review || !done || !progress) {
    throw new Error('Cannot access configured Project/Status/options; check names, token permissions, and owner type.');
  }
  return { id: project.id, fieldId: project.field.id, review: review.id, done: done.id, progress: progress.id };
}

async function projectItem(client, issueId, projectId, field) {
  let after = null;
  do {
    const result = await client.graphql(`query LifecycleItems($id: ID!, $after: String, $field: String!) {
      node(id: $id) { ... on Issue { projectItems(first: 100, after: $after) {
        nodes { id isArchived project { id } fieldValueByName(name: $field) {
          ... on ProjectV2ItemFieldSingleSelectValue { name }
        } }
        pageInfo { hasNextPage endCursor }
      } } }
    }`, { id: issueId, after, field });
    const items = result.node?.projectItems;
    if (!items) throw new Error('Issue project memberships are inaccessible.');
    const found = items.nodes.find((item) => item?.project?.id === projectId);
    if (found) return found;
    if (!items.pageInfo.hasNextPage) return null;
    if (!items.pageInfo.endCursor || items.pageInfo.endCursor === after) throw new Error('Invalid project pagination cursor.');
    after = items.pageInfo.endCursor;
  } while (after);
  return null;
}

function targetsIntegration(pr, config) {
  return pr.base.ref === config.integration_branch && pr.base.repo.full_name === config.repository;
}

async function readyForReview(github, repo, pr) {
  if (pr.state !== 'open' || pr.draft) return false;
  const reviews = await github.paginate(github.rest.pulls.listReviews, { ...repo, pull_number: pr.number, per_page: 100 });
  const latest = new Map();
  for (const review of reviews.slice().sort((a, b) => a.id - b.id)) {
    if (review.user?.login && ['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'].includes(review.state)) {
      latest.set(review.user.login, review.state);
    }
  }
  const requested = new Set((pr.requested_reviewers || []).map((user) => user.login));
  // A push/comment alone does not clear requested changes. Explicitly requesting
  // the blocking person's review again is the handoff back to review.
  return ![...latest].some(([user, state]) => state === 'CHANGES_REQUESTED' && !requested.has(user));
}

async function updateCard(client, project, item, option) {
  await client.graphql(`mutation LifecycleStatus($project: ID!, $item: ID!, $field: ID!, $option: String!) {
    updateProjectV2ItemFieldValue(input: {
      projectId: $project, itemId: $item, fieldId: $field, value: { singleSelectOptionId: $option }
    }) { projectV2Item { id } }
  }`, { project: project.id, item: item.id, field: project.fieldId, option });
}

async function startIssue({ github, projectGithub, context, core, config, issueNumber }) {
  if (!/^[1-9]\d*$/.test(String(issueNumber)) || !Number.isSafeInteger(Number(issueNumber))) throw new Error('Provide a positive issue number.');
  const repo = context.repo;
  validateConfig(config, `${repo.owner}/${repo.repo}`);
  const { data: issue } = await github.rest.issues.get({ ...repo, issue_number: Number(issueNumber) });
  if (issue.pull_request || issue.state !== 'open') { core.info('Not an open issue; skipped.'); return; }
  if (!projectGithub) throw new Error('Missing project-write credential.');
  const project = await projectMetadata(projectGithub, config.project);
  const item = await projectItem(projectGithub, issue.node_id, project.id, config.project.status_field);
  if (!item) throw new Error('Missing project card; add the issue and rerun.');
  if (item.isArchived || [config.project.done_status, config.project.review_status, config.project.progress_status].includes(item.fieldValueByName?.name)) return;
  await updateCard(projectGithub, project, item, project.progress);
  core.info(`Started #${issueNumber}: ${config.project.progress_status}.`);
}

async function synchronize({ github, projectGithub, context, core, config, prNumber }) {
  if (!/^[1-9]\d*$/.test(String(prNumber)) || !Number.isSafeInteger(Number(prNumber))) {
    throw new Error('Provide a positive PR number.');
  }
  const repo = context.repo;
  const fullName = `${repo.owner}/${repo.repo}`;
  validateConfig(config, fullName);
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: Number(prNumber) });
  if (!targetsIntegration(pr, config)) {
    core.info('Different repository/integration branch; skipped.');
    return;
  }
  const merged = pr.state === 'closed' && pr.merged === true;
  const numbers = issueNumbers(pr.body || '');
  if (!numbers.length) { core.info('No completion directives; skipped.'); return; }
  if (!projectGithub) throw new Error('Missing project-write credential.');
  // Validate project configuration before closing issues or moving cards.
  const project = await projectMetadata(projectGithub, config.project);
  const readyCache = new Map();
  const isReady = async (candidate) => {
    if (!readyCache.has(candidate.number)) readyCache.set(candidate.number, await readyForReview(github, repo, candidate));
    return readyCache.get(candidate.number);
  };
  const ready = merged ? false : await isReady(pr);
  let openPRs;
  const anotherReadyPR = async (issueNumber) => {
    if (!openPRs) openPRs = await github.paginate(github.rest.pulls.list, {
      ...repo, state: 'open', base: config.integration_branch, per_page: 100,
    });
    for (const candidate of openPRs) {
      if (candidate.number === Number(prNumber) || !issueNumbers(candidate.body || '').includes(issueNumber)) continue;
      const { data: current } = await github.rest.pulls.get({ ...repo, pull_number: candidate.number });
      if (targetsIntegration(current, config) && issueNumbers(current.body || '').includes(issueNumber) && await isReady(current)) return true;
    }
    return false;
  };
  const missing = [];
  for (const issue_number of numbers) {
    const params = { ...repo, issue_number };
    const { data: issue } = await github.rest.issues.get(params);
    if (issue.pull_request || (!merged && issue.state !== 'open')) {
      core.info(`#${issue_number} is not an eligible issue; skipped.`);
      continue;
    }
    const item = await projectItem(projectGithub, issue.node_id, project.id, config.project.status_field);
    if (!item) {
      core.warning(`#${issue_number} is not readable on the configured board. Add/check its card and rerun.`);
      missing.push(issue_number);
    }
    if (merged && issue.state !== 'closed') {
      await github.rest.issues.update({ ...params, state: 'closed', state_reason: 'completed' });
      core.info(`Closed #${issue_number} after PR #${prNumber} merged.`);
    }
    if (!item || item.isArchived) continue;
    const current = item.fieldValueByName?.name;
    if (!merged && current === config.project.done_status) continue;
    const review = !merged && (ready || await anotherReadyPR(issue_number));
    const target = merged ? config.project.done_status : review ? config.project.review_status : config.project.progress_status;
    if (current === target) continue;
    await updateCard(projectGithub, project, item, merged ? project.done : review ? project.review : project.progress);
    core.info(`Moved #${issue_number} to ${target}.`);
  }
  if (missing.length) throw new Error(`Missing project cards: ${missing.map((n) => `#${n}`).join(', ')}. Other eligible operations may have completed; reconcile and rerun.`);
}

module.exports = { issueNumbers, validateConfig, synchronize, startIssue, readyForReview };
