import { test, expect } from '@playwright/test';
import path from 'node:path';

const fixture = path.resolve(process.cwd(), '../backend/tests/fixtures/classification.csv');

async function startWorkspace(page) {
  await page.goto('/');
  const open = page.getByRole('button', { name: /open workspace/i });
  if (await open.count()) await open.click();
  await page.locator('input[type=file]').first().setInputFiles(fixture);
  await page.getByPlaceholder(/describe what you want/i).fill('Build a classification model');
  const workspaceResponsePromise = page.waitForResponse(response => response.url().includes('/api/workspaces') && response.request().method() === 'POST');
  await page.getByRole('button', { name: /start workflow/i }).click();
  const workspaceResponse = await workspaceResponsePromise;
  const workspace = await workspaceResponse.json();
  // Persist the identity exactly as the application does after workspace creation;
  // this also removes the network-response timing race in the browser harness.
  await page.evaluate(id => window.localStorage.setItem('active_workspace_id', id), workspace.id);
  return workspace.id;
}

async function createProposal(page, workspaceId, stageId = 'training') {
  const configuration = stageId === 'training'
    ? { configuration: { target: 'target', problem_type: 'classification', preprocessing: { missing_values: 'mean', scaling: null, outlier_handling: false, smote: false, pca: false, pca_components: null }, model_id: 'decision_tree_classifier', model_params: { max_depth: 5, min_samples_split: 2, min_samples_leaf: 1 } } }
    : { target: 'target', problem_type: 'classification', preprocessing: { missing_values: 'mean', scaling: null, outlier_handling: false, smote: false, pca: false, pca_components: null }, model_id: 'decision_tree_classifier', model_params: { max_depth: 5, min_samples_split: 2, min_samples_leaf: 1 } };
  const response = await page.request.post('/api/ai/workflow-proposals', { data: { workspace_id: workspaceId, stage_id: stageId, configuration } });
  if (!response.ok()) throw new Error(`proposal create ${response.status()}: ${await response.text()}`);
  return (await response.json()).proposal;
}

async function openApprovalCard(page) {
  const orb = page.locator('.ai-companion.companion .ai-companion-orb');
  await orb.click();
  await expect(page.getByRole('region', { name: 'Workflow approval' })).toBeVisible();
}

test('YES executes once and displays the actual backend output', async ({ page }) => {
  const workspaceId = await startWorkspace(page);
  const proposal = await createProposal(page, workspaceId);
  await openApprovalCard(page);
  let decisionRequests = 0;
  page.on('request', request => { if (request.url().includes('/workflow-proposals/') && request.url().endsWith('/decision')) decisionRequests += 1; });
  await page.getByRole('button', { name: /YES — Execute/ }).click();
  await expect(page.getByText(/Actual backend result/)).toBeVisible({ timeout: 30_000 });
  expect(decisionRequests).toBe(1);
  const state = await page.request.get(`/api/ai/workflow-proposals/${proposal.proposal_id}`);
  expect((await state.json()).proposal.status).toBe('completed');
  const runs = await page.request.get(`/api/workspaces/${workspaceId}/runs`);
  expect((await runs.json()).runs).toHaveLength(1);
});

test('NO records rejection and sends no execution request', async ({ page }) => {
  const workspaceId = await startWorkspace(page);
  await createProposal(page, workspaceId, 'model_configuration');
  await openApprovalCard(page);
  let executionRequests = 0;
  page.on('request', request => { if (request.url().includes('/api/workspaces/') && request.request().method() === 'POST' && !request.url().includes('/column-selection')) executionRequests += 1; });
  await page.getByRole('button', { name: /NO — Cancel/ }).click();
  await expect(page.getByText(/Proposal cancelled/)).toBeVisible();
  expect(executionRequests).toBe(0);
});

test('JUST FILL updates editable workflow controls without execution', async ({ page }) => {
  const workspaceId = await startWorkspace(page);
  await createProposal(page, workspaceId, 'model_configuration');
  await openApprovalCard(page);
  let executionRequests = 0;
  page.on('request', request => { if (request.url().includes('/api/workspaces/') && request.request().method() === 'POST' && !request.url().includes('/column-selection')) executionRequests += 1; });
  await page.getByRole('button', { name: /JUST FILL/ }).click();
  await expect(page.getByText(/Configuration filled into the workflow/)).toBeVisible();
  expect(executionRequests).toBe(0);
  await expect(page.getByText('AI-POPULATED CONFIGURATION')).toBeVisible();
  const parameter = page.locator('input[type=number]').first();
  await expect(parameter).toHaveValue('5');
  await parameter.fill('6');
  await expect(parameter).toHaveValue('6');
});

test('refresh restores an awaiting proposal without duplicating it', async ({ page }) => {
  const workspaceId = await startWorkspace(page);
  const proposal = await createProposal(page, workspaceId, 'model_configuration');
  await page.reload();
  await openApprovalCard(page);
  await expect(page.getByText(proposal.rationale)).toBeVisible();
  expect(await page.locator('.ai-approval-card').count()).toBe(1);
});
