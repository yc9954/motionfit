import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const fixture = JSON.parse(fs.readFileSync(path.resolve('../backend/data/demo/landmarks.json'), 'utf8'));
fixture.source = 'explicit_browser_regression_fixture';
async function describeTask(page: any, payload = '0.05', budget = '3000') {
  await page.goto('/');
  await expect(page.locator('.message')).toHaveCount(0);
  await page.locator('#constraints-panel').evaluate((el: HTMLDetailsElement) => el.open = true);
  await page.locator('#payload').fill(payload);
  await page.locator('#budget').fill(budget);
  await page.locator('#orientation').selectOption('free');
  await page.locator('#prompt').fill('Move a test object along the recorded path.');
  await page.locator('#send').click();
  await expect(page.getByText('Initial arm created.', { exact: false })).toBeVisible();
}
async function confirmAndImport(page: any) {
  await page.locator('#catalog-label').click();
  await expect(page.locator('#catalog-dialog')).toBeVisible();
  await page.locator('#confirm-catalog').click();
  await expect(page.locator('#catalog-dialog')).not.toBeVisible();
  await page.locator('#import-landmarks').setInputFiles({ name: 'recorded-pose-regression.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(fixture)) });
  await expect(page.getByText('Recorded pose data loaded and validated.', { exact: false })).toBeVisible();
}

test('empty workspace, real input verification, measured replay, acceptance and stale export', async ({ page, request }) => {
  const pageErrors: string[] = []; page.on('pageerror', error => pageErrors.push(error.message));
  await describeTask(page);
  await expect(page.locator('#chat-stream')).not.toContainText('238%');
  await expect(page.locator('#viewport-before canvas')).toHaveCount(1);
  await confirmAndImport(page);
  const id = await page.locator('#project-name').textContent();
  const initialProject = await (await request.get(`/api/projects/${id}`)).json();
  expect(initialProject.design.payload_kg).toBe(.05);
  expect(initialProject.target.source).not.toBe('synthetic_fallback');
  const started = page.waitForResponse(response => response.url().endsWith('/run') && response.request().method() === 'POST');
  await page.locator('#run').click();
  const runId = (await (await started).json()).run_id;
  await expect(page.locator('#result-state')).toContainText('numerical checks', { timeout: 60000 });
  const run = await (await request.get(`/api/runs/${runId}`)).json();
  expect(run.status).toBe('completed'); expect(run.passed).toBe(true);
  const project = await (await request.get(`/api/projects/${id}`)).json();
  await expect(page.locator('#chat-stream')).toContainText(`${project.loop_result.final_metrics.ee_rmse_mm.toFixed(2)} mm`);
  await expect(page.locator('#viewport-after canvas')).toHaveCount(1);
  await page.locator('#accept').click();
  await expect(page.locator('#before-label')).toHaveText('Applied design · measured replay');
  await page.locator('#approve').click();
  await page.locator('#engineer').fill('Browser Regression Engineer');
  await page.locator('#save-approval').click();
  await expect(page.locator('#export')).toHaveAttribute('aria-disabled', 'false');
  const zip = await request.get(`/api/projects/${id}/export.zip`);
  expect(zip.status()).toBe(200); expect(zip.headers()['content-type']).toContain('application/zip');
  await page.reload();
  await expect(page.locator('#before-label')).toHaveText('Applied design · measured replay');
  await expect(page.locator('#chat-stream')).toContainText('Browser Regression Engineer');
  await page.locator('#constraints-panel').evaluate((el: HTMLDetailsElement) => el.open = true);
  await page.locator('#payload').fill('20'); await page.locator('#prompt').fill('Increase payload to 20 kg.');
  await page.locator('#send').click();
  await expect(page.locator('#result-state')).toContainText('No passing design found', { timeout: 60000 });
  await expect(page.locator('#accept')).toBeDisabled();
  await expect(page.locator('#export')).toHaveAttribute('aria-disabled', 'true');
  expect((await request.get(`/api/projects/${id}/export.zip`)).status()).toBe(403);
  expect(pageErrors).toEqual([]);
});

test('invalid captured motion is rejected without a preset or passing result', async ({ page, request }) => {
  await describeTask(page, '0.7', '2000');
  const broken = structuredClone(fixture); broken.frames[0].wrist = { x: 1, y: 2, z: 3 };
  await page.locator('#import-landmarks').setInputFiles({ name: 'invalid-capture.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(broken)) });
  await expect(page.locator('#chat-stream')).toContainText('Could not complete this action');
  const id = await page.locator('#project-name').textContent();
  const project = await (await request.get(`/api/projects/${id}`)).json();
  expect(project.target).toBeNull(); expect(project.loop_result).toBeNull();
  await expect(page.locator('#run')).toBeDisabled();
});

test('mobile workspace exposes the task controls without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('#prompt')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/mobile-workspace.png', fullPage: true });
});
