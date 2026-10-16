import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './e2e', timeout: 90000, workers: 1, fullyParallel: false,
  use: { baseURL: 'http://127.0.0.1:8092', headless: true, viewport: { width: 1440, height: 1000 },
    launchOptions: { channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] },
    screenshot: 'only-on-failure', trace: 'retain-on-failure',
  },
  webServer: { command: '../.venv/bin/python ../tests/serve_e2e.py', url: 'http://127.0.0.1:8092/api/health', reuseExistingServer: false, timeout: 30000 },
});
