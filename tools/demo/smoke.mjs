// Headed real-WebGPU qualification. Model outputs/metrics remain Rust code.
import { chromium } from 'playwright';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const url = process.env.DEMO_URL || 'http://localhost:9917/demo/';
const output = path.resolve(process.env.DEMO_OUTPUT || '.data/live-demo/browser');
const fixtures = path.resolve(process.env.DEMO_FIXTURES || '.data/live-demo/native');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: false, executablePath: process.env.CHROME || '/usr/bin/google-chrome',
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist',
    '--enable-features=Vulkan,ForceEnableWebGpuInterop', '--use-angle=vulkan', '--disable-gpu-watchdog'],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1 });
await page.addInitScript(() => {
  const OriginalWorker = window.Worker;
  window.Worker = class extends OriginalWorker {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', event => {
        if (event.data?.kind === 'output') window.gekkoLastPrediction = event.data.result;
      });
    }
  };
});
const errors = [], trace = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('console', message => { if (message.type() === 'error' && !message.text().includes('favicon.ico')) errors.push(message.text()); });
page.on('requestfailed', request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
const state = () => page.evaluate(() => window.gekkoDemo);
const wait = predicate => page.waitForFunction(predicate, null, { timeout: 240000, polling: 500 });
const ready = () => wait(() => window.gekkoDemo?.result_revision != null && !window.gekkoDemo.busy);
async function record(name) {
  const value = await state();
  trace.push({ name, time: new Date().toISOString(), state: value });
  await page.screenshot({ path: path.join(output, `${name}.png`) });
  await writeFile(path.join(output, 'trace.json'), JSON.stringify(trace, null, 2));
  console.log(name, JSON.stringify({ revision: value.revision, psnr: value.psnr, status: value.status }));
  return value;
}
try {
  await page.goto(`${url}?auto_infer=1`, { waitUntil: 'domcontentloaded' });
  await ready();
  let previous = await record('scene');
  assert.equal(previous.cameras, 3);
  assert.equal(previous.has_camera_truth, true);
  assert.ok(Number.isFinite(previous.psnr) && previous.matches > 0);
  await page.locator('#bevy').focus();
  await page.keyboard.press('2');
  await wait(() => window.gekkoDemo?.target === 1 && window.gekkoDemo.result_revision == null);
  await page.keyboard.press('Space');
  await ready();
  previous = await record('camera-2');
  await page.keyboard.press('0');
  await page.waitForTimeout(1200);
  await record('editor-overview');
  await page.keyboard.press('2');
  await page.waitForTimeout(400);
  await page.mouse.move(450, 440);
  await page.mouse.down();
  await page.mouse.move(520, 480, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(600);
  await page.keyboard.press('c');
  await wait(() => window.gekkoDemo?.result_revision == null);
  const moved = await state();
  assert.notDeepEqual(moved.camera_signature, previous.camera_signature);
  await page.keyboard.press('Space');
  await ready();
  await record('camera-moved');
  // Native room captures exercise the real user-facing file picker, without labels.
  const chooser = page.waitForEvent('filechooser');
  await page.keyboard.press('u');
  await (await chooser).setFiles([0, 1, 2].map(i => path.join(fixtures, `input-${i}.png`)));
  await wait(() => window.gekkoDemo?.mode === 'Images' && window.gekkoDemo.result_revision == null);
  await page.keyboard.press('Space');
  await ready();
  const uploaded = await record('uploaded-images');
  assert.equal(uploaded.has_camera_truth, false);
  const native = JSON.parse(await readFile(path.join(fixtures, 'receipt.json'), 'utf8')).result;
  const raw = await page.evaluate(() => window.gekkoLastPrediction);
  await writeFile(path.join(output, 'uploaded-prediction.json'), JSON.stringify(raw));
  assert.deepEqual(raw.visible, native.visible, 'Compare identical native and browser target masks');
  const psnrDelta = Math.abs(uploaded.psnr - native.rgb_score.psnr_db);
  const cameraMaxDelta = Math.max(...uploaded.camera.map((x, i) => Math.abs(x - native.camera[i])));
  assert.ok(psnrDelta < 0.05, `Native/WebGPU PSNR difference ${psnrDelta} dB`);
  assert.ok(cameraMaxDelta < 0.005, `Native/WebGPU camera difference ${cameraMaxDelta}`);
  await page.keyboard.press('Space');
  await page.keyboard.press('2');
  await page.waitForTimeout(4000);
  const stale = await state();
  assert.equal(stale.target, 1);
  assert.equal(stale.result_revision, null, 'An old prediction must not annotate a new target');
  await page.keyboard.press('Space');
  await ready();
  await page.setViewportSize({ width: 1000, height: 720 });
  await page.waitForTimeout(800);
  await record('resized-uploads');
  await page.keyboard.press('n');
  await wait(() => window.gekkoDemo?.mode === 'Scene' && window.gekkoDemo.cameras === 3);
  await page.waitForTimeout(4000);
  await record('regenerated-room');
  assert.deepEqual(errors, [], 'Browser errors invalidate qualification');
  await writeFile(path.join(output, 'result.json'), JSON.stringify({ passed: true, url,
    browser: await browser.version(), psnrDelta, cameraMaxDelta, trace, errors }, null, 2));
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
  await writeFile(path.join(output, 'failure.json'), JSON.stringify({ error: String(error), state: await state().catch(() => null), errors, trace }, null, 2));
  throw error;
} finally {
  await browser.close();
}
