import init, { WebInference, model_files } from './inference/burn_gekko_inference.js';
let engine;
let busy = false;
const progress = message => postMessage({kind: 'progress', message});
const initialized = init();
async function download(root, file) {
  const response = await fetch(`${root}/${file.name}`);
  if (!response.ok) throw new Error(`Model download failed: ${response.status} ${file.name}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== file.bytes) throw new Error(`Model size mismatch: ${file.name}`);
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const sha = Array.from(digest, x => x.toString(16).padStart(2, '0')).join('');
  if (sha !== file.sha256) throw new Error(`Model checksum mismatch: ${file.name}`);
  return bytes;
}
self.onmessage = async ({data}) => {
  if (busy) { postMessage({kind: 'error', message: 'Inference worker is busy.'}); return; }
  busy = true;
  try {
    await initialized;
    if (data.kind === 'load') {
      const root = data.root.replace(/\/$/, '');
      const response = await fetch(`${root}/manifest.toml`);
      if (!response.ok) throw new Error(`Model manifest unavailable (${response.status})`);
      const manifest = await response.text();
      const bundle = model_files(manifest);
      const parts = bundle.foundation;
      const joined = new Uint8Array(parts.reduce((n, x) => n + x.bytes, 0));
      let offset = 0;
      for (let i = 0; i < parts.length; i++) {
        progress(`Loading weights ${i + 1}/${parts.length}`);
        const bytes = await download(root, parts[i]);
        joined.set(bytes, offset); offset += bytes.length;
      }
      const camera = await download(root, bundle.camera);
      const rgb = await download(root, bundle.rgb);
      progress('Initializing WebGPU inference device…');
      engine = await WebInference.load(manifest, joined, camera, rgb);
      postMessage({kind: 'ready'});
    } else if (data.kind === 'infer') {
      if (!engine) throw new Error('Load the model first.');
      const result = await engine.infer(data.request);
      postMessage({kind: 'output', result});
    } else { throw new Error('Unknown inference request.'); }
  } catch (error) {
    postMessage({kind: 'error', message: String(error?.stack || error)});
  } finally { busy = false; }
};
