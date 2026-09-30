#!/usr/bin/env python3
"""Cache the author-hosted HPatches full sequences and prepare RGB-only inputs.

Evaluation-only data; ground-truth homographies are kept in a separate artifact.
No models, checkpoints, or executable code are downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import tomllib
import urllib.request
import zipfile
import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    c = tomllib.loads(a.config.read_text())
    root = Path(c['root'])
    assert root.resolve().is_relative_to(Path('.data').resolve())
    root.mkdir(parents=True, exist_ok=True)
    archive = root/'hpatches-sequences-release.zip'
    if not archive.exists():
        stage = archive.with_suffix('.zip.partial')
        start = time.monotonic()
        # A cache-busting query avoids expired signed redirects from HTTP caches.
        url = c['url'] + '?download=true&cache=' + str(int(time.time()))
        with urllib.request.urlopen(url, timeout=60) as src, stage.open('wb') as out:
            size = 0
            while block := src.read(4 << 20):
                size += len(block)
                assert size <= c['bytes'], 'archive exceeded pinned size'
                out.write(block)
                if time.monotonic()-start > 1800:
                    raise TimeoutError('bounded archive download exceeded 30 minutes')
        assert size == c['bytes'] and digest(stage) == c['sha256'], 'archive checksum mismatch'
        stage.replace(archive)
    assert archive.stat().st_size == c['bytes'] and digest(archive) == c['sha256']
    extracted = root/'raw'
    if not extracted.exists():
        stage = root/'raw.partial'
        stage.mkdir(exist_ok=False)
        with zipfile.ZipFile(archive) as z:
            assert sum(i.file_size for i in z.infolist()) < 8_000_000_000
            for info in z.infolist():
                assert (stage/info.filename).resolve().is_relative_to(stage.resolve()), 'unsafe archive path'
                assert (info.external_attr >> 16) & 0o170000 != 0o120000, 'symlink in archive'
            z.extractall(stage)
        stage.rename(extracted)
    sequences = sorted(p.parent for p in extracted.rglob('H_1_2'))
    assert len(sequences) == 116, f'expected 116 full sequences, got {len(sequences)}'
    prepared = root/f"rgb-{c['image_size']}"
    prepared.mkdir(exist_ok=True)
    scenes, geometry = [], {}
    for folder in sequences:
        views = []
        sizes = []
        for i in range(1, 7):
            image = cv2.imread(str(folder/f'{i}.ppm'), cv2.IMREAD_COLOR)
            assert image is not None
            sizes.append(list(image.shape[:2]))
            image = cv2.cvtColor(cv2.resize(image,(c['image_size'],c['image_size']),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
            file = prepared/f'{folder.name}-{i}.f32'
            (image.astype('<f4')/255.).tofile(file)
            views.append(dict(file=str(file.resolve()),sha256=digest(file),original_hw=sizes[-1]))
        for i in range(2,7):
            matrix = np.loadtxt(folder/f'H_1_{i}',dtype=np.float64)
            assert matrix.shape==(3,3) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix))>1e-12
            geometry[f'{folder.name}_{i}'] = matrix
        scenes.append(dict(name=folder.name,views=views))
    np.savez(root/'homographies.npz',**geometry)
    manifest = dict(schema=1,dataset='HPatches full sequences',image_size=c['image_size'],sequences=scenes,
        archive_sha256=c['sha256'],source_url=c['url'],ground_truth='separate homographies.npz; never model input',
        protocol='RGB bilinear resize to square; all 116 sequences; reference image 1, queries 2..6; no sequence exclusions',
        purpose='external evaluation only; never training or model selection',
        rights='Images retain original source terms; see HPatches references.txt. Not redistributed with model weights.')
    (root/'images.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'prepare.toml').write_text(a.config.read_text())
    print(json.dumps(dict(sequences=len(scenes),images=len(scenes)*6,manifest=str(root/'images.json'),
        manifest_sha256=digest(root/'images.json'),geometry_sha256=digest(root/'homographies.npz'))))


if __name__ == '__main__':
    main()
