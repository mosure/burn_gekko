#!/usr/bin/env python3
"""Prepare the official ETH3D interval pairs with RGB and labels kept separate.

Archives are cached under .data first. This reads only a restricted, audited
numeric pickle schema from the author's pair bundle, never arbitrary classes.
"""
import argparse
import hashlib
import io
import json
import pickle
from pathlib import Path
import tomllib
import zipfile
import cv2
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


class NumericPairs(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('numpy','dtype'):return np.dtype
        if (module,name)==('numpy.core.multiarray','scalar'):return np._core.multiarray.scalar
        raise pickle.UnpicklingError(f'Unsupported pair type: {module}.{name}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();c=tomllib.loads(a.config.read_text());root=Path(c['root'])
    assert root.resolve().is_relative_to(Path('.data').resolve())
    assert not (root/'images.json').exists(), 'preserve previous manifest'
    for archive in c['archives']:
        file=root/archive['file']
        assert file.stat().st_size==archive['bytes'] and digest(file)==archive['sha256']
    rgb=root/'rgb-256';rgb.mkdir(exist_ok=True)
    pairs=[];images={};geometry={}
    with zipfile.ZipFile(root/'info_ETH3D_files.zip') as z:
        for name in sorted(z.namelist()):
            if name.endswith('/'):continue
            scene,interval=Path(name).name.split('_every_5_rate_of_');interval=int(interval)
            assert interval in [3,5,7,9,11,13,15]
            rows=NumericPairs(io.BytesIO(z.read(name))).load()
            assert isinstance(rows,list)
            for number,row in enumerate(rows):
                pair=dict(id=f'{scene}-{interval}-{number:04}',scene=scene,interval=interval)
                for key,field in [('reference','source_image'),('target','target_image')]:
                    path=Path(row[field]);file=root/path
                    assert file.resolve().is_relative_to(root.resolve()) and file.is_file()
                    image_id=hashlib.sha256(str(path).encode()).hexdigest()[:24]
                    if image_id not in images:
                        original=cv2.imread(str(file),cv2.IMREAD_COLOR);assert original is not None
                        out=rgb/f'{image_id}.f32'
                        resized=cv2.cvtColor(cv2.resize(original,(256,256),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
                        (resized.astype('<f4')/255.).tofile(out)
                        images[image_id]=dict(file=str(out.resolve()),sha256=digest(out),original_hw=list(original.shape[:2]),source=str(path),source_sha256=digest(file))
                    pair[key]=image_id
                assert images[pair['target']]['original_hw']==images[pair['reference']]['original_hw']
                points=np.stack([np.asarray(row[k],np.float32) for k in ['Xs','Ys','Xt','Yt']],axis=1)
                assert points.ndim==2 and points.shape[1]==4 and len(points)>0 and np.isfinite(points).all()
                h,w=images[pair['target']]['original_hw']
                assert (np.rint(points[:,2])>=0).all() and (np.rint(points[:,2])<w).all()
                assert (np.rint(points[:,3])>=0).all() and (np.rint(points[:,3])<h).all()
                geometry[pair['id']]=points;pairs.append(pair)
    assert len(pairs)==3365 and len(images)==2448
    assert len({p['scene'] for p in pairs})==10
    # No labels, cameras, depth or point coordinates in the inference manifest.
    manifest=dict(schema=1,dataset='ETH3D official interval pairs',image_size=256,images=images,pairs=pairs,
        protocol='All 3365 pairs, 10 scenes, seven intervals; RGB square bilinear resize. Labels in separate NPZ.',
        role='independent qualification; no model fitting or selection',sources=c['archives'])
    (root/'images.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.savez_compressed(root/'correspondences.npz',**geometry)
    (root/'prepare.toml').write_text(a.config.read_text())
    result=dict(images=len(images),pairs=len(pairs),images_sha256=digest(root/'images.json'),geometry_sha256=digest(root/'correspondences.npz'))
    (root/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
