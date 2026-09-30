#!/usr/bin/env python3
"""Inspect spatial displacement of frozen HPatches predictions without geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    config=tomllib.loads(parser.parse_args().config.read_text())
    source=Path(config['predictions']);output=Path(config['output'])
    assert output.resolve().is_relative_to(Path('.data').resolve())
    groups={}
    for line in source.read_text().splitlines():
        row=json.loads(line)
        assert row['grid']==[16,16] and row['model_image_size']==256
        subset='viewpoint' if row['sequence'].startswith('v_') else 'illumination'
        indices=np.asarray(row['indices']);query=np.arange(256)
        assert indices.shape==(256,) and (indices>=0).all() and (indices<256).all()
        distance=np.hypot(indices%16-query%16,indices//16-query//16)*16
        groups.setdefault((row['method'],subset),[]).append(dict(
            exact_same_position=float(np.mean(indices==query)),
            within_one_patch=float(np.mean(distance<=16)),
            mean_displacement_pixels=float(distance.mean())))
    result=dict(scope='post-evaluation diagnostic; no training or checkpoint selection; all readouts and sequences retained',
                model_image_size=256,predictions_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),methods={})
    for (method,subset),rows in groups.items():
        assert len(rows)==(295 if subset=='viewpoint' else 285)
        result['methods'].setdefault(method,{})[subset]={key:float(np.mean([row[key] for row in rows])) for key in rows[0]}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
