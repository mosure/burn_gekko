#!/usr/bin/env python3
"""Export exact official PyTorch forwards for strict Burn parity and masking audit."""
import argparse,importlib.util,inspect,json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--sample',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(6);torch.set_num_interop_threads(1);sys.path.insert(0,str(args.source))
spec=importlib.util.spec_from_file_location('gekko_reference',args.source/'models/gekko.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
c=json.loads((args.package/'config.json').read_text());allowed=set(inspect.signature(mod.Gekko).parameters)
with torch.device('meta'):model=mod.Gekko(**{k:v for k,v in c.items() if k in allowed})
model.load_state_dict(load_file(args.package/'model.safetensors'),strict=True,assign=True);model.eval()
meta=json.loads((args.sample/'sample.json').read_text());h,w=meta['height'],meta['width'];vis=meta['visible_patch_ids'];args.output.mkdir(parents=True,exist_ok=False)
(args.output/'fixture.json').write_text(json.dumps(dict(height=h,width=w,visible=vis)))
target=np.fromfile(args.sample/'target.f32',dtype='<f4').reshape(h,w,3).copy();ref=np.fromfile(args.sample/'reference-0.f32',dtype='<f4').reshape(h,w,3).copy()
target.tofile(args.output/'target.f32');ref.tofile(args.output/'reference.f32')
mask=torch.ones(1,h*w//256,dtype=torch.bool);mask[:,vis]=False
hidden=mask.numpy().reshape(h//16,w//16).repeat(16,0).repeat(16,1)
intervened=target.copy();intervened[hidden]=np.random.default_rng(39).random((hidden.sum(),3));intervened.tofile(args.output/'intervened.f32')
mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
image=lambda a:(torch.from_numpy(a).permute(2,0,1).unsqueeze(0)-mean)/std
out=args.output/'torch';out.mkdir()
def save(name,t):t.contiguous().numpy().astype('<f4').tofile(out/f'{name}.f32')
with torch.inference_mode():
 sparse,pos,m=model._encode_image(image(target),do_mask=True,masks=mask);full,_,_=model._encode_image(image(target));reference,rpos,_=model._encode_image(image(ref))
 cross=model._decode(sparse,pos,m,model.croco_mask_token,reference,rpos)
 mae=model._decode(sparse,pos,m,model.mae_mask_token)
 dense=model._decode(full,pos,None,model.croco_mask_token,reference,rpos)
 for name,t in dict(sparse=sparse,full=full,reference=reference,**{'cross-features':cross},cross=model.croco_head(cross),mae=model.mae_head(mae),dense=model.croco_head(dense)).items():save(name,t)
 print('official fixture written',args.output)
