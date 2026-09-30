#!/usr/bin/env python3
"""Diagnostic released Gekko inference. Patch-statistic displays are ORACLE.
Never promote these images as standalone predictions or as burn_gekko results.
"""
import argparse,inspect,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=4);args=p.parse_args()
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 sys.path.insert(0,str(args.source))
 spec=importlib.util.spec_from_file_location('gekko_reference',args.source/'models/gekko.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 c=json.loads((args.package/'config.json').read_text());allowed=set(inspect.signature(mod.Gekko).parameters)
 with torch.device('meta'):model=mod.Gekko(**{k:v for k,v in c.items() if k in allowed})
 model.load_state_dict(load_file(args.package/'model.safetensors'),strict=True,assign=True);model.eval()
 args.output.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();rows=[]
 for d in sorted(args.samples.glob('room-*'))[:args.limit]:
  m=json.loads((d/'sample.json').read_text());h,w=m['height'],m['width']
  read=lambda p:torch.from_numpy(np.fromfile(p,dtype='<f4').reshape(h,w,3).copy()).permute(2,0,1).unsqueeze(0)
  target=read(d/'target.f32');refs=[read(d/f'reference-{i}.f32') for i in range(m['reference_count'])]
  mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
  im=(target-mean)/std;mask=torch.ones(1,h*w//256,dtype=torch.bool);mask[:,m['visible_patch_ids']]=False
  with torch.inference_mode():
   patches=model.patchify(im);mu=patches.mean(-1,keepdim=True);sigma=(patches.var(-1,keepdim=True)+1e-6).sqrt()
   predictions=[]
   for ref in refs:
    f,p,_=model._encode_image((ref-mean)/std)
    out=model(im,f,p,do_mask=True,with_mae=True,masks=mask)
    pred=model.unpatchify(model.get_pixels(out['croco_out'])*sigma+mu,im)*std+mean
    predictions.append(pred)
   pred=torch.stack(predictions).mean(0)
   mono=model.unpatchify(out['mae_out']*sigma+mu,im)*std+mean
   hidden=mask.reshape(1,1,h//16,w//16).repeat_interleave(16,2).repeat_interleave(16,3).expand_as(target)
   mse=(pred-target).square()[hidden].mean().item();mono_mse=(mono-target).square()[hidden].mean().item()
  outdir=args.output/d.name;outdir.mkdir()
  target.permute(0,2,3,1).numpy().astype('<f4').tofile(outdir/'target.f32')
  pred.permute(0,2,3,1).numpy().astype('<f4').tofile(outdir/'prediction.f32')
  mono.permute(0,2,3,1).numpy().astype('<f4').tofile(outdir/'mono.f32')
  for i,(ref,pr) in enumerate(zip(refs,predictions)):
   ref.permute(0,2,3,1).numpy().astype('<f4').tofile(outdir/f'reference-{i}.f32')
   pr.permute(0,2,3,1).numpy().astype('<f4').tofile(outdir/f'pairwise-{i}.f32')
  m.update(hidden_rgb_mse=mse,mae_rgb_mse=mono_mse,oracle_statistics=True,method='Released Gekko-L, oracle hidden patch statistics, mean of pairwise RGB predictions')
  (outdir/'sample.json').write_text(json.dumps(m,indent=2)+'\n')
  rows.append(dict(room=m['room_seed'],mse=mse,mae_mse=mono_mse));print(rows[-1],flush=True)
 (args.output/'report.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,rows=rows,oracle_statistics=True),indent=2)+'\n')
if __name__=='__main__':main()
