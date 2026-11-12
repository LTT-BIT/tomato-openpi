"""Cache sparse repair-model detections with frozen source/model identities; no policy training."""
import argparse,json,time,gzip
from pathlib import Path
import av,torch,cv2
from scripts.tomato_repair_locator import RgbLocator
from openpi.subtasks.rgb100 import digest
from openpi.shared.local_storage import new_artifact_directory
parser=argparse.ArgumentParser(description='Cache current-frame conservative hand detections; raw data read-only')
parser.add_argument('--selection',type=Path,required=True)
parser.add_argument('--locator',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--episodes',default='all')
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1];selection=args.selection;sel=json.loads(selection.read_text());model=args.locator
from openpi.subtasks.rgb100 import derived_output_path
if args.resume:
 root=args.out.expanduser().resolve()
 derived_output_path(sel,root/'_resume_path_guard')
else:root=derived_output_path(sel,args.out)
root.mkdir(exist_ok=args.resume)
episodes=set(range(100)) if args.episodes=='all' else set(map(int,args.episodes.split(',')))
if not episodes or not episodes<=set(range(100)):parser.error('Use episode indices0..99')
torch.set_num_threads(2);cv2.setNumThreads(1);loc=RgbLocator(model)
parent=Path(json.loads((model/'receipt.json').read_text())['parent_model_root'])
contract=dict(selection_sha256=digest(selection),locator_sha256=digest(model/'model.pt'),parent_locator_sha256=digest(parent/'model.pt'),repair_locator_code_sha256=digest(repo/'scripts/tomato_repair_locator.py'))
if (root/'contract.json').exists():
 if json.loads((root/'contract.json').read_text())!=contract:raise ValueError('Cache contract changed')
else:(root/'contract.json').write_text(json.dumps(contract,indent=2))
start=time.perf_counter()
for row in sel['episodes']:
 if row['episode_index'] not in episodes:continue
 existing=root/f"episode_{row['episode_index']:06d}"
 if existing.exists():
  receipt=json.loads((existing/'receipt.json').read_text())
  if receipt['contract']!=contract or receipt['frames']!=row['length'] or digest(existing/'detections.jsonl.gz')!=receipt['detections_sha256']:raise ValueError('Changed existing cache')
  continue
 ep=row['episode_index'];old=row['source_episode_index'];rel=f'videos/chunk-{old//1000:03d}/observation.images.cam0/episode_{old:06d}.mp4';source=Path(row['dataset'])/rel;assert digest(source)==row['source_sha256'][rel];count=0
 with new_artifact_directory(root/f'episode_{ep:06d}') as staging:
  with av.open(str(source)) as video,gzip.open(staging/'detections.jsonl.gz','wt') as dst:
   batch=[]
   def process(batch):
    global count
    predictions=loc.predict([x for _,x in batch])
    for (frame,_),ds in zip(batch,predictions,strict=True):dst.write(json.dumps(dict(source_frame=frame,detections=ds))+'\n');count+=1
   for frame,img in enumerate(video.decode(video=0)):
    batch.append((frame,img.to_ndarray(format='rgb24')))
    if len(batch)==32:process(batch);batch=[]
   if batch:process(batch)
  assert count==row['length'];assert digest(source)==row['source_sha256'][rel]
  (staging/'receipt.json').write_text(json.dumps(dict(contract=contract,episode_index=ep,frames=count,source_video_sha256=row['source_sha256'][rel],detections_sha256=digest(staging/'detections.jsonl.gz'),source_writes=False,training_ready=False),indent=2)+'\n')
 print(ep,count,round(time.perf_counter()-start,1),flush=True)
