"""Repair an experimental RGB hand locator using reviewed hard cases and original references.

Only locator training. Current/future phase labels are never inference inputs.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch,av
from PIL import Image
from torchvision.transforms import functional as F
from scripts.tomato_roi_locator import make_locator_model
from openpi.subtasks.rgb100 import digest,derived_output_path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--selection',type=Path,required=True)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--hard-cases',type=Path,required=True)
parser.add_argument('--parent',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--steps',type=int,default=500)
args=parser.parse_args()
if args.steps<1:parser.error('Positive steps required')
selection=args.selection;sel=json.loads(selection.read_text());ann=args.reference;spec=json.loads(ann.read_text());newann=args.hard_cases;hard=json.loads(newann.read_text())
assert hard['selection_sha256']==digest(selection)
new=hard['frames'];keys=[(r['episode_index'],r['source_frame']) for r in new]
args.out=derived_output_path(sel,args.out)
hard_sources={Path(r['source_video']):r['source_sha256'] for r in new};hard_images={}
for source,expected in hard_sources.items():
 assert digest(source)==expected
 requested={(r['episode_index'],r['source_frame']) for r in new if Path(r['source_video'])==source}
 with av.open(str(source)) as c:
  for i,f in enumerate(c.decode(video=0)):
   for key in requested:
    if key[1]==i:hard_images[key]=f.to_image()
 assert requested<=hard_images.keys()
images={};labels={}
for clip in spec['clips']:
 for k in clip['keyframes']:
  boxes=[];ls=[]
  for role,b in k['boxes'].items():
   c={'right_hand':1,'left_hand':2,'left_hand_and_carried_fruit':2,'truss':3,'target':3,'held_target':3}.get(role)
   if c:boxes.append(b);ls.append(c)
  labels[(86,k['frame'])]=(boxes,ls)
row=sel['episodes'][86];old=row['source_episode_index'];source=Path(row['dataset'])/f'videos/chunk-{old//1000:03d}/observation.images.cam0/episode_{old:06d}.mp4'
assert digest(source)==row['source_sha256'][str(source.relative_to(row['dataset']))]
with av.open(str(source)) as c:
 for i,f in enumerate(c.decode(video=0)):
  if (86,i) in labels:images[(86,i)]=f.to_image()
oldkeys=sorted(images);newkeys=[]
for j,item in enumerate(new):
 key=tuple(keys[j]);newkeys.append(key);images[key]=hard_images[key];pairs=[(b,{'right_hand':1,'left_hand':2,'held_object':3}[role]) for role,b in item['boxes'].items()];labels[key]=([b for b,_ in pairs],[c for _,c in pairs])
random.seed(1042);torch.manual_seed(1042);torch.set_num_threads(4)
parent=args.parent;out=args.out;out.mkdir(exist_ok=False)
model=make_locator_model();model.load_state_dict(torch.load(parent/'model.pt',map_location='cpu',weights_only=True)['state_dict']);model=model.cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4);losses=[];start=time.perf_counter()
for step in range(args.steps):
 chosen=random.sample(oldkeys,2)+random.sample(newkeys,2);xs=[];ys=[]
 for key in chosen:
  im=images[key];boxes=np.array(labels[key][0],np.float32);ls=np.array(labels[key][1],np.int64);s=random.uniform(.85,1.15);dx=random.randint(-32,32);dy=random.randint(-16,16)
  im=F.affine(im,angle=0,translate=[dx,dy],scale=s,shear=0);boxes=(boxes-np.array([320,200,320,200]))*s+np.array([320+dx,200+dy,320+dx,200+dy]);boxes=np.clip(boxes,0,[640,400,640,400]);ok=(boxes[:,2]-boxes[:,0]>2)&(boxes[:,3]-boxes[:,1]>2)
  im=F.adjust_brightness(im,random.uniform(.85,1.15));im=F.adjust_contrast(im,random.uniform(.85,1.15));xs.append(F.pil_to_tensor(im).float().div(255).cuda());ys.append(dict(boxes=torch.tensor(boxes[ok],dtype=torch.float32,device='cuda'),labels=torch.tensor(ls[ok],device='cuda')))
 opt.zero_grad(set_to_none=True);parts=model(xs,ys);loss=sum(parts.values());assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
 if step%100==0:print(step,losses[-1],time.perf_counter()-start,flush=True)
model.eval();torch.save(dict(state_dict=model.state_dict(),classes={1:'right_hand',2:'left_hand',3:'held_object'},annotation_sha256=digest(newann)),out/'model.pt')
receipt=dict(status='experimental_training_finished_no_independent_quality_acceptance',parent_model_sha256=digest(parent/'model.pt'),model_sha256=digest(out/'model.pt'),annotation_sha256=digest(newann),original_annotation_sha256=digest(ann),selection_sha256=digest(selection),hard_case_frames=len(newkeys),original_reference_frames=len(oldkeys),steps=args.steps,batch_size=4,seed=1042,source_writes=False,not_pi05_training=True,training_ready=False,independent_quality_accepted=False,elapsed_seconds=time.perf_counter()-start,loss_initial=losses[0],loss_final=losses[-1])
receipt['trainer_sha256']=digest(Path(__file__))
receipt['parent_model_root']=str(parent.resolve())
for source,expected in hard_sources.items():
 assert digest(source)==expected
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
