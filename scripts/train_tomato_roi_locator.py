"""Train an isolated experimental ROI locator from bound manual keyframes; never starts pi05."""
import json,random,time,hashlib
from pathlib import Path
import av,torch,numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import functional as F
from openpi.subtasks.rgb100 import digest, derived_output_path
from scripts.prepare_tomato100_roi import worklist
import argparse

def main():
    parser=argparse.ArgumentParser(description='Experimental palm/finger/held-object locator; not pi05 training')
    parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--steps',type=int,default=1200)
    args=parser.parse_args()
    if args.steps<=0:parser.error('Positive steps required')
    random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    selection=json.loads(args.selection.read_text())
    ann=args.annotations;spec=json.loads(ann.read_text())
    listing=worklist(selection,spec,preview_sha256=digest(ann))
    row=next(r for r in listing['episodes'] if r['roi_clips'])
    relative=f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
    source=Path(row['dataset'])/relative;expected=row['source_sha256'][relative]
    assert digest(source)==expected
    labels={}
    for clip in spec['clips']:
     for key in clip['keyframes']:
      target=[]
      for role,box in key['boxes'].items():
       if role=='right_hand':target.append((box,1))
       if role in ['left_hand','left_hand_and_carried_fruit']:target.append((box,2))
       if role in ['truss','target','held_target']:target.append((box,3))
      labels[key['frame']]=target
    images={}
    with av.open(str(source)) as c:
     for i,frame in enumerate(c.decode(video=0)):
      if i in labels:images[i]=frame.to_image()
    assert len(images)==len(labels)
    out=derived_output_path(selection,args.out);out.mkdir(exist_ok=False)
    (out/'experiment.json').write_text(json.dumps(dict(status='running',source_sha256=expected,annotation_sha256=digest(ann),training_frames=len(images),heldout_recordings_not_evaluated=True,not_pi05_training=True,source_writes=False),indent=2))
    weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights,trainable_backbone_layers=3)
    features=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(features,4)
    model=model.to('cuda');model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.00015,weight_decay=0.0001)
    keys=sorted(images);losses=[];started=time.perf_counter()
    for step in range(args.steps):
     chosen=random.sample(keys,4);xs=[];ys=[]
     for k in chosen:
      im=images[k]
      # Color-only augmentation; keep hand identities and all annotated edges.
      im=F.adjust_brightness(im,random.uniform(.85,1.15));im=F.adjust_contrast(im,random.uniform(.85,1.15))
      xs.append(pil_to_tensor(im).float().div(255).to('cuda'))
      boxes,label=zip(*labels[k]);ys.append(dict(boxes=torch.tensor(boxes,dtype=torch.float32,device='cuda'),labels=torch.tensor(label,dtype=torch.int64,device='cuda')))
     optimizer.zero_grad(set_to_none=True);parts=model(xs,ys);loss=sum(parts.values())
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite locator loss')
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
     if step%100==0:print('step',step,'loss',round(losses[-1],4),'elapsed',round(time.perf_counter()-started,1),flush=True)
    model.eval();torch.save(dict(state_dict=model.state_dict(),classes={1:'right_hand',2:'left_hand',3:'held_object'},annotation_sha256=digest(ann)),out/'model.pt')
    assert digest(source)==expected
    (out/'receipt.json').write_text(json.dumps(dict(status='experimental_training_finished_no_independent_quality_acceptance',model='torchvision fasterrcnn_mobilenet_v3_large_320_fpn',torch=torch.__version__,steps=args.steps,batch_size=4,seed=42,train_frames=len(keys),episode_index=row['episode_index'],source_episode_index=row['source_episode_index'],annotation_sha256=digest(ann),source_sha256=expected,source_hash_unchanged=True,not_pi05_training=True,full_roi_coverage=False,training_ready=False,selection_sha256=digest(args.selection),uses_phase_labels_at_inference=False,classes={1:'right_hand',2:'left_hand',3:'held_object'},elapsed_seconds=time.perf_counter()-started,loss_initial=losses[0],loss_final=losses[-1]),indent=2))
    print('FINISHED',out,flush=True)


if __name__=='__main__':main()
