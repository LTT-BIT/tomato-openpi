"""Review actual native/crop frames for residual losses and previous semantic failures."""
import argparse
import json,gzip,collections
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.audit_tomato_stable_roi import find_episodes,entries
from scripts.review_tomato_stable_roi import sample_frames
from openpi.subtasks.rgb100 import digest
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--selection',type=Path,required=True)
parser.add_argument('--candidates',type=Path,required=True)
parser.add_argument('--accounting',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--reference-cases',type=Path)
args=parser.parse_args()
selection=args.selection;sel=json.loads(selection.read_text());root=args.candidates;folders=find_episodes(root)
if set(folders)!=set(range(100)):raise ValueError('All100 required')
report=json.loads(args.accounting.read_text())
from openpi.subtasks.rgb100 import derived_output_path
out=derived_output_path(sel,args.out);out.mkdir(exist_ok=False);cases=[]
for r in report['longest_remaining_invalid_runs']:cases.append(dict(episode_index=r['episode_index'],source_frame=(r['start']+r['end']-1)//2,reason=f"remaining_invalid_{r['frames']}frames"))
for ep,folder in folders.items():
 rows=entries(folder);worst=None
 for seg in sel['episodes'][ep]['training_segments']:
  if seg['label']=='harvest':continue
  for i in range(seg['start_frame'],seg['end_frame']):
   if rows[i]['valid'] and rows[i]['roles']==[2]:
    dist=min(i-seg['start_frame'],seg['end_frame']-1-i)
    if worst is None or dist>worst['boundary_distance']:worst=dict(episode_index=ep,source_frame=i,reason=f"potential_{seg['label']}_left_only",boundary_distance=dist)
 if worst:cases.append(worst)
prior=json.loads(args.reference_cases.read_text()) if args.reference_cases else {'cases':[]}
for c in prior['cases']:
 if c['reason'].startswith('potential_'):cases.append({**c,'reason':'prior_'+c['reason']})
requests=collections.defaultdict(list)
for c in cases:requests[c['episode_index']].append(c)
tiles=[];samples=[]
for ep,items in sorted(requests.items()):
 row=sel['episodes'][ep];old=row['source_episode_index'];source=Path(row['dataset'])/f'videos/chunk-{old//1000:03d}/observation.images.cam0/episode_{old:06d}.mp4';ids=[c['source_frame'] for c in items];native=sample_frames(source,ids);crops=sample_frames(folders[ep]/'interaction224.mp4',ids);rows=entries(folders[ep]);assert digest(source)==row['source_sha256'][str(source.relative_to(row['dataset']))]
 for c in items:
  i=c['source_frame'];r=rows[i];im=native[i].copy();d=ImageDraw.Draw(im);d.rectangle(r['display_crop_box'],outline='cyan',width=2)
  if r['observed_interaction_box'] is not None:d.rectangle(r['observed_interaction_box'],outline='yellow',width=2)
  im.thumbnail((320,200));tile=Image.new('RGB',(544,260),'#151515');tile.paste(im,(0,48));tile.paste(crops[i],(320,36));d=ImageDraw.Draw(tile);d.text((3,2),f"ep{ep} f{i} {c['reason']}",fill='white');d.text((3,17),f"valid={r['valid']} roles={r['roles']} grasp={r['left_grasp_state_confirmed']} carry={r['carry_evidence']}",fill='white');tiles.append(tile);samples.append({**c,'valid':r['valid'],'roles':r['roles'],'frame_reason':r['reason'],'source_video_sha256':digest(source)})
for page in range((len(tiles)+7)//8):
 sheet=Image.new('RGB',(1088,1040))
 for j,tile in enumerate(tiles[page*8:page*8+8]):sheet.paste(tile,((j%2)*544,(j//2)*260))
 sheet.save(out/f'gallery{page:02d}.jpg',quality=94)
(out/'samples.json').write_text(json.dumps(dict(selection_sha256=digest(selection),reference_cases_sha256=digest(args.reference_cases) if args.reference_cases else None,cases=samples,scope='Actual decoded native/current crop samples: remaining invalid runs and per-recording worst phase-role diagnostic. Not independent semantic accuracy.'),indent=2)+'\n');print(len(samples),'samples')
