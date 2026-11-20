"""Create sampled visual QA and phase-consistency diagnostics; phase labels never generate ROIs."""
import argparse
import gzip
import json
from pathlib import Path
from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path,digest
from scripts.materialize_tomato100_four_view import collect_candidates


def main():
    from PIL import Image,ImageDraw
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    selection=json.loads(a.selection.read_text());contract,episodes=collect_candidates(a.candidates)
    out=derived_output_path(selection,a.out);gallery=[];summary=[];flagged=[]
    with new_artifact_directory(out) as staging:
        for ep,(folder,receipt) in sorted(episodes.items()):
            row=selection['episodes'][ep]
            qa=set(range(0,row['length'],max(1,row['length']//28)))|{row['length']-1}
            qa.update((s['start_frame']+s['end_frame']-1)//2 for s in row['training_segments'])
            required={s['start_frame']+j for s in row['training_segments'] if s['label']!='harvest' for j in range(s['end_frame']-s['start_frame'])}
            wrong=[];frames=[];qa_order=[];bad_examples=0;invalid_run=0;max_invalid=0
            with gzip.open(folder/'index.jsonl.gz','rt') as stream:
                for line in stream:
                    e=json.loads(line);f=e['source_frame'];frames.append(e)
                    if f in qa or (not e['valid'] and bad_examples<3):
                        qa_order.append(f)
                        if not e['valid']:bad_examples+=1
                    if f in required and e['valid'] and 1 not in e['roles']:wrong.append(f)
                    invalid_run=0 if e['valid'] else invalid_run+1;max_invalid=max(max_invalid,invalid_run)
            if len(qa_order)!=receipt['qa_images']:raise ValueError('QA page/frame mapping mismatch')
            for phase in [0,1,6]:
                segment=row['training_segments'][phase];f=(segment['start_frame']+segment['end_frame']-1)//2
                k=qa_order.index(f);page=Image.open(folder/f'qa_{k//12:02d}.jpg')
                j=k%12;x=(j%2)*544+320;y=(j//2)*230
                crop=page.crop((x,y,x+224,y+224));tile=Image.new('RGB',(224,250),'#151515');tile.paste(crop,(0,26))
                label=f"ep{ep} stage{phase+1} f{f} valid={frames[f]['valid']}"
                ImageDraw.Draw(tile).text((3,3),label,fill='white');gallery.append(tile)
            if wrong:flagged.append(dict(episode_index=ep,frames=len(wrong),first_frames=wrong[:10]))
            summary.append(dict(episode_index=ep,frames=row['length'],algorithm_valid_frames=receipt['algorithm_valid_frames'],
                invalid_frames=receipt['algorithm_invalid_frames'],longest_invalid_run_frames=max_invalid,
                nonharvest_valid_frames_without_right_hand=len(wrong)))
        for start in range(0,len(gallery),30):
            page=gallery[start:start+30];image=Image.new('RGB',(672,250*((len(page)+2)//3)))
            for j,tile in enumerate(page):image.paste(tile,((j%3)*224,(j//3)*250))
            image.save(staging/f'gallery_{start//30:02d}.jpg',quality=95)
        result=dict(selection_sha256=digest(a.selection),contract=contract,episodes=100,frames=sum(r['frames'] for r in summary),
            algorithm_valid_frames=sum(r['algorithm_valid_frames'] for r in summary),invalid_frames=sum(r['invalid_frames'] for r in summary),
            representative_gallery_samples=len(gallery),gallery_stages=[1,2,7],
            nonharvest_valid_frames_without_right_hand=sum(r['nonharvest_valid_frames_without_right_hand'] for r in summary),
            phase_labels_used='Offline QA only; never supplied to locator/selector',
            diagnostic_scope='Missing expected right hand is a potential semantic mismatch, not an independently labeled accuracy measure.',
            flagged_episodes=flagged,episode_summary=summary,quality_accepted=False,training_ready=False,source_writes=False)
        (staging/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print({k:result[k] for k in ['episodes','frames','algorithm_valid_frames','invalid_frames','representative_gallery_samples','nonharvest_valid_frames_without_right_hand']})


if __name__=='__main__':main()
