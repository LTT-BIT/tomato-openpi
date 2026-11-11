"""Prepare all100 selection/token receipts only; never launches training.

Run with PYTHONPATH=src python -m scripts.prepare_tomato100_rgb selection ...
or tokens ... . Every output must be new. No source datasets are rewritten.
"""
import argparse
import json
from pathlib import Path

from openpi.subtasks.rgb100 import audit_tokens, digest, freeze_selection, resolve_saved_labels, write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    selection=sub.add_parser('selection')
    selection.add_argument('--manifest',type=Path,required=True)
    selection.add_argument('--out',type=Path,required=True)
    tokens=sub.add_parser('tokens')
    tokens.add_argument('--tokenizer',type=Path,required=True)
    tokens.add_argument('--state-bins-jsonl',type=Path)
    tokens.add_argument('--out',type=Path,required=True)
    resolution=sub.add_parser('resolve-labels')
    resolution.add_argument('--selection',type=Path,required=True)
    resolution.add_argument('--proposals',type=Path,required=True)
    resolution.add_argument('--out',type=Path,required=True)
    resolution.add_argument('--plan-out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new output file')
    if args.command=='selection':
        result=freeze_selection(args.manifest)
        print('Verified100 recordings selected for action training; ROI/normalization still pending')
    elif args.command=='resolve-labels':
        if args.plan_out.exists() or args.plan_out.resolve()==args.out.resolve():
            raise FileExistsError('Choose two distinct new output files')
        result,plan=resolve_saved_labels(args.selection,args.proposals)
        write_new(args.out,result)
        plan['selection_sha256']=digest(args.out)
        write_new(args.plan_out,plan)
        print('Resolved to saved source labels; retained provisional boundaries:',result['boundary_resolution']['boundaries'])
        return
    else:
        import sentencepiece as spm
        sp=spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        rows=None
        if args.state_bins_jsonl:
            rows=[json.loads(line) for line in args.state_bins_jsonl.read_text().splitlines() if line.strip()]
        result=audit_tokens(sp.encode,tokenizer_sha256=digest(args.tokenizer),normalized_rows=rows)
        print('Real tokenizer audited; actual100 normalized distribution:',result['real_distribution_available'])
    write_new(args.out,result)


if __name__=='__main__':
    main()
