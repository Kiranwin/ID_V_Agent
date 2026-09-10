"""Read-only M29 label metrics, image ablations and predicted-state interventions."""
import argparse
import json
from pathlib import Path

import torch

from idv_agent.model.m29_checkpoint import load_m29_checkpoint
from idv_agent.training.m29_dataset import M29Dataset
from idv_agent.training.m29_features import load_m29_encoder
from idv_agent.scripts.train_m29 import evaluate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--model-path')
    args=parser.parse_args(argv)
    if args.output.exists():raise ValueError('output exists; use a new report path')
    model,manifest=load_m29_checkpoint(args.checkpoint,device=args.device)
    dataset=M29Dataset(args.data)
    encoder=load_m29_encoder(args.model_path or manifest['base_model'],args.device)
    normal=evaluate(model,dataset,encoder,args.device)
    def zero(paths):return torch.zeros((len(paths),model.feature_dim))
    report={'schema':'m29.evaluation.v1','checkpoint':str(args.checkpoint),
            'files':dataset.files,'coverage':dataset.coverage(),'normal':normal,
            'image_feature_zero':evaluate(model,dataset,zero,args.device),
            'acceptance':'diagnostic_only; no automatic deployment authorization'}
    groups={}
    for sample,row in zip(dataset.samples,dataset.records):
        for path in sample['paths']:groups[path]=row['scenario_group']
    replacements={}
    for path,group in groups.items():
        alternatives=sorted(p for p,g in groups.items() if g!=group)
        if not alternatives:break
        replacements[path]=alternatives[sum(path.encode())%len(alternatives)]
    if len(replacements)==len(groups):
        report['cross_group_image_shuffle']=evaluate(model,dataset,lambda paths:encoder([replacements[p] for p in paths]),args.device)
    else:
        report['cross_group_image_shuffle']={'status':'unavailable','reason':'requires independent scenario groups'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'coverage':dataset.coverage()}))


if __name__=='__main__':main()
