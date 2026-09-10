import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from idv_agent.scripts.train_m29 import train
from idv_agent.training.m29_dataset import M29Dataset,collate_m29
from idv_agent.model.state_guided_action import StateGuidedActionNetwork,PHASES
from idv_agent.model.m29_checkpoint import load_m29_checkpoint
from idv_agent.training.m29_loss import compute_m29_loss
from idv_agent.scripts.prepare_mvp_v6 import TRAIN_PROTOCOL
from idv_agent.scripts.import_mvp_anylabeling import import_completed
from idv_agent.scripts.assemble_m29_dataset import assemble, canonicalize_mvp_row
from test_import_mvp_anylabeling import _workspace


class Encoder:
    adapter=SimpleNamespace(act_feature_dim=8)
    def __call__(self,paths):
        # Fixed generated features independent of target values.
        return torch.stack([torch.arange(8).float().sin()+int(Path(path).stem)*.01 for path in paths])


def _data(tmp_path):
    _,workspace=_workspace(tmp_path)
    out=tmp_path/'import'
    import_completed(workspace,out,annotator='a',completion_note='complete',split='train',scenario_group='scene-a')
    return out/'prompt_q_all_frames.jsonl'


def test_real_export_shape_and_masks_can_run_new_head_and_loss(tmp_path):
    path=_data(tmp_path)
    dataset=M29Dataset(path,expected_split='train')
    assert dataset.coverage()['q']==50
    assert dataset.coverage()['navigation']==0
    inputs,targets,masks=collate_m29([dataset[0],dataset[-1]],Encoder(),device='cpu')
    core=StateGuidedActionNetwork(8,16)
    loss=compute_m29_loss(core(**inputs),targets,masks)
    loss['total'].backward()
    assert loss['phase']==0 and loss['move']==0
    assert core.prompt_q.weight.grad.abs().sum()>0


def test_train_cli_path_checkpoint_and_reload_use_identical_model(tmp_path,monkeypatch):
    path=_data(tmp_path)
    import idv_agent.scripts.train_m29 as module
    monkeypatch.setattr(module,'load_m29_encoder',lambda *args:Encoder())
    args=SimpleNamespace(data=str(path),val_data=None,output=str(tmp_path/'run'),
                         task='q',device='cpu',model_path='fake_test_encoder',seed=0,
                         hidden_dim=16,batch_size=4,lr=.001,steps=2)
    result=train(args)
    core,manifest=load_m29_checkpoint(tmp_path/'run'/'m29.pt')
    assert result['reload_delta']==0
    assert manifest['task']=='q' and manifest['deployable'] is False
    assert manifest['hierarchy']=='predicted_beliefs_only'
    assert core.hidden_dim==16


def test_full_training_rejects_q_only_data_before_loading_qwen(tmp_path):
    path=_data(tmp_path)
    args=SimpleNamespace(data=str(path),val_data=None,task='full',seed=0,device='cpu')
    with pytest.raises(ValueError,match='complete state/navigation'):
        train(args)


def test_m29_rejects_old_checkpoint_and_edited_q_target(tmp_path):
    old=tmp_path/'old.pt';torch.save({'schema':'m28'},old)
    with pytest.raises(ValueError,match='not M29'):
        load_m29_checkpoint(old)
    path=_data(tmp_path)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['targets']['q']=True
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='disagrees'):
        M29Dataset(path)


def test_full_v6_state_targets_and_past_q_feedback_are_consumed(tmp_path):
    source=_data(tmp_path)
    qrows=[json.loads(line) for line in source.read_text().splitlines()]
    raw=Path(qrows[0]['model_input']['image_path']).parent.parent
    selected=qrows[20:23]
    hashes={f"frames/{Path(row['model_input']['image_path']).name}":row['audit_only']['image_sha256'] for row in selected}
    row={'schema':'idv.mvp_training.v6','id':'ep:00000022','split':'train','scenario_group':'scene-a','protocol':TRAIN_PROTOCOL,
         'model_input':{'raw_root':str(raw),'history_frames':[20,21,22],
                        'history_timestamps_ns':[1_000_000_000,1_050_000_000,1_100_000_000],
                        'observed_ns':1_100_000_000,'task':'decipher',
                        'last_executed_q_ns':1_000_000_000},
         'targets':{'facts':{'cipher_visibility':'visible','target_bbox':[.1,.1,.4,.4],
                             'prompt_bbox':None,'decoding':True,'interact_prompt':False,'path':'direct'},
                    'decision':{'phase':'maintain_decode','steering':'hold'},
                    'action':{'source':'accept_replay','q':False,'move':0,'camera_command':[0,0]}},
         'masks':{'q':True,'navigation':True,'phase':True,'visibility':True,'bbox':True,
                  'prompt_bbox':False,'decoding':True,'steering':True,'path':True},
         'audit_only':{'raw_sha256':hashes}}
    path=tmp_path/'full.jsonl';path.write_text(json.dumps(row)+'\n')
    dataset=M29Dataset(path)
    inputs,targets,masks=collate_m29([dataset[0]],Encoder(),device='cpu')
    assert targets['phase'][0]==PHASES.index('maintain_decode')
    assert masks['decoding'][0] and targets['decoding'][0]==1
    assert inputs['execution_feedback'][0,0]==1
    output=StateGuidedActionNetwork(8,16)(**inputs)
    losses=compute_m29_loss(output,targets,masks)
    losses['total'].backward()
    assert all(torch.isfinite(loss) for loss in losses.values())


def test_all_frame_q_and_full_endpoint_merge_without_double_counting(tmp_path):
    q_path = _data(tmp_path)
    q_rows = [json.loads(line) for line in q_path.read_text().splitlines()]
    selected = q_rows[20:23]
    raw = Path(selected[-1]['model_input']['image_path']).parent.parent
    full = {
        'schema': 'idv.mvp_training.v6', 'id': selected[-1]['id'],
        'split': 'train', 'scenario_group': 'scene-a', 'protocol': TRAIN_PROTOCOL,
        'model_input': {
            'raw_root': str(raw), 'history_frames': [20, 21, 22],
            'history_timestamps_ns': [row['model_input']['observed_ns'] for row in selected],
            'observed_ns': selected[-1]['model_input']['observed_ns'], 'task': 'decipher',
            'last_executed_q_ns': None,
        },
        'targets': {
            'facts': {'cipher_visibility': 'absent', 'target_bbox': None, 'prompt_bbox': None,
                      'decoding': False, 'interact_prompt': False, 'path': 'direct'},
            'decision': {'phase': 'search', 'steering': 'search_sweep'},
            'action': {'source': 'accept_replay', 'q': False, 'move': 0, 'camera_command': [0, 0]},
        },
        'masks': {'q': True, 'navigation': True, 'phase': True, 'visibility': True,
                  'bbox': False, 'prompt_bbox': False, 'decoding': True,
                  'steering': True, 'path': True},
        'audit_only': {'raw_sha256': {
            f"frames/{Path(row['model_input']['image_path']).name}": row['audit_only']['image_sha256']
            for row in selected
        }},
    }
    full_path = tmp_path / 'full.jsonl'
    full_path.write_text(json.dumps(full) + '\n')
    dataset = M29Dataset(f"{q_path},{full_path}")
    assert len(dataset) == len(q_rows)
    merged = next(sample for sample in dataset.samples if sample['id'] == full['id'])
    assert not merged['q_only'] and merged['masks']['navigation']
    output = tmp_path / 'assembled.jsonl'
    report = assemble([q_path, full_path], output)
    assert report['rows'] == len(q_rows)
    assert M29Dataset(output).coverage()['navigation'] == 1


def test_assembly_canonicalizes_prompt_endpoints_to_interact_and_no_navigation(tmp_path):
    source = _data(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    # Build a minimal full endpoint with an intentionally stale approach phase.
    q = rows[-1]
    raw = Path(q['model_input']['image_path']).parent.parent
    full = {
        'schema': 'idv.mvp_training.v6', 'id': q['id'], 'split': 'train',
        'scenario_group': 'scene-a', 'protocol': TRAIN_PROTOCOL,
        'model_input': {'raw_root': str(raw), 'history_frames': [47, 48, 49],
                        'history_timestamps_ns': [1, 2, 3], 'observed_ns': 3,
                        'task': 'decipher', 'last_executed_q_ns': None},
        'targets': {'facts': {'cipher_visibility': 'visible', 'target_bbox': [.1,.1,.2,.2],
                              'prompt_bbox': [.2,.2,.3,.3], 'decoding': False,
                              'interact_prompt': True, 'path': 'direct'},
                    'decision': {'phase': 'approach', 'steering': 'hold'},
                    'action': {'source': 'accept_replay', 'q': True, 'move': 1,
                               'camera_command': [0, 0]}},
        'masks': {'q': True, 'navigation': True, 'phase': True, 'visibility': True,
                  'bbox': True, 'prompt_bbox': True, 'decoding': True,
                  'steering': True, 'path': True},
        'audit_only': {'raw_sha256': {f'frames/{Path(q["model_input"]["image_path"]).name}': q['audit_only']['image_sha256']}},
    }
    result = canonicalize_mvp_row(full)
    assert result['targets']['decision']['phase'] == 'interact'
    assert result['targets']['action']['source'] == 'exclude'
    assert result['masks']['navigation'] is False


def test_discrete_state_bottleneck_has_semantic_forward_values_and_gradients():
    core = StateGuidedActionNetwork(8, 16)
    output = core(**{
        'features': torch.randn(2, 3, 8),
        'valid_mask': torch.ones(2, 3, dtype=torch.bool),
        'relative_times_s': torch.tensor([[-.4, -.2, 0.], [-.4, -.2, 0.]]),
        'execution_feedback': torch.zeros(2, 2),
    })
    categorical = output.beliefs[:, :4]
    assert torch.allclose(categorical.sum(-1), torch.ones(2), atol=1e-6)
    assert not torch.allclose(categorical.detach(), torch.nn.functional.one_hot(
        categorical.argmax(-1), 4).float(), atol=1e-7, rtol=0)
    output.navigation.move_logits.sum().backward()
    assert core.visibility.weight.grad is not None


def test_uncertain_visibility_preserves_bbox_for_navigation_state_embedding():
    core = StateGuidedActionNetwork(8, 16)
    output = core(**{
        'features': torch.randn(2, 3, 8),
        'valid_mask': torch.ones(2, 3, dtype=torch.bool),
        'relative_times_s': torch.tensor([[-.4, -.2, 0.], [-.4, -.2, 0.]]),
        'execution_feedback': torch.zeros(2, 2),
    })
    hidden = output.facts.probabilities()
    assert torch.isfinite(hidden[:, 4:8]).all()
    assert (hidden[:, 4:8].abs().sum(-1) > 0).all()
