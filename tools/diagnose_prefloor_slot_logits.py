import argparse
import os, sys, json, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from core.deep_cfr import DeepCFRAgent, regret_matching_anchors
from solver.benchmark_multiway_river import controlled_s3c_multiboard_states
from tools.sizing_report_naming import extract_iteration, report_path

CKPT = ROOT / 'models' / 'test107' / 'multi_checkpoint_iter_300.pt'
POST = ROOT / 'sizing_current_api_diagnostic.json'
OUT_JSON = ROOT / 'prefloor_slot_logits_tv_iter300.json'
OUT_TXT = ROOT / 'prefloor_slot_logits_tv_iter300.txt'
EPS = 1e-12

def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= EPS:
        return np.full(len(x), 1.0 / len(x), dtype=np.float64)
    return x / s

def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {k: float(v) for k, v in {
        'min': np.min(a), 'p10': np.percentile(a, 10), 'p25': np.percentile(a, 25),
        'median': np.median(a), 'mean': np.mean(a), 'p75': np.percentile(a, 75),
        'p90': np.percentile(a, 90), 'max': np.max(a)}.items()}

def tv(a, b):
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())

def prefloor(agent, state):
    encoded = agent._encode_state_for_sizing(state, int(state.current_player)).astype(np.float32)
    st = torch.from_numpy(encoded).to(agent.device)
    with torch.inference_mode():
        anchor_t = agent.advantage_sizing_net(st.unsqueeze(0))
    anchor_logits = anchor_t[0].detach().cpu().numpy().astype(np.float64)
    dist = regret_matching_anchors(anchor_logits, min_prob=0.0)
    flow = 'regret_matching_anchors(anchor_logits, min_prob=0), before availability/mask/normalization'
    return anchor_logits, normalize(dist), flow

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(CKPT.relative_to(ROOT)))
    parser.add_argument('--json-output', help='Explicit JSON output path')
    parser.add_argument('--txt-output', help='Explicit TXT output path')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    post = json.loads(POST.read_text(encoding='utf-8'))
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    agent = DeepCFRAgent(player_id=0, num_players=6, device='cpu')
    agent.sizing_q_net.load_state_dict(ckpt['sizing_q_net'])
    agent.advantage_sizing_net.load_state_dict(ckpt['advantage_sizing_net'])
    agent.sizing_q_net.eval(); agent.advantage_sizing_net.eval()
    states = controlled_s3c_multiboard_states()
    if len(states) != 24 or len(post['states']) != 24:
        raise RuntimeError(f'Expected 24 states; generated={len(states)}, post={len(post["states"])}')
    rows=[]; pre=[]; postw=[]
    for i, state in enumerate(states):
        al, bl, bd, d, flow = prefloor(agent, state)
        w = normalize(post['states'][i]['slot_weights'])
        rows.append({'state': i+1, 'anchor_logits': al.tolist(), 'bucket_logits': bl.tolist(),
                     'prefloor_bucket_distribution': bd, 'prefloor_slot_distribution': d.tolist(),
                     'postfloor_slot_weights': w.tolist(), 'within_state_tv_prefloor_vs_postfloor': tv(d,w)})
        pre.append(d); postw.append(w)
    pre_pair=[]; post_pair=[]; deltas=[]; pairs=[]
    for i in range(24):
        for j in range(i+1,24):
            a=tv(pre[i],pre[j]); b=tv(postw[i],postw[j])
            pre_pair.append(a); post_pair.append(b); deltas.append(a-b)
            pairs.append({'state_a':i+1,'state_b':j+1,'prefloor_tv':a,'postfloor_tv':b,'prefloor_minus_postfloor_tv':a-b})
    report={'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/') if checkpoint.is_relative_to(ROOT) else str(checkpoint), 'checkpoint_iteration':int(ckpt.get('iteration',300)),
            'source_path':'advantage_sizing_net bucket/anchor logits -> regret_matching_anchors with min_prob=0.0 -> hierarchical multiply; captured before anchor/bucket floors and before availability/mask/normalization',
            'state_source':'controlled_s3c_multiboard_states()', 'state_count':24, 'pair_count':len(pairs),
            'anchors':[float(x) for x in agent.anchors_arr], 'flow':flow,
            'prefloor_pairwise_tv':summary(pre_pair), 'postfloor_pairwise_tv':summary(post_pair),
            'pairwise_tv_delta_prefloor_minus_postfloor':summary(deltas),
            'within_state_tv_prefloor_vs_postfloor':summary([r['within_state_tv_prefloor_vs_postfloor'] for r in rows]),
            'states':rows, 'pairs':pairs}
    iteration = extract_iteration(ckpt, checkpoint)
    out_json = report_path(OUT_JSON, iteration=iteration, explicit_output=args.json_output)
    out_txt = report_path(OUT_TXT, iteration=iteration, explicit_output=args.txt_output)
    if not out_json.is_absolute(): out_json = ROOT / out_json
    if not out_txt.is_absolute(): out_txt = ROOT / out_txt
    out_json.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=[f'Pre-floor slot-logit diagnostic (iteration {iteration})', f'Checkpoint: {report["checkpoint"]}',
           f'States: 24; pairs: {len(pairs)}', f'Source path: {report["source_path"]}', '',
           'Pre-floor pairwise TV: '+json.dumps(report['prefloor_pairwise_tv']),
           'Post-floor slot_weights pairwise TV: '+json.dumps(report['postfloor_pairwise_tv']),
           'Pairwise delta (pre-floor - post-floor): '+json.dumps(report['pairwise_tv_delta_prefloor_minus_postfloor']),
           'Within-state TV (pre-floor vs post-floor): '+json.dumps(report['within_state_tv_prefloor_vs_postfloor']), '']
    out_txt.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'json':str(out_json),'txt':str(out_txt),'states':24,'pairs':len(pairs),
                      'prefloor_pairwise_tv':report['prefloor_pairwise_tv'], 'postfloor_pairwise_tv':report['postfloor_pairwise_tv'],
                      'delta':report['pairwise_tv_delta_prefloor_minus_postfloor']},indent=2))
if __name__ == '__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        raise
