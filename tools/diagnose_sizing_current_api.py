import argparse, os, sys, json, traceback
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

CKPT = ROOT / 'models/test107/multi_checkpoint_iter_300.pt'
OUT = ROOT / 'sizing_current_api_diagnostic.json'
EPS = 1e-8

def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {'positive': int((x > EPS).sum()), 'negative': int((x < -EPS).sum()),
            'zero': int((np.abs(x) <= EPS).sum()), 'min': float(x.min()),
            'median': float(np.median(x)), 'max': float(x.max()),
            'spread': float(x.max() - x.min())}

def current_slot_weights(agent, state, iteration):
    # Mirrors _hierarchical_sizing through construction and masking, before sampling.
    encoded = agent._encode_state_for_sizing(state, int(state.current_player)).astype(np.float32)
    st = torch.from_numpy(encoded).to(agent.device)
    anchor_floor = agent._current_sizing_min_prob(iteration)
    with torch.inference_mode():
        anchor_logits_t = agent.advantage_sizing_net(st.unsqueeze(0))
    anchor_logits = anchor_logits_t[0].cpu().numpy()
    probs = regret_matching_anchors(anchor_logits, min_prob=anchor_floor)
    flow = 'anchor_logits -> regret_matching_anchors -> mask -> normalize'
    if agent.sizing_anchor_availability_enabled:
        avail, allin = agent._anchor_availability(state)
        probs[~avail] = 0.0
    else:
        avail, allin = agent._legal_sizing_anchor_mask(state), None
        probs = agent._apply_sizing_anchor_mask(probs, state, callsite='diagnostic')
    probs = np.clip(probs.astype(np.float64), 0, None)
    probs /= max(probs.sum(), 1e-12)
    probs[-1] = max(1.0 - probs[:-1].sum(), 0.0)
    return st.unsqueeze(0), probs, avail, allin, flow

def generate_report(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    # Direct state-dict load avoids restoring replay buffers; _load_checkpoint is not used.
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    agent = DeepCFRAgent(player_id=0, num_players=6, device='cpu')
    agent.sizing_q_net.load_state_dict(ckpt['sizing_q_net'])
    agent.advantage_sizing_net.load_state_dict(ckpt['advantage_sizing_net'])
    agent.sizing_q_net.eval(); agent.advantage_sizing_net.eval()
    iteration = int(ckpt.get('iteration', 300))
    rows=[]; all_raw=[]
    states = controlled_s3c_multiboard_states()
    for i, state in enumerate(states, 1):
        st, weights, avail, allin, flow = current_slot_weights(agent, state, iteration)
        q = agent._evaluate_sizing_q_for_sizes(st, agent.anchors_arr).astype(np.float64)
        ev = float(np.dot(weights, q))
        raw = q - ev
        rows.append({'state': i, 'q_anchor': q.tolist(), 'slot_weights': weights.tolist(),
                     'ev_sizing': ev, 'raw_q_anchor_minus_ev_sizing': raw.tolist(),
                     'sign_counts': stats(raw), 'available_anchor_mask': np.asarray(avail, dtype=bool).tolist(),
                     'allin_anchor': None if allin is None else int(allin)})
        all_raw.extend(raw.tolist())
    report={'checkpoint': str(checkpoint), 'checkpoint_iteration': iteration, 'state_count': len(rows),
            'load_method': 'direct sizing_q_net and advantage_sizing_net state_dict only; replay buffers not restored',
            'target_flow_verified': 'Source lines 2454-2460 and 2706-2715: q_anchor is evaluated, selected/effective anchor may be grounded, ev_sizing = dot(slot_weights_np, q_anchor), then sizing_cf_regrets = q_anchor - ev_sizing. _hierarchical_sizing lines 2168-2237 constructs slot weights via regret_matching_anchors before this target flow.',
            'slot_weight_flow': flow, 'anchors': [float(x) for x in agent.anchors_arr],
            'states': rows, 'aggregate_raw_q_anchor_minus_ev_sizing': stats(all_raw), 'total_values': len(all_raw)}
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(CKPT.relative_to(ROOT)))
    parser.add_argument('--output', help='Explicit output path')
    args = parser.parse_args()
    report = generate_report(args.checkpoint)
    iteration = extract_iteration(report, args.checkpoint)
    output = report_path(OUT, iteration=iteration, explicit_output=args.output)
    if not output.is_absolute():
        output = ROOT / output
    with open(output, 'w', encoding='utf-8') as f: json.dump(report, f, indent=2)
    print(json.dumps({'output': str(output), 'state_count': report['state_count'],
                      'aggregate': report['aggregate_raw_q_anchor_minus_ev_sizing'],
                      'total_values': report['total_values']}, indent=2))

if __name__ == '__main__':
    try: main()
    except Exception:
        traceback.print_exc(); raise
