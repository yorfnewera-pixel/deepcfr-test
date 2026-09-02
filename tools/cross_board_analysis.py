import argparse, json, math, itertools
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from tools.sizing_report_naming import extract_iteration, report_path
def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else root / path

parser = argparse.ArgumentParser(description='Analyze cross-board sizing diagnostics.')
parser.add_argument('--checkpoint', help='Generate diagnostic data from this checkpoint before analysis.')
parser.add_argument('--input', default='sizing_current_api_diagnostic.json',
                    help='Diagnostic JSON to analyze when --checkpoint is not used.')
parser.add_argument('--output', help='Explicit output JSON path; the text report uses the same stem with .txt.')
args = parser.parse_args()
inp = project_path(args.input)
if args.checkpoint:
    from tools.diagnose_sizing_current_api import generate_report
    data = generate_report(project_path(args.checkpoint))
    inp = project_path(args.checkpoint)
else:
    data = json.loads(inp.read_text(encoding='utf-8'))
iteration = extract_iteration(data, inp)
outj = report_path(root / 'cross_board_analysis.json', iteration=iteration, explicit_output=args.output)
if not outj.is_absolute(): outj = root / outj
outt = outj.with_suffix('.txt')
def find_records(x):
    if isinstance(x, list) and x and all(isinstance(v, dict) for v in x):
        keys = set().union(*(v.keys() for v in x))
        if any('regret' in k.lower() or 'slot_weight' in k.lower() for k in keys): return x
    if isinstance(x, dict):
        for v in x.values():
            r = find_records(v)
            if r: return r
    return None
records = find_records(data)
if not records: raise ValueError('Could not locate board records')
def key_for(d, terms):
    return next((k for k in d if all(t in k.lower() for t in terms)), None)
rk, wk = key_for(records[0], ['raw','q','anchor']), key_for(records[0], ['slot','weight'])
if not rk or not wk: raise ValueError(f'Missing keys: {list(records[0])}')
def flat(v):
    if isinstance(v, (int,float)): return [float(v)]
    if isinstance(v, list): return [z for a in v for z in flat(a)]
    return []
R, W = [flat(x[rk]) for x in records], [flat(x[wk]) for x in records]
n = len(R); L = min(min(map(len,R)), min(map(len,W)))
if n < 2 or not L: raise ValueError('Insufficient boards or anchors')
R, W = [x[:L] for x in R], [x[:L] for x in W]
def std(a):
    m=sum(a)/len(a); return math.sqrt(sum((x-m)**2 for x in a)/len(a))
def summary(a):
    a=sorted(a); q=lambda p:a[round((len(a)-1)*p)]
    return {'min':a[0],'p10':q(.1),'p25':q(.25),'median':q(.5),'mean':sum(a)/len(a),'p75':q(.75),'p90':q(.9),'max':a[-1]}
def cosine(a,b):
    aa=math.sqrt(sum(x*x for x in a)); bb=math.sqrt(sum(x*x for x in b))
    return None if not aa or not bb else sum(x*y for x,y in zip(a,b))/(aa*bb)
def corr(a,b):
    ma=sum(a)/len(a); mb=sum(b)/len(b)
    aa=math.sqrt(sum((x-ma)**2 for x in a)); bb=math.sqrt(sum((x-mb)**2 for x in b))
    return None if not aa or not bb else sum((x-ma)*(y-mb) for x,y in zip(a,b))/(aa*bb)
def regret_dist(a):
    z=[max(x,0) for x in a]; s=sum(z)
    return [x/s for x in z] if s else [1/len(z)]*len(z)
def weight_dist(a):
    z=[max(x,0) for x in a]; s=sum(z)
    return [x/s for x in z] if s else [1/len(z)]*len(z)
def tv(a,b): return .5*sum(abs(x-y) for x,y in zip(a,b))
pairs=list(itertools.combinations(range(n),2))
mad=[sum(abs(a-b) for a,b in zip(R[i],R[j]))/L for i,j in pairs]
cos=[v for i,j in pairs if (v:=cosine(R[i],R[j])) is not None]
cor=[v for i,j in pairs if (v:=corr(R[i],R[j])) is not None]
rtv=[tv(regret_dist(R[i]),regret_dist(R[j])) for i,j in pairs]
wtv=[tv(weight_dist(W[i]),weight_dist(W[j])) for i,j in pairs]
res={'input':str(inp),'boards':n,'anchors':L,'pairs':len(pairs),'per_anchor_population_std':{'raw_regrets':[std([r[k] for r in R]) for k in range(L)],'slot_weights':[std([w[k] for w in W]) for k in range(L)]},'raw_regret_pairwise':{'mean_absolute_difference':summary(mad),'cosine':summary(cos),'correlation':summary(cor)},'regret_matching_tv':summary(rtv),'stored_slot_weights_tv':summary(wtv)}
res['comparison']='Regret-matching TV is '+('lower' if res['regret_matching_tv']['mean'] < res['stored_slot_weights_tv']['mean'] else 'higher')+' on average than stored slot-weight TV.'
res['conclusion']='Descriptive cross-board variation only; it does not establish causal performance differences.'
outj.write_text(json.dumps(res,indent=2),encoding='utf-8')
outt.write_text('\n'.join([f'Cross-board analysis (iteration {iteration})',f'Input: {inp}',f'Boards: {n}; anchors: {L}; pairs: {len(pairs)}','',f"Raw regret pairwise MAD: {json.dumps(res['raw_regret_pairwise']['mean_absolute_difference'])}",f"Raw regret cosine: {json.dumps(res['raw_regret_pairwise']['cosine'])}",f"Raw regret correlation: {json.dumps(res['raw_regret_pairwise']['correlation'])}",f"Regret-matching TV: {json.dumps(res['regret_matching_tv'])}",f"Stored slot-weights TV: {json.dumps(res['stored_slot_weights_tv'])}",'',res['comparison'],res['conclusion']])+'\n',encoding='utf-8')
print(json.dumps({'json':str(outj),'text':str(outt),'boards':n,'anchors':L,'pairs':len(pairs),'regret_tv':res['regret_matching_tv'],'weights_tv':res['stored_slot_weights_tv']},indent=2))
