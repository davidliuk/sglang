"""Recompute evidence from raw outputs; does not need CUDA or torch."""
import ast
import base64
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results'
BASE = '234ccba7f625e12639e621044496bcc6197c3758'
HEAD = 'fcdd5e12ac1865c94f73807a9dc5370518608e0c'
SHA = {'baseline_min_new_tokens.py': 'e27e0827a8c87cfed52abccb618546acce31a8db28dcd35ddafd7719094e3020', 'pr_min_new_tokens.py': 'a0713952f4421f25b38084a66d0fac155380411b6715519fc581b1c01771b97b', 'min_new_tokens_kernel.py': '971c7922c5b215b351200afabe0f78e7146335c9b2bad174a06eeb301f179f2a', 'test_penaltylib.py': '08cefba687479a52537777303e6d944e9b464841c5068369a82a22edf899d08a', 'test_min_new_tokens.py': '47cee6afacd1a8f94302303d07eec2369c3b4b9f525de2b34b31ba5dbb247d82'}
def read(path): return json.loads((R / path).read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(ROOT/'results.tar.gz') == 'b36c8a021941b75fcf4454b4987649382e2cdc27a320d56b5e319f7023449b0b'
for f, sha in SHA.items(): assert digest(R/f) == digest(ROOT/f) == sha, f
assert read('final-tests.json')['exit_code'] == 0
for f in ['final-tests.log', 'final/final-tests.log', 'final/final-tests-after-e2e.log']:
    assert '105 passed, 20 warnings, 23 subtests passed' in (R/f).read_text(), f
assert read('micro-checks.json')['exact_shapes'] == 24
assert read('gpu-correctness.json')['exact_checks'] == 65
for f in ['final', 'high', 'control']:
    assert (R/f/'DONE.json').exists() and not (R/f/'FAILED.json').exists()
for folder in ['final','high']:
    for p in (R/folder).glob('*-info.json'):
        info=json.loads(p.read_text()); assert info['source_sha']==BASE
        assert info['patch_sha256']==SHA['pr_min_new_tokens.py' if '-pr-' in p.name else 'baseline_min_new_tokens.py']
        assert info['kernel_sha256']==SHA['min_new_tokens_kernel.py']
        assert info['server']['dtype']=='bfloat16' and info['server']['tp_size']==1
    configs=[json.loads(p.read_text())['args'] for p in (R/folder).glob('*-info.json')]
    assert len(configs)==4 and all(c==configs[0] for c in configs)

def summarize(path, expected_groups, expected_responses):
    rows=read(path); assert len(rows)==expected_groups
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    total=0
    for row in rows:
        key=(row.get('mode','diagnostic'),row['case'],row['cached'],row['stream'])
        groups[key][row['variant']].append(row)
        outputs=row['outputs']; total+=len(outputs)
        assert sum(o['meta_info']['completion_tokens'] for o in outputs)==row['completion_tokens']
        assert abs(row['tokens_s']-row['completion_tokens']/(row['wall_ms']/1000)) < 1e-6
        for o in outputs:
            assert len(o['output_ids'])==o['meta_info']['completion_tokens']
            assert (o['meta_info']['cached_tokens']>0) if row['cached'] else (o['meta_info']['cached_tokens']==0)
    assert total==expected_responses
    stats=[]
    for (mode,case,cached,stream),variants in groups.items():
        assert variants.keys()=={'baseline','pr'}
        first=variants['baseline'][0]['outputs']
        for variant, items in variants.items():
            for item in items:
                assert len(first)==len(item['outputs'])
                for a,b in zip(first,item['outputs']):
                    assert a['output_ids']==b['output_ids'] and a['text']==b['text'],(path,case,variant)
                    assert a['meta_info']['finish_reason']==b['meta_info']['finish_reason']
        med={v:statistics.median(x['tokens_s'] for x in items) for v,items in variants.items()}
        result={'mode':mode,'case':case,'cached':cached,'stream':stream,'samples_per_variant':{v:len(items) for v,items in variants.items()},'baseline_tokens_s':med['baseline'],'pr_tokens_s':med['pr'],'change_pct':100*(med['pr']/med['baseline']-1)}
        if stream:
            result['ttft_ms']={v:statistics.median(x['ttft_ms'] for x in items) for v,items in variants.items()}
            result['ttft_change_pct']=100*(result['ttft_ms']['pr']/result['ttft_ms']['baseline']-1)
        stats.append(result)
    return {'groups':len(rows),'responses':total,'exact_output_ids_and_text':True,'metrics':stats}
summary={'baseline':BASE,'head':HEAD,'final':summarize('final/e2e-raw.json',140,7080),'high':summarize('high/e2e-raw.json',40,20480),'diagnostic':summarize('control/raw.json',32,2064),'earlier_pre_guard':summarize('e2e-raw.json',280,14160)}
for folder in ['final','high']:
    rows=read(folder+'/natural-raw.json');assert len(rows)==16
    for i in range(4):
        subset=rows[i::4]; ref=subset[0]['response']
        for x in subset:
            o=x['response']; assert o['output_ids']==ref['output_ids'] and o['text']==ref['text']
            assert o['meta_info']['finish_reason']['type']=='stop'
            assert o['meta_info']['completion_tokens']>=x['request']['sampling_params']['min_new_tokens']
    summary[folder]['natural_eos_responses']=len(rows)
rows=read('micro-raw.json'); assert len(rows)==960
micro=collections.defaultdict(lambda:collections.defaultdict(list))
for x in rows:micro[(x['batch'],x['vocab'],x['stops'])][x['variant']].append(x['gpu_us'])
assert len(micro)==24
summary['micro']=[]
for (b,v,k), variants in micro.items():
    assert {name:len(samples) for name,samples in variants.items()}=={'baseline':20,'pr':20}
    med={name:statistics.median(samples) for name,samples in variants.items()}
    summary['micro'].append({'batch':b,'vocab':v,'custom_stops':k,'baseline_us':med['baseline'],'pr_us':med['pr'],'speedup':med['baseline']/med['pr']})
# Confirm bootstrap embeds the scripts now offered for reproduction.
node=next(x for x in ast.parse((ROOT/'send-prepare.py').read_text()).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in x.targets))
for name,data in ast.literal_eval(node.value).items(): assert base64.b64decode(data)==(ROOT/name).read_bytes(),name
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('VERIFIED: source hashes; archive hash; 105 tests / 23 subtests; 24 micro shapes; 65 GPU checks; 27,592 final-head model responses; 2,064 diagnostic responses; 14,160 earlier responses; exact token/text/finish parity; bootstrap consistency.')
for name in ['final','high','diagnostic']:
    for x in summary[name]['metrics']:print(name,x['case'],'cached='+str(x['cached']),'stream='+str(x['stream']),f"{x['change_pct']:+.2f}%")
