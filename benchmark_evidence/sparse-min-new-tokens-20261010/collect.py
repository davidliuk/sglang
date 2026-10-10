import pathlib,json,subprocess,hashlib,sys,platform
r=pathlib.Path('/content/min-tokens')
(r/'hardware.json').write_text(json.dumps({'platform':platform.platform(),'python':subprocess.check_output(['/content/sglang-env/bin/python','--version'],text=True).strip(),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)},indent=2))
next_bench=pathlib.Path('/content/min-tokens-next/bench.py')
if next_bench.exists():(r/'bench-final.py').write_bytes(next_bench.read_bytes())
archive=pathlib.Path('/content/sglang-min-tokens-results.tar.gz')
files=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.py','.json','.jsonl','.log','.txt','.pid')]
import tarfile
with tarfile.open(archive,'w:gz') as out:
    for p in files:out.add(p,arcname='results/'+p.name)
    for name in ['prototype','compiled-prototype','final','high','high-cache-attempt','control']:
        if (r/name).exists():out.add(r/name,arcname='results/'+name)
print('ARCHIVE',str(archive),'BYTES',archive.stat().st_size,'SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())
