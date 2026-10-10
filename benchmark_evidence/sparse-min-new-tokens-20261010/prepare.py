import pathlib, subprocess, os, json, time, sys, hashlib
root=pathlib.Path('/content/min-tokens'); root.mkdir(exist_ok=True)
timings={}; started=time.monotonic()
def run(args,**kw): subprocess.run(args,check=True,**kw)
def stage(name,fn):
    t=time.monotonic(); print('START',name,flush=True); fn(); timings[name]=time.monotonic()-t
    (root/'preparation.json').write_text(json.dumps(timings,indent=2)); print('DONE',name,timings[name],flush=True)
try:
    stage('uv',lambda:run([sys.executable,'-m','pip','install','uv']))
    stage('venv',lambda:run(['uv','venv','--python','3.12','/content/sglang-env']))
    python='/content/sglang-env/bin/python'
    stage('torch',lambda:run(['uv','pip','install','--python',python,'torch==2.14.1','torchvision==0.29.1','--index-url','https://download.pytorch.org/whl/cu130']))
    stage('dependencies',lambda:run(['uv','pip','install','--python',python,'-r',str(root/'requirements.txt'),'pytest']))
    def source():
        run(['git','init','/content/sglang-src']);run(['git','-C','/content/sglang-src','remote','add','origin','https://github.com/sgl-project/sglang.git'])
        run(['git','-C','/content/sglang-src','fetch','--depth','1','origin','234ccba7f625e12639e621044496bcc6197c3758']);run(['git','-C','/content/sglang-src','checkout','--detach','FETCH_HEAD'])
        path=pathlib.Path('/content/sglang-src/python/sglang/srt/sampling/penaltylib/min_new_tokens.py')
        assert path.read_bytes()==(root/'baseline_min_new_tokens.py').read_bytes()
    stage('source',source)
    stage('model',lambda:run([python,'-c',"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',local_dir='/content/min-tokens/model',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])"]))
    env={**os.environ,'PATH':'/content/sglang-env/bin:'+os.environ.get('PATH',''),'PYTHONPATH':'/content/sglang-src/python'}
    stage('import',lambda:run([python,'-c','import torch; from sglang.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer; print(torch.__version__,torch.cuda.get_device_name())'],env=env))
    (root/'freeze.txt').write_text(subprocess.check_output(['uv','pip','freeze','--python',python],text=True))
    timings['total']=time.monotonic()-started;(root/'READY.json').write_text(json.dumps(timings,indent=2))
except BaseException as e:
    (root/'FAILED.json').write_text(json.dumps({'error':repr(e)}));raise
