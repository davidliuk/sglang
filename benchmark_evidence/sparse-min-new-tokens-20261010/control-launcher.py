"""Diagnostic only: fixed model and metadata; choose unchanged apply implementations."""
import importlib.util,pathlib,runpy,json
from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
import sglang.srt.sampling.penaltylib as penaltylib
ROOT=pathlib.Path('/content/min-tokens')
def load(name):
    spec=importlib.util.spec_from_file_location('diagnostic_'+name,ROOT/f'{name}_min_new_tokens.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.BatchedMinNewTokensPenalizer
Baseline=load('baseline');PR=load('pr')
class ControlledMinNewTokens(_BatchedPenalizer):
    def _is_required(self):return any(r.sampling_params.min_new_tokens>0 for r in self.orchestrator.reqs())
    def _prepare(self):
        variants={(r.sampling_params.custom_params or {}).get('bench_variant','baseline') for r in self.orchestrator.reqs()}
        assert len(variants)==1,variants
        self.variant=variants.pop();assert self.variant in ['baseline','pr']
        self.base=Baseline(self.orchestrator);self.pr=PR(self.orchestrator)
        self.base.prepare();self.pr.prepare()
        with open(ROOT/'control/audit.jsonl','a') as out:out.write(json.dumps({'variant':self.variant,'batch':len(self.orchestrator.reqs())})+'\n')
    def _cumulate_output_tokens(self,output_ids):
        self.base.cumulate_output_tokens(output_ids);self.pr.cumulate_output_tokens(output_ids)
    def _apply(self,logits):
        (self.base if self.variant=='baseline' else self.pr).apply(logits)
    def _filter(self,keep_indices):
        self.base.filter(keep_indices);self.pr.filter(keep_indices)
    def _merge(self,their):
        assert self.variant==their.variant
        self.base.merge(their.base);self.pr.merge(their.pr)
    def _teardown(self):
        for name in ['base','pr']:
            impl=getattr(self,name,None)
            if impl is not None:impl.teardown();delattr(self,name)
penaltylib.BatchedMinNewTokensPenalizer=ControlledMinNewTokens
if __name__=='__main__':runpy.run_module('sglang.launch_server',run_name='__main__')
