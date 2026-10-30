"""PR-A goldens: immutable identity, mode restrictions, lifecycle, legacy isolation."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from rl_server.config.run_spec_v3 import RunSpec
from rl_server.runtime.v3.capabilities import validate_capabilities
from rl_server.runtime.v3.lifecycle import RunLifecycle,RunState

EXAMPLE={
 'schema_version':1,'run_id':'run-demo','engine':'v3',
 'algorithm':{'id':'dqn','mode':'off_policy','config':{'gamma':0.99}},
 'environment':{'id':'CartPole-v1','version':'test','num_envs':2,'autoreset_mode':'next_step'},
 'seed':7,'budget':{'max_env_steps':1000,'evaluation_interval':100},
 'reliability':{'mode':'local','commit_window_updates':32},
 'build':{'code_sha':'a'*40,'dependency_lock_digest':'sha256:example-placeholder'},
}

def example():
    return json.loads(json.dumps(EXAMPLE))

def test_canonical_digest_order_independent():
    obj=example()
    assert RunSpec.from_dict(obj).digest==RunSpec.from_dict(dict(reversed(list(obj.items())))).digest

@pytest.mark.parametrize('field',['seed','environment','algorithm'])
def test_semantic_changes_affect_digest(field):
    obj=example()
    if field=='seed': obj['seed']=8
    elif field=='environment': obj['environment']['version']='different'
    else: obj['algorithm']['config']['gamma']=.98
    assert RunSpec.from_dict(obj).digest != RunSpec.from_dict(EXAMPLE).digest

@pytest.mark.parametrize('algorithm,mode',[('ppo_discrete','off_policy'),('sac','on_policy')])
def test_wrong_mode_rejected(algorithm,mode):
    obj=example()
    obj['algorithm']['id']=algorithm;obj['algorithm']['mode']=mode
    with pytest.raises(ValueError):RunSpec.from_dict(obj)

@pytest.mark.parametrize('key,bad',[('run_id','bad/id'),('seed',True),('extra','bad')])
def test_invalid_fields_rejected(key,bad):
    obj=example();obj[key]=bad
    with pytest.raises(ValueError):RunSpec.from_dict(obj)

def test_immutable_nested_configuration_and_resume():
    obj=example();spec=RunSpec.from_dict(obj);obj['algorithm']['config']['gamma']=22
    assert spec.to_dict()['algorithm']['config']['gamma']==.99
    with pytest.raises(TypeError):spec.data['algorithm']['config']['gamma']=99
    spec.assert_resume('run-demo',spec.digest)
    with pytest.raises(ValueError):spec.assert_resume('run-demo','bad')

def test_fail_closed_capabilities():
    spec=RunSpec.from_dict(EXAMPLE)
    validate_capabilities(spec,action_space='discrete')
    with pytest.raises(ValueError):validate_capabilities(spec,multi_agent=True)
    with pytest.raises(ValueError):validate_capabilities(spec,action_space='continuous')
    with pytest.raises(ValueError):validate_capabilities(spec,action_mask=True)

def test_lifecycle_requires_checkpoint_and_final_eval():
    run=RunLifecycle('run-demo')
    for state in (RunState.VALIDATING,RunState.QUEUED,RunState.STARTING,RunState.RUNNING):
        run.transition(state,'test')
    run.transition(RunState.PAUSING,'checkpoint')
    with pytest.raises(ValueError):run.transition(RunState.PAUSED,'bad')
    run.transition(RunState.PAUSED,'saved',committed_checkpoint=True)
    assert run.state==RunState.PAUSED

def test_dry_run_does_not_import_torch_gym_redis(tmp_path):
    specfile=tmp_path/'run.json';specfile.write_text(json.dumps(EXAMPLE))
    program='''import sys
from rl_server.entrypoints.v3_train import main
main(['--spec',sys.argv[1],'--dry-run'])
assert 'torch' not in sys.modules
assert 'gymnasium' not in sys.modules
assert 'redis' not in sys.modules
'''
    proc=subprocess.run([sys.executable,'-c',program,str(specfile)],capture_output=True,text=True,check=True)
    assert json.loads(proc.stdout)['status']=='validated_no_process_started'
