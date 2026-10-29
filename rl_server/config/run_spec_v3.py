"""RunSpec v3: immutable canonical configuration and fail-closed validation.

No Gym, torch or Redis imports are allowed here.  Workers receive the canonical
RunSpec rather than interpolating their own environment variables.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Mapping

_ALGO_MODE = {'dqn':'off_policy','sac':'off_policy','td3':'off_policy',
              'ppo_discrete':'on_policy','ppo_continuous':'on_policy'}
_FIELDS = {
 'root':{'schema_version','run_id','engine','algorithm','environment','seed','budget','reliability','build'},
 'algorithm':{'id','mode','config'},
 'environment':{'id','version','num_envs','autoreset_mode','wrapper_digest'},
 'budget':{'max_env_steps','evaluation_interval'},
 'reliability':{'mode','commit_window_updates','message_limit_bytes'},
 'build':{'code_sha','dependency_lock_digest','image_digest'},
}
_REQUIRED = {'algorithm':{'id','mode','config'},
 'environment':{'id','version','num_envs','autoreset_mode'},
 'budget':{'max_env_steps','evaluation_interval'},
 'reliability':{'mode','commit_window_updates'},
 'build':{'code_sha','dependency_lock_digest'}}
_RUN_ID = re.compile(r'[A-Za-z0-9_.-]{1,64}\Z')
_GIT_SHA = re.compile(r'[0-9a-f]{40}\Z')

def _integer(x: Any, key: str, minimum: int, maximum=None):
    if type(x) is not int or x < minimum or (maximum is not None and x > maximum):
        raise ValueError(f'{key}: out-of-range integer')

def _section(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object')
    extra = set(value) - _FIELDS[name]
    missing = _REQUIRED[name] - set(value)
    if extra or missing:
        raise ValueError(f'{name}: unknown={sorted(extra)} missing={sorted(missing)}')
    return value

def _validate(data: dict):
    if set(data) != _FIELDS['root']:
        raise ValueError('Unknown or missing RunSpec root fields')
    if type(data['schema_version']) is not int or data['schema_version'] != 1 or data['engine'] != 'v3':
        raise ValueError('Unsupported RunSpec schema/engine')
    if not isinstance(data['run_id'], str) or not _RUN_ID.fullmatch(data['run_id']):
        raise ValueError('Invalid run_id')
    _integer(data['seed'], 'seed', 0)
    a, e = _section(data, 'algorithm'), _section(data, 'environment')
    if a['id'] not in _ALGO_MODE or a['mode'] != _ALGO_MODE.get(a['id']):
        raise ValueError('Invalid algorithm/mode pairing')
    if not isinstance(a['config'], dict):
        raise ValueError('algorithm.config must be a dict')
    if not isinstance(e['id'], str) or not e['id'].strip() or not isinstance(e['version'], str) or not e['version'].strip():
        raise ValueError('Environment id and version required')
    _integer(e['num_envs'], 'environment.num_envs', 1)
    if e['autoreset_mode'] not in ('next_step','same_step','disabled'):
        raise ValueError('Unsupported autoreset mode')
    b = _section(data, 'budget')
    _integer(b['max_env_steps'], 'budget.max_env_steps', 1)
    _integer(b['evaluation_interval'], 'budget.evaluation_interval', 1)
    r = _section(data, 'reliability')
    if r['mode'] not in ('local','durable'):
        raise ValueError('Unsupported reliability mode')
    _integer(r['commit_window_updates'], 'reliability.commit_window_updates', 1)
    if 'message_limit_bytes' in r:
        _integer(r['message_limit_bytes'], 'message_limit_bytes', 1024, 33554432)
    build = _section(data, 'build')
    if not isinstance(build['code_sha'], str) or not _GIT_SHA.fullmatch(build['code_sha']):
        raise ValueError('Invalid code_sha')
    if not isinstance(build['dependency_lock_digest'], str) or len(build['dependency_lock_digest']) < 16:
        raise ValueError('Invalid dependency lock digest')
    for section,key in ((e,'wrapper_digest'),(build,'image_digest')):
        if key in section and (not isinstance(section[key],str) or not section[key]):
            raise ValueError(f'Invalid {key}')

def _freeze(x):
    if isinstance(x, dict):
        return MappingProxyType({k:_freeze(v) for k,v in x.items()})
    if isinstance(x,list):
        return tuple(_freeze(i) for i in x)
    return x

def _thaw(x):
    if isinstance(x,Mapping):
        return {k:_thaw(v) for k,v in x.items()}
    if isinstance(x,tuple):
        return [_thaw(i) for i in x]
    return x

@dataclass(frozen=True)
class RunSpec:
    data: Mapping[str,Any]
    canonical_json: str
    digest: str

    @classmethod
    def from_dict(cls, raw):
        try:
            canonical = json.dumps(raw,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            copied = json.loads(canonical)
        except (TypeError,ValueError) as e:
            raise ValueError('RunSpec must contain finite JSON') from e
        if not isinstance(copied,dict):
            raise ValueError('RunSpec root must be a dict')
        _validate(copied)
        return cls(_freeze(copied),canonical,sha256(canonical.encode()).hexdigest())

    @classmethod
    def from_file(cls, filename):
        with open(filename,encoding='utf-8') as handle:
            return cls.from_dict(json.load(handle))

    @property
    def run_id(self):
        return self.data['run_id']

    def to_dict(self):
        return _thaw(self.data)

    def assert_resume(self, run_id: str, digest: str):
        if self.run_id != run_id or self.digest != digest:
            raise ValueError('RunSpec mismatch: refusing resume')
