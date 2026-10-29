"""Explicit algorithm / environment capability contract."""
from __future__ import annotations
from dataclasses import dataclass
from rl_server.config.run_spec_v3 import RunSpec

@dataclass(frozen=True)
class AlgorithmCapabilities:
    algorithm_id: str
    training_mode: str
    action_space: str
    supports_action_mask: bool
    supports_multi_agent: bool = False

REGISTRY = {
 'dqn': AlgorithmCapabilities('dqn','off_policy','discrete',False),
 'ppo_discrete': AlgorithmCapabilities('ppo_discrete','on_policy','discrete',False),
 'ppo_continuous': AlgorithmCapabilities('ppo_continuous','on_policy','continuous',False),
 'sac': AlgorithmCapabilities('sac','off_policy','continuous',False),
 'td3': AlgorithmCapabilities('td3','off_policy','continuous',False),
}

def validate_capabilities(spec: RunSpec, *, action_space=None, multi_agent=False, action_mask=False):
    a = spec.data['algorithm']
    caps = REGISTRY[a['id']]
    if a['mode'] != caps.training_mode:
        raise ValueError('Algorithm mode mismatch')
    if action_space is not None and action_space != caps.action_space:
        raise ValueError('Action space incompatibility')
    if multi_agent and not caps.supports_multi_agent:
        raise ValueError('Multi-agent environment unsupported')
    if action_mask and not caps.supports_action_mask:
        raise ValueError('Action mask unsupported')
    return caps
