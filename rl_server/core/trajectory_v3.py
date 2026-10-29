"""Versioned V3 PPO rollout [T,N] and boundary-safe GAE."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RolloutBatch:
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    valid_mask: np.ndarray
    policy_versions: np.ndarray

    def __post_init__(self):
        shape=np.asarray(self.rewards).shape
        if len(shape)!=2 or any(s<=0 for s in shape):
            raise ValueError('Rollout must have [time,env] dimensions')
        for name in ('values','next_values','terminated','truncated','valid_mask','policy_versions'):
            if np.asarray(getattr(self,name)).shape!=shape:
                raise ValueError(f'{name} must have shape {shape}')
        for name in ('rewards','values','next_values'):
            if not np.isfinite(np.asarray(getattr(self,name))).all():
                raise ValueError(f'Non-finite {name}')
        valid=np.asarray(self.valid_mask,dtype=bool)
        versions=np.asarray(self.policy_versions)
        if (versions[valid]<0).any() or np.unique(versions[valid]).size>1:
            raise ValueError('PPO rollout has mixed or invalid policy versions')

def compute_gae(rollout: RolloutBatch, gamma: float=.99, lam: float=.95):
    """Truncation bootstraps at true final obs but severs cross-episode GAE.

    Termination stops the value bootstrap.  Reset-only (invalid) NEXT_STEP
    slots do not count as an environment transition or carry advantages.
    """
    if not 0<=gamma<=1 or not 0<=lam<=1:
        raise ValueError('Invalid gamma/lambda')
    r=np.asarray(rollout.rewards,dtype=np.float64)
    v=np.asarray(rollout.values,dtype=np.float64)
    nxt=np.asarray(rollout.next_values,dtype=np.float64)
    term=np.asarray(rollout.terminated,dtype=bool)
    trunc=np.asarray(rollout.truncated,dtype=bool)
    valid=np.asarray(rollout.valid_mask,dtype=bool)
    advantages=np.zeros_like(r)
    following=np.zeros(r.shape[1],dtype=np.float64)
    for t in reversed(range(r.shape[0])):
        delta=r[t]+gamma*(~term[t])*nxt[t]-v[t]
        can_continue=(~term[t])&(~trunc[t])&valid[t]
        following=np.where(valid[t],delta+gamma*lam*can_continue*following,0.)
        advantages[t]=following
    returns=np.where(valid,advantages+v,0.)
    return advantages.astype(np.float32),returns.astype(np.float32)
