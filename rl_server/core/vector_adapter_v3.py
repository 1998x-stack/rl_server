"""Gymnasium VectorEnv V3 episode-safe step extraction.

Consumes arrays emitted by actual Gymnasium vector environments; dependency
remains optional and no Gym runtime is imported in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4
import numpy as np

@dataclass(frozen=True)
class TransitionV3:
    observation: np.ndarray
    action: object
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    env_id: int
    episode_id: int
    sequence_id: int
    policy_version: int
    run_id: str
    message_id: str

class VectorTransitionAssembler:
    def __init__(self, initial_observations, *, run_id: str, mode: str, policy_version=0):
        observations=np.asarray(initial_observations)
        if observations.ndim<2 or observations.shape[0]<1:
            raise ValueError('Initial observations must have [N,...] shape')
        if mode not in ('same_step','next_step','disabled'):
            raise ValueError('Unsupported autoreset mode')
        if not isinstance(run_id,str) or not run_id or policy_version<0:
            raise ValueError('Invalid run identity')
        self.current=observations.copy()
        self.num_envs=observations.shape[0]
        self.mode=mode
        self.run_id=run_id
        self.policy_version=policy_version
        self.episodes=np.zeros(self.num_envs,dtype=np.int64)
        self.steps=np.zeros(self.num_envs,dtype=np.int64)
        self.reset_only=np.zeros(self.num_envs,dtype=bool)
        self.needs_reset=np.zeros(self.num_envs,dtype=bool)

    @staticmethod
    def _final_obs(info,i):
        if isinstance(info,dict):
            for key in ('final_obs','final_observation'):
                if key not in info:
                    continue
                mask=info.get('_'+key)
                if mask is not None and not bool(mask[i]):
                    continue
                observation=info[key][i]
                if observation is not None:
                    return np.asarray(observation)
        elif isinstance(info,(list,tuple)):
            for key in ('final_obs','final_observation'):
                if key in info[i] and info[i][key] is not None:
                    return np.asarray(info[i][key])
        raise ValueError('SAME_STEP missing final observation; refusing reset observation as target')

    def manual_reset(self,i,new_observation):
        if self.mode!='disabled' or not self.needs_reset[i]:
            raise ValueError('Unexpected manual reset')
        obs=np.asarray(new_observation)
        if obs.shape!=self.current[i].shape:
            raise ValueError('Invalid reset observation')
        self.current[i]=obs
        self.episodes[i]+=1
        self.steps[i]=0
        self.needs_reset[i]=False

    def step(self,actions,observations,rewards,terminated,truncated,info=None):
        obs=np.asarray(observations)
        actions=np.asarray(actions)
        rewards=np.asarray(rewards)
        terminal=np.asarray(terminated,dtype=bool)
        cutoff=np.asarray(truncated,dtype=bool)
        n=self.num_envs
        if obs.shape!=self.current.shape:
            raise ValueError('Observation shape changed')
        if any(x.ndim<1 or x.shape[0]!=n for x in (actions,rewards,terminal,cutoff)):
            raise ValueError('Mismatched batch dimensions')
        if rewards.ndim!=1 or terminal.ndim!=1 or cutoff.ndim!=1:
            raise ValueError('Reward and boundary flags must have [N] shape')
        if not np.isfinite(obs).all() or not np.isfinite(rewards).all():
            raise ValueError('Nonfinite observations or rewards')
        transitions=[]
        for i in range(n):
            if self.needs_reset[i]:
                raise ValueError(f'Environment {i} requires manual reset')
            if self.mode=='next_step' and self.reset_only[i]:
                self.current[i]=obs[i].copy()
                self.reset_only[i]=False
                self.episodes[i]+=1
                self.steps[i]=0
                continue
            finished=bool(terminal[i] or cutoff[i])
            next_true=obs[i]
            if self.mode=='same_step' and finished:
                next_true=self._final_obs(info,i)
                if next_true.shape!=self.current[i].shape:
                    raise ValueError('Invalid final observation shape')
            transitions.append(TransitionV3(
                observation=self.current[i].copy(),
                action=np.asarray(actions[i]).copy(),
                reward=float(rewards[i]),
                next_observation=next_true.copy(),
                terminated=bool(terminal[i]),
                truncated=bool(cutoff[i]),
                env_id=i,
                episode_id=int(self.episodes[i]),
                sequence_id=int(self.steps[i]),
                policy_version=self.policy_version,
                run_id=self.run_id,
                message_id=uuid4().hex,
            ))
            self.current[i]=obs[i].copy()
            self.steps[i]+=1
            if finished:
                if self.mode=='same_step':
                    self.episodes[i]+=1
                    self.steps[i]=0
                elif self.mode=='next_step':
                    self.reset_only[i]=True
                else:
                    self.needs_reset[i]=True
        return transitions
