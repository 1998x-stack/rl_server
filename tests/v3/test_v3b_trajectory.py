"""Three Gym autoreset modes and per-env PPO GAE boundary goldens."""
import numpy as np
import pytest
from rl_server.core.trajectory_v3 import RolloutBatch,compute_gae
from rl_server.core.vector_adapter_v3 import VectorTransitionAssembler

def test_ppo_gae_isolated_lanes():
    r=RolloutBatch(np.ones((3,2)),np.zeros((3,2)),np.ones((3,2)),
      np.array([[False,False],[True,False],[False,False]]),
      np.zeros((3,2),bool),np.ones((3,2),bool),np.zeros((3,2),int))
    adv,_=compute_gae(r,gamma=1,lam=1)
    assert adv[0,0]==3
    assert adv[0,1]==6

def test_terminal_stops_bootstrap():
    r=RolloutBatch(np.array([[2.]]),np.array([[0.]]),np.array([[99.]]),
      np.array([[True]]),np.array([[False]]),np.array([[True]]),np.array([[1]]))
    advantage,returns=compute_gae(r)
    assert advantage[0,0]==returns[0,0]==2

def test_truncation_bootstraps_final_value_without_next_episode_leak():
    r=RolloutBatch(np.array([[1.],[100.]]),np.zeros((2,1)),
      np.array([[7.],[1.]]),np.zeros((2,1),bool),
      np.array([[True],[False]]),np.ones((2,1),bool),np.zeros((2,1),int))
    advantage,_=compute_gae(r,gamma=.9,lam=1)
    assert advantage[0,0]==pytest.approx(7.3,abs=1e-6)

def test_mixed_policy_version_rejected():
    with pytest.raises(ValueError,match='mixed'):
      RolloutBatch(np.ones((2,1)),np.zeros((2,1)),np.zeros((2,1)),
        np.zeros((2,1),bool),np.zeros((2,1),bool),np.ones((2,1),bool),
        np.array([[1],[2]]))

def test_same_step_final_obs_and_reset_obs_separated():
    adapter=VectorTransitionAssembler(np.array([[1.]]),run_id='r',mode='same_step')
    item=adapter.step([0],np.array([[0.]]),[1],[False],[True],
       {'final_obs':np.array([[42.]]),'_final_obs':np.array([True])})[0]
    assert item.next_observation[0]==42 and adapter.current[0,0]==0
    with pytest.raises(ValueError,match='final observation'):
      adapter.step([0],np.array([[0.]]),[1],[True],[False],{})

def test_next_step_reset_only_frame_is_not_training_data():
    adapter=VectorTransitionAssembler(np.array([[0.],[10.]]),run_id='r',mode='next_step')
    first=adapter.step([1,0],np.array([[5.],[11.]]),[1,1],[True,False],[False,False])
    assert len(first)==2
    second=adapter.step([0,0],np.array([[0.],[12.]]),[0,1],[False,False],[False,False])
    assert len(second)==1 and second[0].env_id==1
    third=adapter.step([0,0],np.array([[1.],[13.]]),[1,1],[False,False],[False,False])
    assert third[0].episode_id==1 and third[0].observation[0]==0

def test_disabled_mode_must_reset_explicitly():
    adapter=VectorTransitionAssembler(np.array([[1.]]),run_id='r',mode='disabled')
    adapter.step([0],np.array([[2.]]),[1],[True],[False])
    with pytest.raises(ValueError,match='manual reset'):
      adapter.step([0],np.array([[3.]]),[1],[False],[False])
    adapter.manual_reset(0,np.array([9.]))
    assert adapter.step([0],np.array([[8.]]),[1],[False],[False])[0].observation[0]==9

def test_invalid_rollout_shape_rejected():
    with pytest.raises(ValueError):
      RolloutBatch(np.ones((3,2)),np.zeros((3,1)),np.zeros((3,2)),
        np.zeros((3,2),bool),np.zeros((3,2),bool),np.ones((3,2),bool),np.zeros((3,2),int))
