"""Run lifecycle with fail-closed transitions and a local audit trail."""
from dataclasses import dataclass,field
from enum import Enum
from time import time

class RunState(str,Enum):
    CREATED='CREATED'; VALIDATING='VALIDATING'; QUEUED='QUEUED'
    STARTING='STARTING'; RUNNING='RUNNING'; PAUSING='PAUSING'
    PAUSED='PAUSED'; RESUMING='RESUMING'; RECOVERING='RECOVERING'
    COMPLETING='COMPLETING'; COMPLETED='COMPLETED'
    CANCELLING='CANCELLING'; CANCELLED='CANCELLED'; FAILED='FAILED'

_ALLOWED={
 RunState.CREATED:{RunState.VALIDATING,RunState.CANCELLING},
 RunState.VALIDATING:{RunState.QUEUED,RunState.FAILED,RunState.CANCELLING},
 RunState.QUEUED:{RunState.STARTING,RunState.CANCELLING},
 RunState.STARTING:{RunState.RUNNING,RunState.FAILED,RunState.CANCELLING},
 RunState.RUNNING:{RunState.PAUSING,RunState.RECOVERING,RunState.COMPLETING,RunState.FAILED,RunState.CANCELLING},
 RunState.PAUSING:{RunState.PAUSED,RunState.FAILED,RunState.CANCELLING},
 RunState.PAUSED:{RunState.RESUMING,RunState.CANCELLING},
 RunState.RESUMING:{RunState.RUNNING,RunState.FAILED,RunState.CANCELLING},
 RunState.RECOVERING:{RunState.RUNNING,RunState.FAILED,RunState.CANCELLING},
 RunState.COMPLETING:{RunState.COMPLETED,RunState.FAILED},
 RunState.CANCELLING:{RunState.CANCELLED,RunState.FAILED},
}

@dataclass
class RunLifecycle:
    run_id: str
    state: RunState = RunState.CREATED
    audit: list = field(default_factory=list)

    def transition(self, next_state: RunState, reason: str, *,
                   committed_checkpoint=False, final_evaluation=False):
        if next_state not in _ALLOWED.get(self.state,set()):
            raise ValueError(f'Illegal transition {self.state} -> {next_state}')
        if next_state == RunState.PAUSED and not committed_checkpoint:
            raise ValueError('PAUSED requires committed checkpoint')
        if next_state == RunState.COMPLETED and not final_evaluation:
            raise ValueError('COMPLETED requires final evaluation')
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('Nonempty transition reason required')
        self.audit.append({'run_id':self.run_id,'from':self.state.value,'to':next_state.value,
                           'reason':reason,'timestamp':time()})
        self.state=next_state
