"""Explicit v3 dry-run entrypoint; intentionally never imports torch/Gym/Redis."""
from __future__ import annotations
import argparse
import json
from rl_server.config.run_spec_v3 import RunSpec
from rl_server.runtime.v3.capabilities import validate_capabilities

def main(argv=None):
    parser=argparse.ArgumentParser(description='V3 opt-in RunSpec validator')
    parser.add_argument('--spec',required=True)
    parser.add_argument('--dry-run',action='store_true',required=True)
    args=parser.parse_args(argv)
    spec=RunSpec.from_file(args.spec)
    caps=validate_capabilities(spec)
    print(json.dumps({'engine':'v3','run_id':spec.run_id,'run_spec_sha256':spec.digest,
                      'algorithm':caps.algorithm_id,'mode':caps.training_mode,
                      'num_envs':spec.data['environment']['num_envs'],
                      'status':'validated_no_process_started'},sort_keys=True))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
