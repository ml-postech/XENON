import json
import os
import argparse
import yaml
import fcntl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration_uuid", type=str, required=True)
    args = parser.parse_args()

    # Read evaluate.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    conf_path = os.path.join(current_dir, '..', 'conf', 'evaluate.yaml')
    with open(conf_path) as f:
        conf = yaml.safe_load(f)

    version = conf['version']
    recipe_base = conf['memory']['recipe']['base']
    prefix = conf['prefix']
    path_str = conf['memory']['path'].replace('${prefix}', prefix).replace('${version}', version).replace('src/optimus1/', '')

    json_path = os.path.join(current_dir, '..', path_str, recipe_base, 'current_exploring_goals.json')

    with open(json_path, "r+") as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            data = json.load(fp)
        except json.JSONDecodeError:
            data = {}

        if args.exploration_uuid in data:
            del data[args.exploration_uuid]

        fp.seek(0)
        json.dump(data, fp)
        fp.truncate()
        fcntl.flock(fp, fcntl.LOCK_UN)
