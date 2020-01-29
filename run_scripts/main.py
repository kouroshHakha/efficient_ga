import argparse

from utils.importlib import import_class
from utils.file import read_yaml
from utils.pdb import register_pdb_hook

register_pdb_hook()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('fname', type=str, help='the main yaml file that sets the settings')
    parser.add_argument('--seed', '-s', type=int, default=10,
                        help='the main yaml file that sets the settings')

    args = parser.parse_args()

    setting = read_yaml(args.fname)

    agent_cls = import_class(setting['agent_cls'])
    agent = agent_cls(args.fname)

    agent.main()
