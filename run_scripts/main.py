from typing import Type

import argparse

from utils.importlib import import_class
from utils.file import read_yaml
from utils.pdb import register_pdb_hook

from bagnet.agent.base import Agent

from bagnet.viz.plot import plot_cost, plot_cost_from_dict, get_dataset
import matplotlib.pyplot as plt

register_pdb_hook()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('fname', type=str, help='the main yaml file that sets the settings')
    parser.add_argument('--seed', '-s', type=int, default=10,
                        help='the main yaml file that sets the settings')
    parser.add_argument('--plot', '-p', action='store_true', default=False,
                        help='True to plot the dbs')

    args = parser.parse_args()

    setting = read_yaml(args.fname)

    agent_cls: Type[Agent] = import_class(setting['agent_cls'])
    agent: Agent = agent_cls(args.fname)

    agent.main()
    # Hacky plotting
    plot_cost([agent.data_set_list])
    plt.savefig(agent.output_path/'cost_vs_iter.png', dpi=200)
    data = get_dataset(str(agent.output_path), time=True, old=False)
    plot_cost_from_dict([data],x_axis='n_nn_query') #bagnet option: change 'n_query' to 'n_nn_query'
    plt.savefig(agent.output_path/'cost_vs_nquery.png', dpi=200)
