import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np
import math
import pdb

from bagnet.viz.plot import (
    get_dataset, plot_everything, plot_cost_from_dict, plot_cost, print_best_design
)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    args = parser.parse_args()

    data = []
    for log_dir in args.logdir:
        data.append(get_dataset(log_dir, False))

    # plot_everything(data, args.legend)
    # plot_cost(data, args.legend)
    plot_cost_from_dict(data)
    plt.savefig('cost.png')
    # print_best_design(data, args.legend)
    # plot_cost2(data, 'n_query', args.legend)


