import matplotlib.pyplot as plt
from utils.pdb import register_pdb_hook
register_pdb_hook()

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
        data.append(get_dataset(log_dir, time=True, old=False))

    # plot_everything(data, args.legend)
    plot_cost([x['db'] for x in data], args.legend)
    plt.savefig('cost_vs_niter.png')
    # plot_cost_from_dict(data)
    # plt.savefig('cost_vs_n_query.png')
    # print_best_design(data, args.legend)
    # plot_cost2(data, 'n_query', args.legend)


