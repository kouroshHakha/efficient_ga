"""Very hacky module for plotting"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np
import math
import pdb

def get_dataset(log_dir, time=False, old=True):
    if time:
        fname = '/db_time.pkl'
    else:
        fname = '/db.pkl'
    if old:
        fname = ''

    with open(log_dir + fname, 'rb') as f:
        return pickle.load(f)
def plot_everything(data, legends):
    sns.set_style("darkgrid")
    # golden_vals = [1.15, 0, -1.15]
    # golden_vals = [300, 10e6, 60, 90e-9, 50, 50, 1e-3 , 0.2e-3]
    max_itr = min([len(x) for x in data])
    design_sample = data[0][0][0]
    # n_sub_plots is the number of spec keywords and cost
    n_sub_plots = len(list(design_sample.specs.keys())) + 1
    n_rows = int(math.sqrt(n_sub_plots))
    n_cols = int(math.ceil(n_sub_plots / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_rows, 3*n_cols))
    for exp, legend in zip(data, legends):
        n_evals = 0
        for ax, kwrd in zip(axs.ravel(), design_sample.specs.keys()): #, golden_vals): , val
            db = []
            avg_to_plot = []
            for itr, offsprings in enumerate(exp):
                db += offsprings
                db_sorted = sorted(db, key=lambda x: x['cost'])
                top_10 = db_sorted[:10]
                spec_value = [dsn.specs[kwrd] for dsn in top_10]
                avg_to_plot.append(np.mean(spec_value))

            ax.set_title(kwrd)
            ax.plot(avg_to_plot)
            # sns.lineplot(y=avg_to_plot, legend=False, ax=ax) # min version 0.9.0
            # ax.plot(np.repeat(val, len(avg_to_plot)), 'r--')
            # ax.legend(loc='lower right')


def plot_cost(data, legends=None, ax=None):
    if ax is None:
        plt.close()
        ax = plt.gca()
    max_itr = min([len(x) for x in data])
    avg_plt_objects = []
    if legends:
        combo = zip(data, legends)
    else:
        combo = data
    for item in combo:
        if legends:
            exp, legend = item
        else:
            exp = item
            legend = 'avg'
        db = []
        avg_to_plot = []
        min_to_plot = []
        n_evals = 0
        for i, offsprings in enumerate(exp):
            # print(offsprings)
            n_evals += len(offsprings)
            if any([x['cost'] == 0 for x in offsprings]):
                print("solution found on iter: {} neval: {}".format(i, n_evals))
                print("solution found on iter: {} neval: {}".format(i, n_evals))
            db += offsprings
            db_sorted = sorted(db, key=lambda x: x['cost'])
            avg_cost = np.mean([x['cost'] for x in db_sorted[:20]])
            min_cost = db_sorted[0]['cost']
            avg_to_plot.append(avg_cost)

        avg_plt_objects.append(ax.plot(avg_to_plot, label=legend))
        # min_plt_objects.append(ax.plot(min_to_plot, '--', label='min_'+legend))

    if legends:
        ax.legend()
    ax.set_title('cost')
    # ax.set_ylabel('cost')
    # ax.set_xlabel('iteration')

def print_best_design(data, legends):
    for exp, legend in zip(data, legends):
        db = []
        for offsprings in exp:
            db += offsprings

        db_sorted = sorted(db, key=lambda x: x['cost'])
        solutions = [x for x in db_sorted if x['cost'] == 0]
        for sol in solutions:
            print("{} -> {}".format(sol, sol.specs))
        # best_sol = min(solutions, key=lambda x: x.specs['ibias_cur'])
        # print("{}: {} -> {}".format(legend, best_sol, best_sol.specs))


def plot_cost_from_dict(data, x_axis='n_nn_query', legends=None, ax=None):
    if ax is None:
        plt.close()
        ax = plt.gca()
    if legends:
        combo = zip(data, legends)
    else:
        combo = data
    for item in combo:
        if legends:
            exp, legend = item
        else:
            exp = item
            legend = 'avg'
    avg_plt_objects = []
    min_plt_objects = []
    for item in combo:
        if legends:
            exp, legend = item
        else:
            exp = item
            legend = 'avg'

        db = []
        avg_to_plot = []
        min_to_plot = []
        n_evals = 0
        for i, offsprings in enumerate(exp['db']):
            n_evals += len(offsprings)
            db += offsprings
            db_sorted = sorted(db, key=lambda x: x['cost'])
            avg_cost = np.mean([x['cost'] for x in db_sorted[:20]])
            min_cost = db_sorted[0]['cost']
            avg_to_plot.append(avg_cost)
            min_to_plot.append(min_cost)

        avg_plt_objects.append(ax.plot(exp[x_axis], avg_to_plot, label='avg_'+legend))
        # min_plt_objects.append(ax.plot(exp[x_axis], min_to_plot, '--', label='min_'+legend))

    if legends:
        ax.legend()
    ax.set_title('cost')
    # ax.set_ylabel('cost')
    ax.set_xlabel(x_axis)

