"""
This module implements the BagNet algorithm
"""
from typing import cast, Type, List
import time
import os.path as osp

from utils.importlib import import_class
from .base import Agent
from ..model.model import Model


class BagNetAgent(Agent):

    def __init__(self, fname):
        Agent.__init__(self, fname)

        model_cls = cast(Type[Model], import_class(self.specs['model_cls']))
        self.model = model_cls(num_params_per_design=len(self.bb_env.params),
                               spec_kwrd_list=list(self.bb_env.spec_range.keys()),
                               logger=self._logger,
                               **self.specs['model_params'])

        # get the parameters in specs that are related to the model
        self.valid_frac = self.specs['valid_frac']
        self.max_n_retraining = self.specs['max_n_steps']
        # from db designs with indices less than this number are considered during training
        self.k = self.specs['k']
        # training settings
        self.ngrad_update_per_iter = self.specs.get('ngrad_update_per_iter', 1000)
        self.batch_size = self.specs['batch_size']
        self.display_step = self.specs['display_step']
        self.ckpt_step = self.specs['ckpt_step']
        self.max_iter = self.specs['max_iter']

        self.n_new_samples = self.specs['n_new_samples']
        # paper stuff variables
        self.n_sims = 0
        self.sim_time = 0
        self.n_queries = 0
        self.query_time = 0
        self.n_training = 0
        self.training_time = 0
        self.n_sims_list, self.sims_time_list = [], []
        self.n_nn_queries_list, self.nn_query_time_list = [], []
        self.n_training_list, self.training_time_list = [], []
        self.total_time_list = []

    def train(self):
        t_minus = time.time()
        ds = self.model.get_train_valid_ds(self.db, self.k, self.bb_env, self.valid_frac)
        self.model.train(ds, self.batch_size, self.ngrad_update_per_iter, self.ckpt_step, self.display_step)
        t_plus = time.time()
        self.n_training += 1
        self.training_time += (t_plus - t_minus)
        self.info("Training done %.2fSec" % (t_plus - t_minus))

    def run_per_iter(self):
        db_list = list(self.db)
        self.decision_box.update_heuristics(db_list, self.k)

        if self.decision_box.has_converged():
            return [], True

        parent1, parent2, ref_design = self.decision_box.get_parents_and_ref_design(db_list, self.k)

        offsprings = []
        n_iter = 0

        self.log(30*"-")
        self.info("Running model ... ")
        q_start = time.time()

        while_time = 0
        gen_time = 0
        check_time = 0
        q_time = 0
        decision_time = 0
        deletion_time = 0

        self.ea.prepare_for_generation(db_list, self.k)
        while_s = time.time()
        while len(offsprings) < self.n_new_samples and n_iter < self.max_iter:
            gen_s = time.time()
            new_designs = self.ea.get_next_generation_candidates(parent1, parent2)
            gen_e = time.time()
            gen_time += gen_e - gen_s

            for new_design in new_designs:
                check_s = time.time()
                if new_design in self.db or new_design in offsprings:
                    # if design is already in the design pool skip ...
                    self.debug(f"Design {new_design} already exists")
                    continue
                check_e = time.time()
                check_time += check_e - check_s

                n_iter += 1
                q_s = time.time()
                prediction = self.model.query(input1=new_design, input2=ref_design)
                q_e = time.time()
                q_time += q_e - q_s
                self.n_queries += 1

                decision_s = time.time()
                is_new_design_better = self.decision_box.accept_new_design(prediction)
                decision_e = time.time()
                decision_time += decision_e - decision_s

                deletion_s = time.time()
                if is_new_design_better:
                    offsprings.append(new_design)
                deletion_e = time.time()
                deletion_time += deletion_e - deletion_s

        while_e = time.time()
        while_time = while_e - while_s
        self.query_time += time.time() - q_start

        self.info("Avg_gen_time = {}".format(gen_time/n_iter))
        self.info("Avg_check_time = {}".format(check_time/n_iter))
        self.info("Avg_q_time = {}".format(q_time/n_iter))
        self.info("Avg_decision_time = {}".format(decision_time/n_iter))
        self.info("Avg_deletion_time = {}".format(deletion_time/n_iter))
        self.info("Avg_while_time = {}".format(while_time/n_iter))

        if len(offsprings) < self.n_new_samples:
            return offsprings, True

        self.log(30*"-")
        s = time.time()
        offsprings = cast(List, self.bb_env.evaluate(offsprings))
        e = time.time()
        self.n_sims += len(offsprings)
        self.sim_time += e - s

        self.info('Design evaluation time: {:.2f}'.format(e-s))
        list_to_be_removed = []
        for child in offsprings:
            if not child['valid']:
                list_to_be_removed.append(child)
                self.debug(f"Design {child} did not produce valid results")
            else:
                self.info(f"Added: {child} , cost = {child['cost']}")

        for design in list_to_be_removed:
            offsprings.remove(design)

        self.info(f"New designs tried: {n_iter}")
        self.info(f"New candidates size: {len(offsprings)} ")

        return offsprings, False

    def main(self):
        start = time.time()

        self.get_init_population()
        self.data_set_list.append(list(self.db))
        self.update_time_info(0)

        self.model.init()
        self.train()

        for i in range(self.max_n_retraining):
            self.info(f'********** Iter {i} **********')
            offsprings, is_converged = self.run_per_iter()

            if is_converged:
                break
            elif len(offsprings) == 0:
                continue

            self.db.extend(offsprings)
            self.data_set_list.append(offsprings)
            self._logger.store_db(self.data_set_list)
            self.update_time_info(time.time()-start)
            self.store_database_and_times()

            # adjust dataset size for training, if not desired, comment the agent.k_top= ... line
            db_sorted = sorted(self.db, key=lambda x: x['cost'])
            worst_offspring = max(offsprings, key=lambda x: x['cost'])
            self.info(f'k_top alternative: {db_sorted.index(worst_offspring)}')
            # self.k = max(self.n_init_samples, db_sorted.index(worst_offspring))

            self.train()
            self.info(f"Nqueries = {self.n_queries}")
            self.info(f"Query_time = {self.query_time}")
            self.info(f"Total_time = {time.time()-start}")
            self.info(f"sim_time = {self.sim_time}")
            self.info(f"n_training = {self.n_training}")
            self.info(f"training_time = {self.training_time}")
            self.info(f"total_time = {time.time() - start}")

        self._logger.store_db(self.data_set_list)
        self.store_database_and_times()

        sorted_db = sorted(self.db, key=lambda x: x['cost'])
        # paper stuff
        self.info(f"n_queries = {self.n_queries}")
        self.info(f"query_time = {self.query_time}")
        self.info(f"n_simulations = {self.n_sims}")
        self.info(f"sim_time = {self.sim_time}")
        self.info(f"n_training = {self.n_training}")
        self.info(f"training_time = {self.training_time}")
        self.info(f"total_time = {time.time() - start}")
        self.info(f"total_n_evals = {len(self.db)}")
        self.info(f"best_solution = {sorted_db[0]}")
        self.info(f"id = {sorted_db[0]['id']}")
        self.info(f"cost = {sorted_db[0]['cost']}")
        self.info(f"Performance \n{sorted_db[0].specs} ")
        for ind in sorted_db[:self.decision_box.ref_index]:
            self.log(f"{ind}, cost = {ind['cost']}")

    def update_time_info(self, total_time):
        self.n_sims_list.append(self.n_sims)
        self.sims_time_list.append(self.sim_time)
        self.n_nn_queries_list.append(self.n_queries)
        self.nn_query_time_list.append(self.query_time)
        self.n_training_list.append(self.n_training)
        self.training_time_list.append(self.training_time)
        self.total_time_list.append(total_time)

    def store_database_and_times(self):
        dict_to_save = dict(
            db=self.data_set_list,
            n_nn_query=self.n_nn_queries_list,
            nn_query_time=self.nn_query_time_list,
            n_sims=self.n_sims_list,
            sims_time=self.sims_time_list,
            n_training=self.n_training_list,
            training_time=self.training_time_list,
            total_time=self.total_time_list,
        )
        self._logger.store_db(dict_to_save, fpath=osp.join(self._logger.log_path, 'db_time.pkl'))
