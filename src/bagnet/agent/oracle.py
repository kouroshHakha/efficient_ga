from typing import Union


import time
from pathlib import Path
from copy import deepcopy
import os.path as osp

from .base import Agent
from ..util.ga import is_x_better_than_y


class OracleAgent(Agent):

    def __init__(self, fname: Union[str, Path]):
        Agent.__init__(self, fname)
        # get the parameters in agent_params
        self.max_n_retraining = self.specs['max_n_steps']

        # from db designs with indices less than this number are considered during training
        self.k = self.specs['k']
        self.max_iter = self.specs['max_iter']

        self.n_new_samples = self.specs['n_new_samples']

        # paper stuff variables
        self.n_queries = 0
        self.query_time = 0
        self.n_queries_list, self.query_time_list, self.total_time_list = [], [], []
        self.oracle_save_to = self.specs.get("oracle_save_to", None)
        if self.oracle_save_to:
            self.oracle_queries = dict(
                inputs1=[],
                inputs2=[],
                critical_specs=[]
            )
            for kwrd in self.bb_env.spec_range.keys():
                self.oracle_queries[kwrd] = []

    def run_per_iter(self):
        db_list = list(self.db)
        self.decision_box.update_heuristics(db_list, self.k)

        if self.decision_box.has_converged():
            return [], True

        parent1, parent2, ref_design = self.decision_box.get_parents_and_ref_design(db_list, self.k)

        offsprings = []
        n_iter = 0

        self.log(30*"-")
        self.log("[INFO] running model ... ")
        q_start = time.time()

        self.ea.prepare_for_generation(db_list, self.k)

        while len(offsprings) < self.n_new_samples and n_iter < self.max_iter:
            new_designs = self.ea.get_next_generation_candidates(parent1, parent2)

            for new_design in new_designs:
                if new_design in self.db or new_design in offsprings:
                    # if design is already in the design pool skip ...
                    self.debug(f"design {new_design} already exists")
                    continue

                n_iter += 1
                new_design = self.bb_env.evaluate([new_design])[0]
                if new_design['valid']:
                    all_better = True
                    for kwrd in self.decision_box.critical_specs:
                        new = new_design.specs[kwrd]
                        ref = ref_design.specs[kwrd]
                        all_better = all_better & is_x_better_than_y(self.bb_env, new, ref, kwrd)

                    if all_better:
                        offsprings.append(new_design)

                    if self.oracle_save_to:
                        self.store_oracle_query(new_design, ref_design,
                                                self.decision_box.critical_specs)

                self.n_queries += 1

        self.query_time += time.time() - q_start

        if len(offsprings) < self.n_new_samples:
            return offsprings, True

        self.log(30*"-")

        self.info(f"New designs tried: {n_iter}")
        self.info(f"New candidates size: {len(offsprings)}")
        for child in offsprings:
            self.info(f"Added: {child} , cost = {child['cost']}")

        return offsprings, False

    def store_oracle_query(self, input1, input2, critical_specs):
        self.oracle_queries['inputs1'].append(deepcopy(input1))
        self.oracle_queries['inputs2'].append(deepcopy(input2))
        self.oracle_queries['critical_specs'].append(critical_specs.copy())
        for kwrd in self.bb_env.spec_range.keys():
            self.oracle_queries[kwrd].append(is_x_better_than_y(self.bb_env,
                                                                input1.specs[kwrd],
                                                                input2.specs[kwrd],
                                                                kwrd))
        self._logger.store_db(self.oracle_queries, fpath=self.oracle_save_to)

    def main(self):
        start = time.time()

        self.get_init_population()
        self.data_set_list.append(list(self.db))
        self.query_time_list.append(self.query_time)
        self.n_queries_list.append(self.n_queries)
        self.total_time_list.append(0)

        for i in range(self.max_n_retraining):
            self.info(f'********** Iter {i} **********')
            offsprings, is_converged = self.run_per_iter()

            if is_converged:
                break
            elif len(offsprings) == 0:
                continue

            self.db.extend(offsprings)
            self.data_set_list.append(offsprings)
            self.query_time_list.append(self.query_time)
            self.n_queries_list.append(self.n_queries)
            self.total_time_list.append((time.time()-start))
            self._logger.store_db(self.data_set_list)
            self.store_database_and_times()

            # adjust dataset size for training, if not desired, comment the agent.k_top= ... line
            db_sorted = sorted(self.db, key=lambda x: x['cost'])
            worst_offspring = max(offsprings, key=lambda x: x['cost'])
            self.info(f'k_top alternative: {db_sorted.index(worst_offspring)}')
            self.k = max(self.n_init_samples, db_sorted.index(worst_offspring))

            self.info(f"Nqueries = {self.n_queries}")
            self.info(f"Query_time = {self.query_time}")
            self.info(f"Total_time = {time.time()-start}")

        self._logger.store_db(self.data_set_list)

        sorted_db = sorted(self.db, key=lambda x: x['cost'])
        # paper stuff
        self.info(f"Nqueries = {self.n_queries}")
        self.info(f"Query_time = {self.query_time}")
        self.info(f"Total_time = {time.time()-start}")
        self.info(f"Total_n_evals = {len(self.db)}")
        self.info(f"Best_solution = {sorted_db[0]}")
        self.info(f"id = {sorted_db[0]['id']}")
        self.info(f"Cost = {sorted_db[0]['cost']}")
        self.info(f"Performance \n{sorted_db[0].specs} ")
        for ind in sorted_db[:self.decision_box.ref_index]:
            self.log(f"{ind}, cost = {ind['cost']}")

    def store_database_and_times(self):
        dict_to_save = dict(
            db=self.data_set_list,
            n_query=self.n_queries_list,
            query_time=self.query_time_list,
            total_time=self.total_time_list,
        )
        self._logger.store_db(dict_to_save, fpath=osp.join(self._logger.log_path, 'db_time.pkl'))
