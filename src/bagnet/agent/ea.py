"""
This module implements the normal evolutionary algorithm (not boosted)
"""
from typing import Union, cast, List

from pathlib import Path
import time
import os.path as osp

from .base import Agent


class EvoAgent(Agent):

    def __init__(self, fname: Union[str, Path]) -> None:
        """

        Parameters
        ----------
        fname: Union[str, Path]
            Yaml spec. Other than the files in Agent requirements, it should include the following:
                n_new_samples: number of new samples added in each iteration
                k: number of k sample to consider when forming heuristics in decision box ...
                max_n_steps: maximum number of iterations (i.e. generations)
        """
        Agent.__init__(self, fname)
        self.k = self.specs['k']
        self.max_n_steps = self.specs['max_n_steps']
        self.n_new_samples = self.specs['n_new_samples']

        self.n_sims = 0
        self.sim_time = 0
        self.n_sims_list, self.sim_time_list, self.total_time_list = [], [], []

    def run(self):
        db_list = list(self.db)

        self.decision_box.update_heuristics(db_list, self.k)
        if self.decision_box.has_converged():
            return [], True

        parent1, parent2, _ = self.decision_box.get_parents_and_ref_design(db_list, self.k)

        offsprings = []
        n_iter = 0

        self.log(30*"-")
        self.info("Running model ... ")

        self.ea.prepare_for_generation(db_list, self.k)
        while len(offsprings) < self.n_new_samples:
            new_designs = self.ea.get_next_generation_candidates(parent1, parent2)

            for new_design in new_designs:
                if new_design in self.db or new_design in offsprings:
                    # if design is already in the design pool skip ...
                    self.debug(f"Design {new_design} already exists")
                    continue

                n_iter += 1
                offsprings.append(new_design)

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

        return offsprings, False

    def main(self):
        start = time.time()

        self.get_init_population()
        self.data_set_list.append(list(self.db))
        self.n_sims_list.append(self.n_sims)
        self.sim_time_list.append(self.sim_time)
        self.total_time_list.append(0)

        for i in range(self.max_n_steps):
            offsprings, is_converged = self.run()

            if is_converged:
                break
            elif len(offsprings) == 0:
                continue

            self.db.extend(offsprings)
            self.data_set_list.append(offsprings)
            self.n_sims_list.append(self.n_sims)
            self.sim_time_list.append(self.sim_time)
            self.total_time_list.append(time.time()-start)
            if i % 10 == 0:
                self._logger.store_db(self.data_set_list)
                self.store_database_and_times()

            self.info(f"n_iter = {i}")
            self.info(f"n_simulations = {self.n_sims}")
            self.info(f"sim_time = {self.sim_time}")
            self.info(f"total_time = {time.time()-start}")

        self._logger.store_db(self.data_set_list)
        self.store_database_and_times()

        sorted_db = sorted(self.db, key=lambda x: x['cost'])

        self.info("n_simulations = {}".format(self.n_sims))
        self.info("sim_time = {}".format(self.sim_time))
        self.info("total_time = {}".format(time.time()-start))

        for ind in sorted_db[:10]:
            self.log(f"{ind} -> {ind['cost']}")

    def store_database_and_times(self):
        dict_to_save = dict(
            db=self.data_set_list,
            n_query=self.n_sims_list,
            query_time=self.sim_time_list,
            total_time=self.total_time_list,
        )
        self._logger.store_db(dict_to_save, fpath=osp.join(self._logger.log_path, 'db_time.pkl'))
