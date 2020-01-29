"""
The base agent module which takes care of initializing the Logger, DecisionBox, black box
environment.
"""
from typing import Union

import random
from pathlib import Path

from ..util.ga import clean, relable
from ..logger import Logger

from utils.file import read_yaml
from utils.importlib import import_class
from utils.file import read_pickle, write_pickle
from bb_eval_engine.util.importlib import import_bb_env

class Agent:

    def __init__(self, fname: Union[str, Path]):

        _params = read_yaml(fname)
        self.specs = _params['agent_params']

        self.db = None
        self.data_set_list = []

        # create evaluation core instance
        self.bb_env = import_bb_env(_params['bb_env'])

        # create the ea algorithm
        ea_cls = import_class(_params['ea_cls'])
        self.ea = ea_cls(**_params['ea_params'], eval_core=self.bb_env)

        self._logger = Logger(log_path=_params['outputs'])
        # self._logger.store_settings(fname, self.specs['circuit_yaml_file'])

        # self.circuit_content = read_yaml(self.specs['circuit_yaml_file'])
        # self.init_pop_dir = self.circuit_content['database_dir']
        # self.num_params_per_design = self.eval_core.num_params

        self.init_data_path = Path(_params['outputs']) / 'init_data.pickle'
        self.n_init_samples = self.specs['n_init_samples']


    def log(self, msg: str):
        self._logger.log_text(msg)

    def get_init_population(self, re_sim: bool = False):
        # load/create db
        if re_sim is False and self.init_data_path.exists():
            self.db = read_pickle(self.init_data_path)
        else:
            self.db = self.bb_env.generate_rand_designs(self.n_init_samples, evaluate=True)
            write_pickle(self.init_data_path, self.db)

        if len(self.db) >= self.n_init_samples:
            random.shuffle(self.db)
            self.db = self.db[:self.n_init_samples]
        else:
            raise Warning('Number of init_samples is larger than the length of the '
                          'initial data base, using the len(db) instead of n_init_samples')

        # self.db = clean(self.db, self.bb_env)
        # self.db = relable(self.db, self.bb_env)
        self.db = sorted(self.db, key=lambda x: x.cost)
        # HACK for paper
        # self.db = self.db[1:]
        self._logger.log_text(f'[INFO] Best cost in init_pop = {self.db[0].cost}')

    def main(self):
        raise NotImplementedError
