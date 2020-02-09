"""
The base agent module which takes care of initializing the Logger, DecisionBox, black box
environment.
"""
from typing import Union, Type, cast

import random
import numpy as np
import warnings
from pathlib import Path

from ..logger import Logger

from utils.file import read_yaml
from utils.importlib import import_class
from utils.data.database import Database

from ..ea.ea import EA
from bb_eval_engine.util.importlib import import_bb_env
from bb_eval_engine.data.design import Design
from bb_eval_engine.base import EvaluationEngineBase

from ..helper_module import DecisionBox


class Agent:

    def __init__(self, fname: Union[str, Path]):

        _params = read_yaml(fname)
        self.specs = _params['agent_params']
        seed = self.specs['seed']
        self._set_seed(seed)

        self.db: Database = Database(Design, allow_repeat=False)
        self.data_set_list = []

        # create evaluation core instance
        self.bb_env: EvaluationEngineBase = import_bb_env(_params['bb_env'])

        # create the ea algorithm
        ea_cls = cast(Type[EA], import_class(_params['ea_cls']))
        self.ea = ea_cls(**_params['ea_params'], eval_core=self.bb_env)

        self._logger = Logger(log_path=_params['outputs'])
        # self._logger.store_settings(fname, self.specs['circuit_yaml_file'])

        # self.circuit_content = read_yaml(self.specs['circuit_yaml_file'])
        # self.init_pop_dir = self.circuit_content['database_dir']
        # self.num_params_per_design = self.eval_core.num_params

        self.init_data_path = Path(_params['outputs']) / 'init_data.pickle'
        self.n_init_samples = self.specs['n_init_samples']

        self.decision_box = DecisionBox(self.specs['ref_dsn_idx'],
                                        self.bb_env,
                                        self._logger)

    def log(self, msg: str):
        self._logger.log_text(msg)

    def info(self, msg: str):
        self._logger.info(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def get_init_population(self, re_sim: bool = False):
        # load/create db
        # if re_sim is False and self.init_data_path.exists():
        #     self.db = read_pickle(self.init_data_path)
        # else:
        init_designs = self.bb_env.generate_rand_designs(self.n_init_samples, evaluate=True)
        self.db.extend(init_designs)
        # write_pickle(self.init_data_path, self.db)

        if len(self.db) < self.n_init_samples:
            warnings.warn('Number of init_samples is larger than the length of the initial '
                          'database, using the len(db) instead of n_init_samples', RuntimeWarning)

        # self.db = clean(self.db, self.bb_env)
        # self.db = relable(self.db, self.bb_env)
        db_sorted = sorted(self.db, key=lambda x: x['cost'])
        # HACK for paper
        # self.db = self.db[1:]
        self._logger.log_text(f'[INFO] Best cost in init_pop = {db_sorted[0]["cost"]}')


    def _set_seed(self, seed: int):
        """override to initialize other libraries of use. e.g. pytorch"""
        random.seed(seed)
        np.random.seed(seed)

    def main(self):
        raise NotImplementedError
