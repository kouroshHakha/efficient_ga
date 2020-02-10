"""
The base agent module which takes care of initializing the Logger, DecisionBox, black box
environment.
"""
from typing import Union, Type, cast

import random
import numpy as np
import warnings
from pathlib import Path

from utils.file import read_yaml
from utils.importlib import import_class
from utils.data.database import Database

from bb_eval_engine.util.importlib import import_bb_env
from bb_eval_engine.data.design import Design
from bb_eval_engine.base import EvaluationEngineBase

from ..ea.ea import EA
from ..util.logger import Logger
from ..decisionbox import DecisionBox


class Agent:

    def __init__(self, fname: Union[str, Path]) -> None:
        """
        Parameters
        ----------
        fname: Path
            Yaml Path. It should contain the following fields:
                outputs
                bb_env
                agent_params
                    n_init_samples
                    ref_dsn_idx
                ea_cls
                ea_params
        """

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

    def get_init_population(self):
        init_designs = self.bb_env.generate_rand_designs(self.n_init_samples, evaluate=True)
        self.db.extend(init_designs)

        if len(self.db) < self.n_init_samples:
            warnings.warn('Number of init_samples is larger than the length of the initial '
                          'database, using the len(db) instead of n_init_samples', RuntimeWarning)

        db_sorted = sorted(self.db, key=lambda x: x['cost'])
        self._logger.log_text(f'[INFO] Best cost in init_pop = {db_sorted[0]["cost"]}')

    def _set_seed(self, seed: int):
        """override to initialize other libraries of use. e.g. pytorch"""
        random.seed(seed)
        np.random.seed(seed)

    def main(self):
        raise NotImplementedError
