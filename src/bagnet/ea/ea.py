from typing import List, Optional

from bb_eval_engine.data.design import Design

class EA:

    def __init__(self, eval_core, *args, **kwargs) -> None:
        self.eval_core = eval_core

    def get_next_generation_candidates(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_for_generation(self, db: List[object], n):
        raise NotImplementedError


    def update_value_dict_offsprings_inplace(self, offsprings):
        for child in offsprings:
            for idx, (param_name, param_vec) in enumerate(self.eval_core.params_vec.items()):
                child.value_dict[param_name] = param_vec[child[idx]]


def set_parents_and_sibling(design: Design, parent1: Design, parent2: Optional[Design],
                            sibling: Optional[Design]):
    design['parent1'] = parent1
    design['parent2'] = parent2
    design['sibiling'] = sibling

def is_init_population(dsn):
    if dsn.get('parent1', None) is None:
        return True
    else:
        return False

def is_mutated(dsn: Design):
    if dsn.get('parent1', None) is not None:
        if dsn.get('parent2', None) is None:
            return True
    else:
        return False

def genocide(*args: Design):
    for dsn in args:
        for family in ['parent1', 'parent2', 'sibiling']:
            if family in dsn:
                del dsn[family]