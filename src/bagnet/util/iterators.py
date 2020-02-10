import random

class BatchIndexIterator:
    """A class for iterating through indices of a dataset to produce a batch via slicing"""
    def __init__(self, data_set_size, batch_size):
        self._data_size = data_set_size
        self._batch_size = batch_size
        self._segment = self._data_size // batch_size
        self.last_index = 0
        self._permutations = list(range(data_set_size))
        random.shuffle(self._permutations)

    def next(self):

        if ((self.last_index+1)*self._batch_size > self._data_size):
            indices1 = self._permutations[self.last_index * self._batch_size:]
            indices2 = self._permutations[:((self.last_index+1)*self._batch_size)%self._data_size]
            indices = indices1 + indices2
        else:
            indices = self._permutations[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size]

        self.last_index = (self.last_index+1) % (self._segment+1)
        return indices