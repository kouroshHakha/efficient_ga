from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def init(self):
        """
        After call to init the caller has access to the following attributes:

        self.sess: tf.Sess
        self.input1, self.input2:
        self.true_labels:
        self.mu, self.std:
        self.update_op
        self.accuracy
        self.total_loss
        self.loss
        self.out_predictions

        """
        raise NotImplementedError

    @abstractmethod
    def get_train_valid_ds(self, *args, **kwargs):
        """
        returns training and validation datasets with their proper output labels for the model in
        numpy arrays
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        """

        Trains the model the way it should be trained, this should be implemented compatible with
        get_train_valid_ds and model parameters
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, **kwargs):
        raise NotImplementedError