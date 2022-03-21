import pprint
import os
import time
from shutil import copyfile
import pickle

from colorama import init
from termcolor import colored

from pathlib import Path

class Logger:

    def __init__(self, log_path, seed=0, time_stamped=True):
        init()
        if Path(log_path).is_file():
            raise ValueError('{} is not a file path, please provide a directory path')

        self.log_path = Path(log_path).absolute()
        folder_name = f's_{seed}'
        if time_stamped:
            folder_name += '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        self.log_path /= folder_name

        Path(self.log_path).mkdir(exist_ok=True, parents=True)

        self.log_txt_fname = self.log_path / 'progress_log.txt'
        if self.log_txt_fname.exists():
            os.remove(self.log_txt_fname)

        self.log_db_fname = self.log_path / 'db.pkl'

        log_model_path = self.log_path / 'checkpoint'
        self.log_model_fname = log_model_path / 'checkpoint.ckpt'


    def log_text(self, str, stream_to_file=True, stream_to_stdout=True, pretty=False, fpath=None):
        if fpath:
            stream = open(fpath, 'a')
        else:
            stream = open(self.log_txt_fname, 'a')

        if pretty:
            printfn = pprint.pprint
        else:
            printfn = print

        if stream_to_file:
            printfn(str, file=stream)
        if stream_to_stdout:
            printfn(str)

        stream.close()

    def debug(self, str, stream_to_file=True, stream_to_stdout=True, pretty=False, fpath=None):
        msg = colored('[DEBUG] ', 'red') + str
        self.log_text(msg, stream_to_file, stream_to_stdout, pretty, fpath)

    def info(self, str, stream_to_file=True, stream_to_stdout=True, pretty=False, fpath=None):
        msg = colored('[INFO] ', 'green') + str
        self.log_text(msg, stream_to_file, stream_to_stdout, pretty, fpath)


    def store_db(self, db, fpath=None):
        if fpath is None:
            fpath = self.log_db_fname

        with open(fpath, 'wb') as f:
            pickle.dump(db, f)

    def store_model(self, *args):
        if len(args) == 2:
            self.store_model_tf(*args)
        else:
            self.store_model_torch(*args)

    def store_model_tf(self, tf_saver, tf_session):
        tf_saver.save(tf_session, str(self.log_model_fname))

    def store_model_torch(self, model):
        import torch
        self.log_model_fname.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.log_model_fname)

    def store_settings(self, agent_yaml, circuit_yaml):
        agent_fname = Path(agent_yaml).absolute()
        circuit_fname = Path(circuit_yaml).absolute()
        copyfile(agent_fname, self.log_path / 'agent.yaml')
        copyfile(circuit_fname, self.log_path / 'circuit.yaml')
