import random
import numpy as np
from bagnet.util.ga import is_x_better_than_y
from bagnet.util.iterators import BatchIndexIterator
from .model import Model
# import tensorflow as tf
import pandas as pd
import pickle
import os
import itertools
import tqdm

import torch
from torch.optim import Adam

from torch.utils.data import Subset
from torch.utils.data import DataLoader as VecLoader
from torch_geometric.loader import DataLoader as GNNLoader
from torch_geometric.loader.dataloader import Collater
from pytorch_lightning.callbacks import ModelCheckpoint

from cgl.bagnet_gnn.model import BagNetComparisonModel
from cgl.bagnet_gnn.data import BagNetDataset, BagNetOnlineDataset, InputHandler

class ModelCheckpointNoOverride(ModelCheckpoint):
    """This checkpoint call back does not delete the already saved checkpoint"""
    def _del_model(self, filepath: str) -> None:
        pass


class SimpleBagNetDataset(BagNetDataset):
            
    def __init__(self, design_list, eval_core, is_graph):
        super().__init__(is_graph=is_graph)
        self.design_list = design_list
        self.eval_core = eval_core


    def __len__(self):
        n = len(self.design_list)
        return n * (n-1) // 2

    def _get_paired_idx(self):
        # imagine an upper triangular matric of pairs
        # select row r = rand(0, n-2), excluding r_{n-1}
        # then select the col = rand(r+1, n)
        n = len(self.design_list)
        idx_a = np.random.randint(n - 1)
        idx_b = np.random.randint(idx_a + 1, n)

        if np.random.rand() < 0.5:
            idx_a, idx_b = idx_b, idx_a

        return idx_a, idx_b

    def __getitem__(self, idx):

        idx_a, idx_b = self._get_paired_idx()

        dsn_a = self.design_list[idx_a]
        dsn_b = self.design_list[idx_b]

        input_a = self.input_handler.get_input_repr(dict(params=dsn_a.value_dict))
        input_b = self.input_handler.get_input_repr(dict(params=dsn_b.value_dict))

        label = {}
        for kwrd in self.eval_core.spec_range:
            label[kwrd] = torch.tensor(is_x_better_than_y(eval_core=self.eval_core,
                                                              x=dsn_a.specs[kwrd],
                                                              y=dsn_b.specs[kwrd],
                                                              kwrd=kwrd)).long()

            # if kwrd in ('gain', 'ugbw', 'pm', 'psrr', 'cmrr'):
            #     label[kwrd] = torch.tensor(dsn_a.specs[kwrd] > dsn_b.specs[kwrd]).long()
            # else:
            #     label[kwrd] = torch.tensor(dsn_a.specs[kwrd] <= dsn_b.specs[kwrd]).long()

        return dict(input_a=input_a, input_b=input_b, **label)
    

class SimpleModel(Model):

    def __init__(self,
                 num_params_per_design,
                 spec_kwrd_list,
                 logger,
                 learning_rate=None,
                #  compare_nn_hidden_dim_list,
                #  feat_ext_hidden_dim_list,
                 **kwargs,
                 ):
        Model.__init__(self, **kwargs)
        self.kwargs = kwargs
        self.seed = kwargs.get('seed', None)
        self.num_params_per_design = num_params_per_design
        self.spec_kwrd_list = spec_kwrd_list
        # self.feat_ext_dim_list = [num_params_per_design] + feat_ext_hidden_dim_list
        # self.compare_nn_dim_list = \
        #     [2*feat_ext_hidden_dim_list[-1]] + compare_nn_hidden_dim_list + [2]
        self.lr = learning_rate 
        self.logger = logger

        self.evaluate_flag = True if 'eval_save_to_path' in kwargs.keys() else False
        if self.evaluate_flag:
            self.file_base_name = 'acc'
            self._initialize_evaluation(**kwargs)

    def _initialize_evaluation(self, **kwargs):
        self.eval_save_to = kwargs['eval_save_to_path']
        if self.eval_save_to == 'log_path':
            self.eval_save_to = self.logger.log_path
        os.makedirs(self.eval_save_to, exist_ok=True)

        self.df_accuracy = pd.DataFrame()
        oracle_db_loc = kwargs['oracle_db_loc']
        with open(oracle_db_loc, 'rb') as f:
            oracle_data = pickle.load(f)

        self.df = pd.DataFrame.from_dict(oracle_data)
        keys = oracle_data['inputs1'][0].specs.keys()

        # creates a vector for all designs indicating whether input1 is better with respect to
        # critical designs
        self.oracle_is_1_better = []
        for index, row in self.df.iterrows():
            is_1_better = all(row[row['critical_specs']])
            self.oracle_is_1_better.append(is_1_better)

        self.oracle_input1 = np.array(self.df["inputs1"].tolist())
        self.oracle_input2 = np.array(self.df["inputs2"].tolist())

        # for true labels we should provide one hot encoded versions, so:
        # 1. get the colomn df.as_matrix(columns=[kwrd] as matrix and flatten it.
        # 2. multiply it by 1 to get all 1s and 0s.
        # 3. use it as the indices and create one hot encoded vector
        self.labels = dict()
        for kwrd in keys:
            labels = np.zeros((len(self.df), 2))
            col_num = (self.df.as_matrix(columns=[kwrd]).flatten())*1
            labels[np.arange(len(self.df)), col_num] = 1
            self.labels[kwrd] = labels

        self.acc_txt_file = os.path.join(self.eval_save_to, self.file_base_name + ".txt")
        if os.path.exists(self.acc_txt_file):
            os.remove(self.acc_txt_file)


    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def init(self):
        if self.seed is not None:
            self._set_seed(self.seed)

        gnn_ckpt = self.kwargs.get('gnn_ckpt', '')
        self.is_graph = bool(gnn_ckpt)
        dropout = 1 - self.kwargs.get('keep_prob')

        if gnn_ckpt:
            feature_ext_config = dict(
                gnn_ckpt_path=gnn_ckpt,
                output_features=self.kwargs['feat_out_dim'],
                rand_init=self.kwargs.get('rand_init', False),
                freeze=False,
            )
        else:
            feature_ext_config = dict(
                input_features=self.num_params_per_design,
                output_features=self.kwargs['feat_out_dim'],
                hidden_dim=self.kwargs['feat_hdim'],
                n_layers=self.kwargs['feat_nlayer'],
                drop_out=dropout,
            )

        comparison_config = dict(
            hidden_dim=self.kwargs['comp_hdim'],
            n_layers=self.kwargs['comp_nlayer'],
            drop_out=dropout,
        )

        self.nn = BagNetComparisonModel(
            comparison_kwrds=self.spec_kwrd_list,
            feature_exractor_config=feature_ext_config,
            comparison_model_config=comparison_config,
            is_gnn=self.is_graph,
        )

        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn.to(device)
        
        if self.is_graph:
            self.nn.feature_extractor.gnn.to(self.device)

        self.input_handler = InputHandler(is_graph=self.is_graph)
        self.LOADER = GNNLoader if self.is_graph else VecLoader

    def get_train_valid_ds(self, db, k, eval_core, validation_frac):
        """
        There are four possible ways to do this:
        1. sort the database, choose top k, where k can be adjusted from outside in case error
        is too large in low cost regions
        2. sort the database, choose top k (i.e. 100)
        3. Randomly picks k elements in db and constructs the model training set, the
        remaining un-chosen elements will not contribute to gradient updates in this round.
        4. construct the training set based on the entire db and shuffle it. In the batch update,
        take constant number of gradient updates. In this way, all elements in db can contribute.
        """

        train_db = list(db)
        # train_db = random.choices(db, k=k)
        # db_cost_sorted = sorted(db, key=lambda x: x['cost'])[:k]

        category = {}
        nn_input1, nn_input2 = [], []
        nn_labels = {}
        for kwrd in self.spec_kwrd_list:
            category[kwrd] = []

        n = len(train_db)
        for i in range(n-1):
            for j in range(i+1, n):
                rnd = random.random()
                if rnd < 0.5:
                    nn_input1.append(train_db[i].value_dict)
                    nn_input2.append(train_db[j].value_dict)
                    for kwrd in self.spec_kwrd_list:
                        label = 1 if is_x_better_than_y(eval_core=eval_core,
                                                        x=train_db[i].specs[kwrd],
                                                        y=train_db[j].specs[kwrd],
                                                        kwrd=kwrd) else 0
                        category[kwrd].append(label)
                else:
                    nn_input1.append(train_db[j].value_dict)
                    nn_input2.append(train_db[i].value_dict)
                    for kwrd in self.spec_kwrd_list:
                        label = 0 if is_x_better_than_y(eval_core=eval_core,
                                                        x=train_db[i].specs[kwrd],
                                                        y=train_db[j].specs[kwrd],
                                                        kwrd=kwrd) else 1
                        category[kwrd].append(label)

        self.logger.info(f"Dataset size: {len(db)}")
        self.logger.info(f"K : {k}")
        self.logger.info(f"Combine size: {len(nn_input1)}")

        permutation = np.random.permutation(len(nn_input1))
        nn_input1 = [nn_input1[i] for i in permutation]
        nn_input2 = [nn_input2[i] for i in permutation]
        for kwrd in self.spec_kwrd_list:
            nn_labels[kwrd] = [category[kwrd][i] for i in permutation]

        boundry_index = len(nn_input1) - int(len(nn_input1)*validation_frac)

        train_input1 = nn_input1[:boundry_index]
        train_input2 = nn_input2[:boundry_index]
        valid_input1 = nn_input1[boundry_index:]
        valid_input2 = nn_input2[boundry_index:]

        train_labels, valid_labels = {}, {}
        for kwrd in self.spec_kwrd_list:
            train_labels[kwrd] = nn_labels[kwrd][:boundry_index]
            valid_labels[kwrd] = nn_labels[kwrd][boundry_index:]

        ds = {
            'training_ds': dict(
                input_a=train_input1,
                input_b=train_input2,
                labels=train_labels,
            ),
            'validation_ds': dict(
                input_a=valid_input1,
                input_b=valid_input2,
                labels=valid_labels,
            )
        }
        return ds

    def ff(self, batch):
        model_output = self.nn(batch)

        loss = torch.tensor(0., device=self.device)
        total_heads = len(self.nn.comparison_heads)
        for key in self.nn.comparison_heads:
            loss += model_output['losses'][key] / total_heads

        acc = {}
        all_correct = None
        for key in self.nn.comparison_heads:
            pred = model_output['outputs'][key]['prob'].argmax(-1)
            target = batch[key]

            cond = pred == target
            acc[f'acc_{key}'] = cond.float().mean(0).detach()
            if all_correct is None:
                all_correct = cond
            else:
                all_correct = all_correct & cond

        acc['acc_all'] = all_correct.float().mean(0).detach()

        return {'loss': loss, **acc}


    def train(self, db, eval_core, batch_size, ngrad_step_per_run, ckpt_step, log_step):
        
        dataset = SimpleBagNetDataset(db, eval_core=eval_core, is_graph=self.is_graph)
        # split is not really important here since we randomly generate index on the fly
        train_set = dataset
        valid_set = Subset(train_set, list(range(1000)))
        # train_set = BagNetOnlineDataset(**data_set['training_ds'], is_graph=self.is_graph)
        # valid_set = BagNetOnlineDataset(**data_set['validation_ds'], is_graph=self.is_graph)

        # mean = self.mean = np.concatenate([data_set['training_ds']['input_a'], data_set['training_ds']['input_b']], 0).mean(0)
        # std = self.std = np.concatenate([data_set['training_ds']['input_a'], data_set['training_ds']['input_b']], 0).std(0)
        # class CustomDataset(Dataset):

        #     def __init__(self, input_a, input_b, labels) -> None:
        #         super().__init__()
        #         self.input_a = input_a
        #         self.input_b = input_b
        #         self.labels = labels

        #     def __len__(self):
        #         return len(self.input_a)

        #     def __getitem__(self, index):
        #         input_a = torch.as_tensor((self.input_a[index] - mean) / std).float()
        #         input_b = torch.as_tensor((self.input_b[index] - mean) / std).float()

        #         # input_a = self.input_a[index]
        #         # input_b = self.input_b[index]
        #         label = {k: torch.tensor(v[index]).long() for k, v in self.labels.items()}
        #         return dict(input_a=input_a, input_b=input_b, **label)


        # train_set = CustomDataset(**data_set['training_ds'])
        # valid_set = CustomDataset(**data_set['validation_ds'])

        # loader = self.LOADER(train_set, len(train_set))
        # batch = next(iter(loader))
        # print(batch['input_a'])
        # breakpoint()

        total_n_batches = int(len(train_set) // batch_size)

        self.logger.info("Training the model with dataset ....")
        self.logger.info(f"Number of total batches: {total_n_batches}")
        self.logger.log_text(30*"-")

        train_loader = self.LOADER(train_set, batch_size, num_workers=4, shuffle=True)
        valid_loader = self.LOADER(valid_set, batch_size, num_workers=4)

        optimizer = Adam(self.nn.parameters(), lr=self.lr)

        train_acc_list = {k: [] for k in self.spec_kwrd_list}
        total_loss_list = []

        pbar = tqdm.tqdm(range(ngrad_step_per_run), total=ngrad_step_per_run)
        desc_map = {}
        iter_cnt = 0
        train_done = False
        while not train_done:
            for batch in train_loader:

                if iter_cnt > ngrad_step_per_run:
                    train_done = True
                    break

                self.nn.train()
                optimizer.zero_grad()
                # clone this so that any in-place op on the batch values would not persist
                batch = {k: v.clone().to(self.device) for k, v in batch.items()}
                ret = self.ff(batch)
                ret['loss'].backward()
                optimizer.step()
                
                desc_map['loss'] = ret['loss'].item()
                total_loss_list.append(ret['loss'].item())
                for k in self.spec_kwrd_list:
                    train_acc_list[k].append(ret[f'acc_{k}'].item())


                if iter_cnt % ckpt_step == 0:
                    self.logger.store_model(self.nn)
                if iter_cnt % log_step == 0:
                    # running validation
                    self.nn.eval()
                    valid_acc_list = {k: [] for k in self.spec_kwrd_list}
                    valid_acc_list.update(all=[])
                    for vbatch in valid_loader:
                        vbatch = {k: v.to(self.device) for k, v in vbatch.items()}
                        ret_vbatch = self.ff(vbatch)
                        for k in valid_acc_list:
                            valid_acc_list[k].append(ret_vbatch[f'acc_{k}'].item())

                    self.logger.log_text(10*"-", stream_to_stdout=False)
                    self.logger.log_text(f"[iter {iter_cnt}] total_loss: {np.mean(total_loss_list)}", stream_to_stdout=True)
                    desc_map['log_loss'] = np.mean(total_loss_list)
                    for kwrd in self.spec_kwrd_list:
                        # self.logger.log_text(f"{kwrd}")
                        # self.logger.log_text(f"[{kwrd}] loss: {np.mean(loss_list[kwrd])}")
                        self.logger.log_text(f"[{kwrd:10}] "
                                            f"train_acc = {np.mean(train_acc_list[kwrd]) * 100:.2f}%,"
                                            f"valid_acc = {np.mean(valid_acc_list[kwrd]) * 100:.2f}%", stream_to_stdout=True)

                    desc_map['valid_acc_all'] = np.mean(valid_acc_list['all'])
                    # reset the list of the next round until we log again
                    train_acc_list = {k: [] for k in self.spec_kwrd_list}
                    total_loss_list = []

                desc = ''
                desc_str_list = []
                for name, val in desc_map.items():
                    desc_str_list.append(f'{name}={val:.4f}') 
                
                desc = ', '.join(desc_str_list)
                pbar.set_description(desc)
                pbar.update()
                iter_cnt += 1
            

    def query(self, input1, input2):
        input_a = self.input_handler.get_input_repr(dict(params=input1.value_dict))
        input_b = self.input_handler.get_input_repr(dict(params=input2.value_dict))

        # input_a = torch.as_tensor((np.array(input1) - self.mean) / self.std)
        # input_b = torch.as_tensor((np.array(input2) - self.mean) / self.std)

        # input_a = torch.as_tensor((np.array(list(input1.value_dict.values())) - self.mean) / self.std)
        # input_b = torch.as_tensor((np.array(list(input2.value_dict.values())) - self.mean) / self.std)

        if self.is_graph:
            batch = Collater(follow_batch=None, exclude_keys=None)(
                [
                    dict(
                        input_a=input_a.to(self.device), 
                        input_b=input_b.to(self.device)
                    )
                ]
            )
        else:
            batch = dict(
                input_a=input_a[None].float().to(self.device), 
                input_b=input_b[None].float().to(self.device)
            )

        self.nn.eval()
        model_output = self.nn(batch) 

        predictions = {}
        for key in self.spec_kwrd_list:
            predictions[key] = model_output['outputs'][key]['prob'].detach().cpu().numpy()

        return predictions

    # def evaluate(self):
    #     "A function that evaluates the nn with oracle data to see how they compare"
    #     assert self.evaluate_flag, 'To evaluate the evalute flage must be set to True'

    #     oracle_feed_dict = {
    #         self.input1: self.oracle_input1,
    #         self.input2: self.oracle_input2,
    #     }

    #     for kwrd, tensor in self.true_labels.items():
    #         oracle_feed_dict[tensor] = self.labels[kwrd]

    #     accuracy, predictions = self.sess.run([self.accuracy, self.out_predictions],
    #                                           feed_dict=oracle_feed_dict)

    #     # see if nn says input1 is better than input2 for all rows according to the critical specs
    #     nn_is_1_better = []
    #     for i in range(len(self.df)):
    #         is_1_better = all([random.random() > predictions[kwrd][i][0]
    #                            for kwrd in self.df['critical_specs'][i]])
    #         nn_is_1_better.append(is_1_better)


    #     # compute all accuracy numbers (oracle_nn): false_false, true_true, false_true, true_false
    #     ff, tt = 0, 0
    #     ft, tf = 0, 0
    #     for nn_vote, oracle_vote in zip(nn_is_1_better, self.oracle_is_1_better):
    #         if not nn_vote and not oracle_vote:
    #             ff+=1
    #         elif nn_vote and oracle_vote:
    #             tt+=1
    #         elif not nn_vote and oracle_vote:
    #             tf+=1
    #         elif nn_vote and not oracle_vote:
    #             ft+=1
    #     total_accuracy = (tt+ff)/(tt+ff+tf+ft)
    #     # how many of those that oracle says are good nn says are good: very important, should be 1
    #     a1 = tt/(tf+tt)
    #     # how many of those that nn says good are actually good: very important, should be 1,
    #     a2 = tt/(ft+tt)
    #     # indicates that nn doesn't add useless data
    #     # how many of those that oracle says are bad nn says are bad: should be 1, indicates that
    #     a3 = ff/(ff+ft)
    #     #  nn can prune out the space efficiently
    #     # how many of those that nn says bad are actually bad: should be 1
    #     a4 = ff/(tf+ff)

    #     accuracy["total_acc"] = total_accuracy
    #     accuracy["a1"] = a1
    #     accuracy["a2"] = a2
    #     accuracy["a3"] = a3
    #     accuracy["a4"] = a4
    #     accuracy["tt"] = tt
    #     accuracy["ff"] = ff
    #     accuracy["tf"] = tf
    #     accuracy["ft"] = ft
    #     self.df_accuracy = self.df_accuracy.append(accuracy, ignore_index=True)

    #     self.logger.store_db(self.df_accuracy, fpath=os.path.join(self.eval_save_to,
    #                                                               self.file_base_name + '.pkl'))
    #     self.logger.log_text(accuracy, stream_to_stdout=False, fpath=self.acc_txt_file)
