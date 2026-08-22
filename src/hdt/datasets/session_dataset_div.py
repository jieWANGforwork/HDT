import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
from pandas import DataFrame


def _read_replay_buffer(args, split: str, default_relative_path: str):
    attr = {"train": "train_file", "val": "valid_file", "test": "test_file"}[split]
    relative_path = getattr(args, attr, None) or default_relative_path
    path = Path(args.data_path) / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing {split} replay buffer: {path}")
    return pd.read_pickle(path)


class Transform(ABC):
    @abstractmethod
    def __call__(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        pass


class DenseIndexing(Transform):
    def __init__(self, item_map_df: DataFrame):
        self.item_map: Dict[Any, int] = self.refine_map_from(item_map_df)
        self.indexer = np.vectorize(self.item_map.get)

    @staticmethod
    def refine_map_from(item_map: DataFrame) -> Dict[int, int]:
        return {Id: idx for Id, idx in item_map[["item_id", "item_idx"]].values}

    def __call__(self, sample):
        inputs, targets = sample["inputs"], sample["targets"]

        sample.update({
            "inputs": self.indexer(inputs),
            "targets": self.indexer(targets),
        })
        return sample


class Masking(Transform):
    batch_size: int

    def get_mask(self, indices):
        mask = np.zeros(shape=(self.batch_size, 1))
        mask[indices, :] = 1.0
        return mask

    def __call__(self, sample):
        self.batch_size = len(sample["inputs"])

        session_change_idx = sample["session_change"]
        user_change_idx = sample['user_change']
        poss_reward_idx = sample["poss_reward"]
        sess_reward_idx = sample["sess_reward"]

        sample.update({
            "session_change": self.get_mask(session_change_idx),
            "user_change": self.get_mask(user_change_idx),
            "poss_reward": self.get_mask(poss_reward_idx),
            "soss_reward": self.get_mask(sess_reward_idx),

        })
        return sample


class MaskingReward(Transform):
    batch_size: int

    def get_mask(self, indices):
        mask = np.zeros(shape=(self.batch_size, 1))
        mask[indices, :] = 1.0
        return mask

    def __call__(self, sample):
        self.batch_size = len(sample["inputs"])

        session_change_idx = sample["session_change"]
        user_change_idx = sample['user_change']
        poss_reward_idx = sample["poss_reward"]
        sess_reward_idx = sample["sess_reward"]

        sample.update({
            "session_change": self.get_mask(session_change_idx),
            "user_change": self.get_mask(user_change_idx),
            "poss_reward": self.get_mask(poss_reward_idx),
            "sess_reward": self.get_mask(sess_reward_idx),

        })
        return sample


class ToTensor(Transform):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return {
            "inputs": torch.LongTensor(sample["inputs"]),
            "targets": torch.LongTensor(sample["targets"]),
            "user_change": torch.FloatTensor(sample["user_change"]),
            "reward": torch.FloatTensor(sample["user_change"]),

        }


class ToTensorReward(Transform):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return {
            "inputs": torch.LongTensor(sample["inputs"]),
            "targets": torch.LongTensor(sample["targets"]),
            "user_change": torch.FloatTensor(sample["user_change"]),
            "poss_reward": torch.FloatTensor(sample["poss_reward"]),
            "sess_reward": torch.FloatTensor(sample["sess_reward"]),
            "poss_rtg": torch.FloatTensor(sample["poss_rtg"]),

        }


class TestToTensor(Transform):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        '''return {
            "inputs": torch.LongTensor(sample["inputs"]).to(self.device),
            "targets": torch.LongTensor(sample["targets"]).to(self.device),
            "user_change": torch.FloatTensor(sample["user_change"]).to(self.device),
            "reward": torch.FloatTensor(sample["user_change"]),
            "is_eval": torch.BoolTensor(sample["is_eval"]).to(self.device),

        }'''
        return {
            "inputs": torch.LongTensor(sample["inputs"]),
            "targets": torch.LongTensor(sample["targets"]),
            "user_change": torch.FloatTensor(sample["user_change"]),
            "reward": torch.FloatTensor(sample["user_change"]),
            "is_eval": torch.BoolTensor(sample["is_eval"]),

        }


class TestToTensorReward(Transform):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        '''return {
            "inputs": torch.LongTensor(sample["inputs"]).to(self.device),
            "targets": torch.LongTensor(sample["targets"]).to(self.device),
            "user_change": torch.FloatTensor(sample["user_change"]).to(self.device),
            "reward": torch.FloatTensor(sample["user_change"]),
            "is_eval": torch.BoolTensor(sample["is_eval"]).to(self.device),

        }'''
        return {
            "inputs": torch.LongTensor(sample["inputs"]),
            "targets": torch.LongTensor(sample["targets"]),
            "user_change": torch.FloatTensor(sample["user_change"]),
            "poss_reward": torch.FloatTensor(sample["poss_reward"]),
            "sess_reward": torch.FloatTensor(sample["sess_reward"]),
            "poss_rtg": torch.FloatTensor(sample["poss_rtg"]),
            "is_eval": torch.BoolTensor(sample["is_eval"]),

        }


class Sampler(object):
    def __init__(self, df, n_samples, sample_alpha=0.75, sample_store=10000):
        self.df = df
        self.n_samples = n_samples
        self.sample_alpha = sample_alpha  # 3/4
        self.sample_store = sample_store
        self.generate_size = sample_store // n_samples

        self.neg_samples = self._generate_neg_samples(self.generate_size)
        self.sample_pointer = -1

    def __next__(self):
        self.sample_pointer += 1

        if self.sample_pointer == self.generate_size:
            self.neg_samples = self._generate_neg_samples(self.generate_size)
            self.sample_pointer = 0

        return self.neg_samples[self.sample_pointer]

    def __iter__(self):
        return self

    def _init_sample(self):
        self.neg_samples = self._generate_neg_samples(self.generate_size)
        self.sample_pointer = 0

    def _generate_neg_samples(self, length):
        n_item = self._pop.size
        sample_size = length * self.n_samples

        if self.sample_alpha > 0:
            draw = np.random.rand(sample_size)
            sample = np.searchsorted(self._pop, draw)
        else:
            sample = np.random.choice(n_item, size=sample_size)

        if length > 1:
            sample = sample.reshape((length, self.n_samples))
        return sample

    @property
    def _pop(self):
        """ Calculate the distribution P(w_i) of negative sampling """
        prob = self.df["item_id"].value_counts() ** self.sample_alpha
        return prob.cumsum() / prob.sum()


class DataLoader(IterableDataset):
    def __init__(self, args, type=None, transforms: Optional[List[Transform]] = None):
        if type is None:
            replay_buffer = pd.read_pickle(args.data_path + 'train_session_df_replay_buffer.df')

        elif type == 'val':
            replay_buffer = pd.read_pickle(args.data_path + 'valid_session_df_replay_buffer.df')
        elif type == 'test':
            replay_buffer = pd.read_pickle(args.data_path + 'test_session_df_replay_buffer.df')
        else:
            raise Exception('dataset has to be either None, val or test')

        self.num_session_for_user = np.r_[
            0, replay_buffer.groupby("userID", sort=False)["sessionID"].nunique().cumsum().values]
        self.df = replay_buffer.sort_values(by=["userID", 'sessionID'])
        # print(self.df)
        self.batch_size = args.batch_size
        # self.neg_sampler = Sampler(df, args.n_samples)
        self.transforms = transforms
        self.users = self.df["userID"].unique()
        self.user_idx_arr = np.arange(len(self.users))
        self.session_num = max(self.num_session_for_user)
        self.session_idx_arr = np.arange(self.session_num)

    def __iter__(self):
        # session_offset：所有session的开始点击位置
        # user offset：所有user的开始点击位置
        # num_session_for_user：所有user的session数量及其位置
        # user_iter：该位置迭代到的user顺序
        # session_iter：该位置迭代到的session顺序
        # print( self.df["itemsID"])
        itemsIDs = self.df["itemsID"].values
        itemsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in itemsIDs]).long()
        # print(itemsIDs)
        actionsIDs = self.df['actionsID'].values
        actionsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in actionsIDs]).long()
        batch_size = min(self.batch_size, len(self.users))

        user_iter = np.arange(batch_size)
        max_user_iter = np.max(user_iter)

        user_session_start = self.num_session_for_user[user_iter]
        user_session_end = self.num_session_for_user[user_iter + 1]
        user_change = []
        reward = []
        finished = False
        # print(user_session_start, user_session_end)
        while not finished:
            min_session_length = np.min(user_session_end - user_session_start)  # 最小session长度

            for i in range(min_session_length):
                inputs = itemsIDs[user_session_start + i]  # df已经按顺序排过了
                targets = actionsIDs[user_session_start + i]
                # print(inputs)
                # user_change = self.df["is_user_done"].values[user_session_start+i]
                user_mask = np.zeros(shape=(self.batch_size, 1))
                user_mask[user_change, :] = 1.0
                user_change = user_mask
                reward = np.ones(shape=(self.batch_size, 1))
                samples = {
                    "inputs": inputs,
                    "targets": targets,
                    "user_change": user_change,
                    'reward': reward
                }
                if self.transforms is not None:
                    for transform in self.transforms:
                        samples = transform(samples)  # only to tensor

                yield samples
                user_change = []

            user_session_start += min_session_length
            user_change = np.arange(batch_size)[
                user_session_end - user_session_start <= 0]  # session change是1，取出change的user位置
            # user_change标识是在下一个user开始的位置
            for idx in user_change:
                if max_user_iter + 1 >= len(self.users):
                    finished = True
                    continue  # min_session_length=0，结束循环
                else:
                    max_user_iter += 1
                    # user change的移动到下一个user
                    user_iter[idx] = max_user_iter
                    user_session_start[idx] = self.num_session_for_user[max_user_iter]
                    user_session_end[idx] = self.num_session_for_user[max_user_iter + 1]


class TestDataLoader(IterableDataset):
    def __init__(self, args, type='test', transforms: Optional[List[Transform]] = None):
        if type is None:
            replay_buffer = pd.read_pickle(args.data_path + 'session_df_replay_buffer.df')
        elif type == 'val':
            replay_buffer = pd.read_pickle(args.data_path + 'valid_session_df_replay_buffer.df')
        elif type == 'test':
            replay_buffer = pd.read_pickle(args.data_path + 'test_session_df_replay_buffer.df')
        else:
            raise Exception('dataset has to be either None, val or test')

        self.num_session_for_user = np.r_[0, replay_buffer.groupby("userID", sort=False).size().cumsum().values]
        self.df = replay_buffer.sort_values(by=["userID", 'sessionID'])
        self.batch_size = args.batch_size
        # self.neg_sampler = Sampler(df, args.n_samples)
        self.transforms = transforms
        self.users = self.df["userID"].unique()
        self.user_idx_arr = np.arange(len(self.users))
        self.session_num = max(self.num_session_for_user)
        self.session_idx_arr = np.arange(self.session_num)

    def __iter__(self):
        itemsIDs = self.df["itemsID"].values
        itemsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in itemsIDs]).long()
        # print(itemsIDs)
        actionsIDs = self.df['actionsID'].values
        actionsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in actionsIDs]).long()
        is_eval = self.df['is_eval'].values
        # is_eval = torch.from_numpy(np.fromiter(is_eval, dtype=np.int64)).long()
        batch_size = min(self.batch_size, len(self.users))

        user_iter = np.arange(batch_size)
        max_user_iter = np.max(user_iter)
        # 10,5,4
        # 0, 10, 15
        # 10, 15, 19
        # 第一轮：ls=4
        # 0123,10111213,15161718
        # 4,14,19
        # 10, 15,19
        user_session_start = self.num_session_for_user[user_iter]
        user_session_end = self.num_session_for_user[user_iter + 1]
        user_change = []
        eval = []
        reward = []
        finished = False
        # print(user_session_start, user_session_end)
        while not finished:
            min_session_length = np.min(user_session_end - user_session_start)  # 最小session长度

            for i in range(min_session_length):
                inputs = itemsIDs[user_session_start + i]  # df已经按顺序排过了
                targets = actionsIDs[user_session_start + i]
                eval = is_eval[user_session_start + i]

                user_mask = np.zeros(shape=(self.batch_size, 1))
                user_mask[user_change, :] = 1.0
                user_change = user_mask
                reward = np.ones(shape=(self.batch_size, 1))
                samples = {
                    "inputs": inputs,
                    "targets": targets,
                    "user_change": user_change,
                    'reward': reward,
                    'is_eval': eval
                }
                if self.transforms is not None:
                    for transform in self.transforms:
                        samples = transform(samples)  # only to tensor

                yield samples
                user_change = []

            user_session_start += min_session_length
            user_change = np.arange(batch_size)[
                user_session_end - user_session_start <= 0]  # session change是1，取出change的user位置

            for idx in user_change:
                if max_user_iter + 1 >= len(self.users):
                    finished = True
                    continue  # min_session_length=0，结束循环
                else:
                    max_user_iter += 1
                    # user change的移动到下一个user
                    user_iter[idx] = max_user_iter
                    user_session_start[idx] = self.num_session_for_user[max_user_iter]
                    user_session_end[idx] = self.num_session_for_user[max_user_iter + 1]


class DataLoaderClick(IterableDataset):
    def __init__(self, args, type=None, transforms: Optional[List[Transform]] = None):
        if type is None:
            # replay_buffer = pd.read_pickle(args.data_path + 'train_session_df_replay_buffer.df')
            replay_buffer = pd.read_pickle(args.data_path + 'train_click_df_replay_buffer.df')

        elif type == 'val':
            replay_buffer = pd.read_pickle(args.data_path + 'valid_session_df_replay_buffer.df')
        elif type == 'test':
            replay_buffer = pd.read_pickle(args.data_path + 'test_df_replay_buffer.df')
        else:
            raise Exception('dataset has to be either None, val or test')

        self.num_session_for_user = np.r_[0, replay_buffer.groupby("userID_", sort=False)["sessionID_"].cumsum().values]
        self.df = replay_buffer.sort_values(by=["userID_", 'sessionID_'])
        # print(self.df)
        self.batch_size = args.batch_size
        # self.neg_sampler = Sampler(df, args.n_samples)
        self.transforms = transforms
        self.users = self.df["userID_"].unique()
        self.user_idx_arr = np.arange(len(self.users))
        self.session_num = max(self.num_session_for_user)
        self.session_idx_arr = np.arange(self.session_num)

    def __iter__(self):
        # session_offset：所有session的开始点击位置
        # user offset：所有user的开始点击位置
        # num_session_for_user：所有user的session数量及其位置
        # user_iter：该位置迭代到的user顺序
        # session_iter：该位置迭代到的session顺序
        # print( self.df["itemsID"])
        itemsIDs = self.df["itemsID_"].values
        itemsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in itemsIDs]).long()
        # print(itemsIDs)
        actionsIDs = self.df['actionsID_'].values
        actionsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in actionsIDs]).long()
        batch_size = min(self.batch_size, len(self.users))

        user_iter = np.arange(batch_size)
        max_user_iter = np.max(user_iter)

        user_session_start = self.num_session_for_user[user_iter]
        user_session_end = self.num_session_for_user[user_iter + 1]
        user_change = []
        reward = []
        finished = False
        # print(user_session_start, user_session_end)
        while not finished:
            min_session_length = np.min(user_session_end - user_session_start)  # 最小session长度

            for i in range(min_session_length):
                inputs = itemsIDs[user_session_start + i]  # df已经按顺序排过了
                targets = actionsIDs[user_session_start + i]
                # print(inputs)
                # user_change = self.df["is_user_done"].values[user_session_start+i]
                user_mask = np.zeros(shape=(self.batch_size, 1))
                user_mask[user_change, :] = 1.0
                user_change = user_mask
                reward = np.ones(shape=(self.batch_size, 1))
                samples = {
                    "inputs": inputs,
                    "targets": targets,
                    "user_change": user_change,
                    'reward': reward
                }
                if self.transforms is not None:
                    for transform in self.transforms:
                        samples = transform(samples)  # only to tensor

                yield samples
                user_change = []

            user_session_start += min_session_length
            user_change = np.arange(batch_size)[
                user_session_end - user_session_start <= 0]  # session change是1，取出change的user位置

            for idx in user_change:
                if max_user_iter + 1 >= len(self.users):
                    finished = True
                    continue  # min_session_length=0，结束循环
                else:
                    max_user_iter += 1
                    # user change的移动到下一个user
                    user_iter[idx] = max_user_iter
                    user_session_start[idx] = self.num_session_for_user[max_user_iter]
                    user_session_end[idx] = self.num_session_for_user[max_user_iter + 1]


class DataLoaderUDT(IterableDataset):
    def __init__(self, args, type=None, transforms: Optional[List[Transform]] = None):
        if type is None:
            replay_buffer = _read_replay_buffer(args, "train", "train_test/train_session_df_replay_buffer_nov2_user.df")

        elif type == 'val':
            replay_buffer = _read_replay_buffer(args, "val", "valid_session_df_replay_buffer.df")
        elif type == 'test':
            replay_buffer = _read_replay_buffer(args, "test", "test_session_df_replay_buffer.df")
        else:
            raise Exception('dataset has to be either None, val or test')

        self.num_session_for_user = np.r_[
            0, replay_buffer.groupby("userID", sort=False)["sessionID"].nunique().cumsum().values]
        self.df = replay_buffer.sort_values(by=["userID", 'sessionID'])
        # print(self.df)
        self.batch_size = args.batch_size
        # self.neg_sampler = Sampler(df, args.n_samples)
        self.transforms = transforms
        self.users = self.df["userID"].unique()
        self.user_idx_arr = np.arange(len(self.users))
        self.session_num = max(self.num_session_for_user)
        self.session_idx_arr = np.arange(self.session_num)

    def __iter__(self):
        # session_offset：所有session的开始点击位置
        # user offset：所有user的开始点击位置
        # num_session_for_user：所有user的session数量及其位置
        # user_iter：该位置迭代到的user顺序
        # session_iter：该位置迭代到的session顺序
        # print( self.df["itemsID"])
        itemsIDs = self.df["itemsID"].values
        itemsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in itemsIDs]).long()
        # print(itemsIDs)
        actionsIDs = self.df['actionsID'].values
        actionsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in actionsIDs]).long()

        # sessRewards = self.df['nov_sess'].values
        sessRewards = self.df['user_div_rtgs'].values

        sessRewards = torch.from_numpy(np.fromiter(sessRewards, dtype=np.float64)).float()

        possRewards = self.df['div_scores'].values
        possRewards = torch.stack([torch.from_numpy(np.array(i, dtype=np.float64)) for i in possRewards]).float()

        possRtgs = self.df['div_rtgs'].values
        possRtgs = torch.stack([torch.from_numpy(np.array(i, dtype=np.float64)) for i in possRtgs]).float()

        batch_size = min(self.batch_size, len(self.users))

        user_iter = np.arange(batch_size)
        max_user_iter = np.max(user_iter)

        user_session_start = self.num_session_for_user[user_iter]
        user_session_end = self.num_session_for_user[user_iter + 1]
        user_change = []
        reward = []
        finished = False
        # print(user_session_start, user_session_end)
        while not finished:
            min_session_length = np.min(user_session_end - user_session_start)  # 最小session长度

            for i in range(min_session_length):
                inputs = itemsIDs[user_session_start + i]  # df已经按顺序排过了
                targets = actionsIDs[user_session_start + i]
                sess_reward = sessRewards[user_session_start + i]
                poss_reward = possRewards[user_session_start + i]
                poss_rtg = possRtgs[user_session_start + i]

                # print(inputs)
                # user_change = self.df["is_user_done"].values[user_session_start+i]
                user_mask = np.zeros(shape=(self.batch_size, 1))
                user_mask[user_change, :] = 1.0
                user_change = user_mask
                samples = {
                    "inputs": inputs,
                    "targets": targets,
                    "user_change": user_change,
                    'sess_reward': sess_reward,
                    'poss_reward': poss_reward,
                    'poss_rtg': poss_rtg
                }
                if self.transforms is not None:
                    for transform in self.transforms:
                        samples = transform(samples)  # only to tensor

                yield samples
                user_change = []

            user_session_start += min_session_length
            user_change = np.arange(batch_size)[
                user_session_end - user_session_start <= 0]  # session change是1，取出change的user位置
            # user_change标识是在下一个user开始的位置
            for idx in user_change:
                if max_user_iter + 1 >= len(self.users):
                    finished = True
                    continue  # min_session_length=0，结束循环
                else:
                    max_user_iter += 1
                    # user change的移动到下一个user
                    user_iter[idx] = max_user_iter
                    user_session_start[idx] = self.num_session_for_user[max_user_iter]
                    user_session_end[idx] = self.num_session_for_user[max_user_iter + 1]


class TestDataLoaderUDT(IterableDataset):
    def __init__(self, args, type='test', transforms: Optional[List[Transform]] = None):
        if type is None:
            replay_buffer = _read_replay_buffer(args, "train", "session_df_replay_buffer_nov_users.df")
        elif type == 'val':
            replay_buffer = _read_replay_buffer(args, "val", "valid_session_df_replay_buffer.df")
        elif type == 'test':
            replay_buffer = _read_replay_buffer(args, "test", "train_test/test_session_df_replay_buffer_nov2_users_h.df")
        else:
            raise Exception('dataset has to be either None, val or test')

        self.num_session_for_user = np.r_[0, replay_buffer.groupby("userID", sort=False).size().cumsum().values]
        self.df = replay_buffer.sort_values(by=["userID", 'sessionID'])
        self.batch_size = args.batch_size
        # self.neg_sampler = Sampler(df, args.n_samples)
        self.transforms = transforms
        self.users = self.df["userID"].unique()
        self.user_idx_arr = np.arange(len(self.users))
        self.session_num = max(self.num_session_for_user)
        self.session_idx_arr = np.arange(self.session_num)

    def __iter__(self):
        itemsIDs = self.df["itemsID"].values
        itemsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in itemsIDs]).long()
        # print(itemsIDs)
        actionsIDs = self.df['actionsID'].values
        actionsIDs = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in actionsIDs]).long()

        # sessRewards = self.df['nov_sess'].values
        sessRewards = self.df['user_div_rtgs'].values

        sessRewards = torch.from_numpy(np.fromiter(sessRewards, dtype=np.float64)).float()

        possRewards = self.df['div_scores'].values
        possRewards = torch.stack([torch.from_numpy(np.array(i, dtype=np.float64)) for i in possRewards]).float()

        possRtgs = self.df['div_rtgs'].values
        possRtgs = torch.stack([torch.from_numpy(np.array(i, dtype=np.float64)) for i in possRtgs]).float()

        is_eval = self.df['is_eval'].values
        # is_eval = torch.from_numpy(np.fromiter(is_eval, dtype=np.int64)).long()
        batch_size = min(self.batch_size, len(self.users))

        user_iter = np.arange(batch_size)
        max_user_iter = np.max(user_iter)
        # 10,5,4
        # 0, 10, 15
        # 10, 15, 19
        # 第一轮：ls=4
        # 0123,10111213,15161718
        # 4,14,19
        # 10, 15,19
        user_session_start = self.num_session_for_user[user_iter]
        user_session_end = self.num_session_for_user[user_iter + 1]
        user_change = []
        eval = []
        reward = []
        finished = False
        # print(user_session_start, user_session_end)
        while not finished:
            min_session_length = np.min(user_session_end - user_session_start)  # 最小session长度

            for i in range(min_session_length):
                inputs = itemsIDs[user_session_start + i]  # df已经按顺序排过了
                targets = actionsIDs[user_session_start + i]
                sess_reward = sessRewards[user_session_start + i]
                poss_reward = possRewards[user_session_start + i]
                poss_rtg = possRtgs[user_session_start + i]

                eval = is_eval[user_session_start + i]

                user_mask = np.zeros(shape=(self.batch_size, 1))
                user_mask[user_change, :] = 1.0
                user_change = user_mask
                samples = {
                    "inputs": inputs,
                    "targets": targets,
                    "user_change": user_change,
                    'sess_reward': sess_reward,
                    'poss_reward': poss_reward,
                    'poss_rtg': poss_rtg,
                    'is_eval': eval
                }
                if self.transforms is not None:
                    for transform in self.transforms:
                        samples = transform(samples)  # only to tensor

                yield samples
                user_change = []

            user_session_start += min_session_length
            user_change = np.arange(batch_size)[
                user_session_end - user_session_start <= 0]  # session change是1，取出change的user位置

            for idx in user_change:
                if max_user_iter + 1 >= len(self.users):
                    finished = True
                    continue  # min_session_length=0，结束循环
                else:
                    max_user_iter += 1
                    # user change的移动到下一个user
                    user_iter[idx] = max_user_iter
                    user_session_start[idx] = self.num_session_for_user[max_user_iter]
                    user_session_end[idx] = self.num_session_for_user[max_user_iter + 1]
