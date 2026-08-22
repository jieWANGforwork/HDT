import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#from scipy.sparse import dok_matrix
import sys
import pickle


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


def set_device():
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def prepare_dataloader_hdt(data_path, batch_size, dataset=None, item_num = None):
    if dataset is None:
        replay_buffer = pd.read_pickle(data_path + 'train_replay_buffer_sas.df')
        #rewards_nov = np.load(data_path + 'nov_rtgs_user_train.npy', allow_pickle=True)
        #rewards_acc = np.load(data_path + 'acc_rtgs_user_train.npy', allow_pickle=True)
        #rewards_div = np.load(data_path + 'div_rtgs_user_train.npy', allow_pickle=True)
        #batch是user，仍旧是从每个user每个session的每个点击开始取，因为每个点击的state都要训练，所以仍旧要加session change,分两种情况：
        #换session时user才变或者每进行一个点击，user都变。
        #支不过输入的点击变成了一个session的长度，而不是一个一个的输入。
        #1.先用一个session的输入跑结果训练
    elif dataset == 'val':
        replay_buffer = pd.read_pickle(data_path + 'session_df_replay_buffer.df')
    elif dataset == 'test':
        replay_buffer = pd.read_pickle(data_path + 'test_df_replay_buffer.df')
    else:
        raise Exception('dataset has to be either None, val or test')

    '''data_session = {
        'userID': userID,
        'sessionID': sessionID,
        'itemsID': itemsID,
        'actionsID': actionsID,
        'is_user_done': is_user_done,

    }'''
    replay_buffer_dic = replay_buffer.to_dict()
    userID = replay_buffer_dic['userID'].values()
    userID = torch.from_numpy(np.fromiter(userID, dtype=np.int64)).long()
    #sessionID = replay_buffer_dic['sessionID'].values()
    #sessionID = torch.from_numpy(np.fromiter(sessionID, dtype=np.int64)).long()
    itemsID = replay_buffer_dic['itemsID'].values()
    itemsID = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in itemsID]).long()
    actionsID = replay_buffer_dic['actionsID'].values()
    actionsID = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in actionsID]).long()
    is_user_done = replay_buffer_dic['is_done'].values()
    is_user_done = torch.from_numpy(np.fromiter(is_user_done, dtype=np.int64)).long()

    train_data = TensorDataset(userID, itemsID, actionsID,is_user_done)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=8, batch_size=batch_size, drop_last=True)
    return train_loader


def prepare_dataloader(data_path, batch_size, item_num = 12452, dataset=None):
    if dataset is None:
        replay_buffer = pd.read_pickle(data_path + 'train_test/train_replay_buffer_sas.df')
    elif dataset == 'val':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_val.df')
    elif dataset == 'test':
        replay_buffer = pd.read_pickle(data_path + 'train_test/replay_buffer_test.df')
    else:
        raise Exception('dataset has to be either None, val or test')


    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    print(len(states))
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.int64)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.int64)).long()
    next_states = replay_buffer_dic['next_state'].values()
    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.int64)).long()
    #is_buy = replay_buffer_dic['is_buy'].values()
    #is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.int64)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool_))
    train_data = TensorDataset(states, len_states, actions, next_states,
                               len_next_states, is_done)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=8, batch_size=batch_size, drop_last=True)
    return train_loader

def prepare_dataloader_rewards(data_path, batch_size, dataset=None):
    if dataset is None:
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer.df')
        rewards_nov = np.load(data_path + 'nov_rtgs_train.npy', allow_pickle=True)
        #for i, line in enumerate(rewards_nov):
         #   rewards_nov[i] = rewards_nov[i] - rewards_nov[i, 0]
        rewards_acc = np.load(data_path + 'acc_rtgs_train.npy', allow_pickle=True)
    elif dataset == 'val':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_val.df')
    elif dataset == 'test':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_test.df')
    else:
        raise Exception('dataset has to be either None, val or test')


    # actions
    def fun(row):
        if row['len_next_states'] in [1, 10]:
            ss = row['next_state']
        else:
            ss = row['next_state'][1:]
            ss.append(70852)
        return ss
    replay_buffer['next_state'] = replay_buffer.apply(lambda x: fun(x), axis=1)
    '''for i, row  in enumerate(replay_buffer['next_state']):
        row.append(26702)
        replay_buffer.loc['next_state', i] = row[1:]'''
    # state: states   1234
    #rewards nov buffer: 0110
    # next_state: actions 2345:0110
    # len_state: 4
    # rewards acc: 4321
    # rewards nov: 2210或者2210/4321

    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.int64)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.int64)).long()
    next_states = replay_buffer_dic['next_state'].values()

    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.int64)).long()
    is_buy = replay_buffer_dic['is_buy'].values()
    is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.int64)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool_))

    acc_rtgs = torch.from_numpy(rewards_acc).float()
    acc_rtgs = acc_rtgs.unsqueeze(-1)
    nov_rtgs = torch.from_numpy(rewards_nov).float()
    nov_rtgs = nov_rtgs.unsqueeze(-1)
    #nov_rtgs = nov_rtgs/acc_rtgs

    train_data = TensorDataset(states, len_states, actions, next_states,
                               len_next_states, is_buy, is_done, acc_rtgs, nov_rtgs)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

def prepare_dataloader_mulrewards(data_path, batch_size, dataset=None, item_num = None):
    if dataset is None:
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer.df')
        rewards_nov = np.load(data_path + 'nov_rtgs_train.npy', allow_pickle=True)
        #for i, line in enumerate(rewards_nov):
         #   rewards_nov[i] = rewards_nov[i] - rewards_nov[i, 0]
        rewards_acc = np.load(data_path + 'acc_rtgs_train.npy', allow_pickle=True)
        rewards_div = np.load(data_path + 'div_rtgs_train.npy', allow_pickle=True)

    elif dataset == 'val':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_val.df')
    elif dataset == 'test':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_test.df')
    else:
        raise Exception('dataset has to be either None, val or test')


    # actions
    def fun(row):
        if row['len_next_states']==1 or(row['len_next_states']==10 and row['len_state']==10):
            ss = row['next_state']
        else:
            ss = row['next_state'][1:]
            ss.append(item_num)
        return ss
    replay_buffer['next_state'] = replay_buffer.apply(lambda x: fun(x), axis=1)
    '''for i, row  in enumerate(replay_buffer['next_state']):
        row.append(26702)
        replay_buffer.loc['next_state', i] = row[1:]'''
    # state: states   1234
    #rewards nov buffer: 0110
    # next_state: actions 2345:0110
    # len_state: 4
    # rewards acc: 4321
    # rewards nov: 2210或者2210/4321

    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.int64)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.int64)).long()
    next_states = replay_buffer_dic['next_state'].values()

    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.int64)).long()
    is_buy = replay_buffer_dic['is_buy'].values()
    is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.int64)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool_))

    acc_rtgs = torch.from_numpy(rewards_acc).float()
    acc_rtgs = acc_rtgs.unsqueeze(-1)
    nov_rtgs = torch.from_numpy(rewards_nov).float()
    nov_rtgs = nov_rtgs.unsqueeze(-1)

    div_rtgs = torch.from_numpy(rewards_div).float()
    div_rtgs = div_rtgs.unsqueeze(-1)
    #nov_rtgs = nov_rtgs/acc_rtgs

    train_data = TensorDataset(states, len_states, actions, next_states,
                               len_next_states, is_buy, is_done, acc_rtgs, nov_rtgs, div_rtgs)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=16, batch_size=batch_size, drop_last=True)
    return train_loader

def prepare_dataloader_lfm_rewards(data_path, batch_size, dataset=None, item_num = None):
    if dataset is None:
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer.df')
        rewards_nov = np.load(data_path + 'nov_rtgs_train.npy', allow_pickle=True)
        #for i, line in enumerate(rewards_nov):
         #   rewards_nov[i] = rewards_nov[i] - rewards_nov[i, 0]
        rewards_acc = np.load(data_path + 'acc_rtgs_train.npy', allow_pickle=True)
        #rewards_div = np.load(data_path + 'div_rtgs_train.npy', allow_pickle=True)

    elif dataset == 'val':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_val.df')
    elif dataset == 'test':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_test.df')
    else:
        raise Exception('dataset has to be either None, val or test')


    # actions
    def fun(row):
        if row['len_next_states'] in [1, 10]:
            ss = row['next_state']
        else:
            ss = row['next_state'][1:]
            ss.append(item_num)
        return ss
    replay_buffer['next_state'] = replay_buffer.apply(lambda x: fun(x), axis=1)
    '''for i, row  in enumerate(replay_buffer['next_state']):
        row.append(26702)
        replay_buffer.loc['next_state', i] = row[1:]'''
    # state: states   1234
    #rewards nov buffer: 0110
    # next_state: actions 2345:0110
    # len_state: 4
    # rewards acc: 4321
    # rewards nov: 2210或者2210/4321

    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.int64)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.int64)).long()
    next_states = replay_buffer_dic['next_state'].values()

    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.int64)).long()
    is_buy = replay_buffer_dic['is_buy'].values()
    is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.int64)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool_))

    acc_rtgs = torch.from_numpy(rewards_acc).float()
    acc_rtgs = acc_rtgs.unsqueeze(-1)
    nov_rtgs = torch.from_numpy(rewards_nov).float()
    nov_rtgs = nov_rtgs.unsqueeze(-1)

    #div_rtgs = torch.from_numpy(rewards_div).float()
    #div_rtgs = div_rtgs.unsqueeze(-1)
    #nov_rtgs = nov_rtgs/acc_rtgs

    train_data = TensorDataset(states, len_states, actions, next_states,
                               #len_next_states, is_buy, is_done, acc_rtgs, nov_rtgs, div_rtgs)
                                len_next_states, is_buy, is_done, acc_rtgs, nov_rtgs)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader

def prepare_dataloader_mul_rp_rewards(data_path, batch_size, dataset=None, item_num = None):
    if dataset is None:
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer.df')
        rewards_nov = np.load(data_path + 'RP_nov_rtgs_train.npy', allow_pickle=True)
        rewards_acc = np.load(data_path + 'RP_acc_rtgs_train.npy', allow_pickle=True)
        rewards_div = np.load(data_path + 'RP_div_rtgs_train.npy', allow_pickle=True)

    elif dataset == 'val':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_val.df')
    elif dataset == 'test':
        replay_buffer = pd.read_pickle(data_path + 'replay_buffer_test.df')
    else:
        raise Exception('dataset has to be either None, val or test')


    # actions
    def fun(row):
        if row['len_next_states'] in [1, 10]:
            ss = row['next_state']
        else:
            ss = row['next_state'][1:]
            ss.append(item_num)
        return ss
    #replay_buffer['next_state'] = replay_buffer.apply(lambda x: fun(x), axis=1)
    #直接用next_state数据训练

    # state: states   1234
    #rewards nov buffer: 0110
    # next_state: actions 2345:0110
    # len_state: 4
    # rewards acc: 4321
    # rewards nov: 2210或者2210/4321

    replay_buffer_dic = replay_buffer.to_dict()
    states = replay_buffer_dic['state'].values()
    states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in states]).long()
    len_states = replay_buffer_dic['len_state'].values()
    len_states = torch.from_numpy(np.fromiter(len_states, dtype=np.int64)).long()
    actions = replay_buffer_dic['action'].values()
    actions = torch.from_numpy(np.fromiter(actions, dtype=np.int64)).long()
    next_states = replay_buffer_dic['next_state'].values()

    next_states = torch.stack([torch.from_numpy(np.array(i, dtype=np.int64)) for i in next_states]).long()
    len_next_states = replay_buffer_dic['len_next_states'].values()
    len_next_states = torch.from_numpy(np.fromiter(len_next_states, dtype=np.int64)).long()
    is_buy = replay_buffer_dic['is_buy'].values()
    is_buy = torch.from_numpy(np.fromiter(is_buy, dtype=np.int64)).long()
    is_done = replay_buffer_dic['is_done'].values()
    is_done = torch.from_numpy(np.fromiter(is_done, dtype=np.bool_))

    acc_rtgs = torch.from_numpy(rewards_acc).float()
    acc_rtgs = acc_rtgs.unsqueeze(-1)
    nov_rtgs = torch.from_numpy(rewards_nov).float()
    nov_rtgs = nov_rtgs.unsqueeze(-1)

    div_rtgs = torch.from_numpy(rewards_div).float()
    div_rtgs = div_rtgs.unsqueeze(-1)
    #nov_rtgs = nov_rtgs/acc_rtgs

    train_data = TensorDataset(states, len_states, actions, next_states,
                               len_next_states, is_buy, is_done, acc_rtgs, nov_rtgs, div_rtgs)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def get_one_hot_item_sess(data_path):
    sorted_events = pd.read_csv(data_path + 'sorted_events.csv')
    item_sess_one_hot = dok_matrix(
        shape=(sorted_events.item_id.max() + 1, sorted_events.session_id.max() + 1),
        dtype=np.int32
    )
    for item_id, session_id in zip(sorted_events.item_id.values, sorted_events.session_id.values):
        item_sess_one_hot[item_id, session_id] = 1
    return item_sess_one_hot


def calculate_hit(sorted_list, topk, true_items, rewards, r_click,
                  total_reward, hit_click, ndcg_click, hit_purchase, ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                #if rewards[j] == r_click:
                hit_click[i] += 1.0
                ndcg_click[i] += 1.0 / np.log2(rank + 1)
                #else:
                #   hit_purchase[i] += 1.0
                #   ndcg_purchase[i] += 1.0 / np.log2(rank + 1)

    #return hit_click, ndcg_click

def calculate_hit_acce(sorted_list, topk, true_items, rewards, r_click,
                  total_reward, hit_click, ndcg_click, hit_purchase, ndcg_purchase):
    true_items = np.expand_dims(true_items, 1)
    rec_list20 = sorted_list[:, -20:]
    rec_list20 = rec_list20 - true_items
    pos_bz = np.where(rec_list20 == 0, 1, 0)
    for i in range(len(topk)):
        pos_bz_n = pos_bz[:, -topk[i]:]
        hit_click[i] += np.sum(pos_bz_n)
        rank = topk[i] - np.argwhere(pos_bz_n == 1)[:, 1]
        ndcg_click[i] += sum(1.0 / np.log2(rank + 1))

def calculate_hit_acce_torch(sorted_list, topk, true_items, rewards, r_click,
                  total_reward, hit_click, ndcg_click, hit_purchase, ndcg_purchase):
    #print(sorted_list.shape, true_items.shape)
    true_items = true_items.unsqueeze(dim = 1)
    #print(true_items)
    rec_list20 = sorted_list[:, -20:]
    #print(rec_list20)
    rec_list20 = rec_list20 - true_items
    #print(rec_list20)

    pos_bz = torch.where(rec_list20 == 0, 1, 0)
    #print(pos_bz)
    for i in range(len(topk)):
        pos_bz_n = pos_bz[:, -topk[i]:]
        hit_click[i] += torch.sum(pos_bz_n)
        rank = topk[i] - torch.nonzero(pos_bz_n)[:, 1]
        ndcg_click[i] += sum(1.0 / torch.log2(rank + 1))

def get_last_clicks(state, len_state, k=3):
    if len_state < k:
        return state[:len_state]
    else:
        return state[len_state - k: len_state]


def calculate_session_repetitions(session_ids, top_20_preds):
    top_10_preds = [x[-10:] for x in top_20_preds]
    top_5_preds = [x[-5:] for x in top_20_preds]
    rpt_df = pd.DataFrame({
        'session_id': session_ids,
        'top_20_preds': top_20_preds,
        'top_10_preds': top_10_preds,
        'top_5_preds': top_5_preds
    })
    num_rpts_20 = 0
    num_rpts_10 = 0
    num_rpts_5 = 0
    rpt_groups = rpt_df.groupby('session_id')
    for _, group in rpt_groups:
        all_top_20 = np.concatenate(group.top_20_preds.values)
        all_top_10 = np.concatenate(group.top_10_preds.values)
        all_top_5 = np.concatenate(group.top_5_preds.values)
        num_rpts_20 += len(all_top_20) - len(np.unique(all_top_20))
        num_rpts_10 += len(all_top_10) - len(np.unique(all_top_10))
        num_rpts_5 += len(all_top_5) - len(np.unique(all_top_5))
    num_rpts_20 /= rpt_groups.ngroups
    num_rpts_10 /= rpt_groups.ngroups
    num_rpts_5 /= rpt_groups.ngroups
    return num_rpts_5, num_rpts_10, num_rpts_20

def calculate_user_repetitions(rpt_df):

    #top_10_preds = [x[-10:] for x in top_20_preds]
    #top_5_preds = [x[-5:] for x in top_20_preds]

    '''rpt_df = pd.DataFrame({
        'user_id': user_id,
        'session_ids':session_ids,
        'top_20_preds': top_20_preds,
        'top_10_preds': top_10_preds,
        'top_5_preds': top_5_preds
    })'''

    num_rpts_20 = 0
    num_rpts_10 = 0
    num_rpts_5 = 0
    #print(rpt_df)
    rpt_groups = rpt_df.groupby('user_id')
    for _, group in rpt_groups:
        all_top_20 = np.concatenate(group.top_20_preds.values)
        all_top_10 = np.concatenate(group.top_10_preds.values)
        all_top_5 = np.concatenate(group.top_5_preds.values)
        #print(all_top_20,len(all_top_20), len(np.unique(all_top_20)))
        num_rpts_20 += (len(all_top_20)*20 - len(np.unique(all_top_20)))
        num_rpts_10 += (len(all_top_10)*10 - len(np.unique(all_top_10)))
        num_rpts_5 += (len(all_top_5)*5 - len(np.unique(all_top_5)))
    #print(rpt_groups.ngroups,num_rpts_20)
    num_rpts_20 /= rpt_groups.ngroups
    num_rpts_10 /= rpt_groups.ngroups
    num_rpts_5 /= rpt_groups.ngroups
    return num_rpts_5, num_rpts_10, num_rpts_20

def calculate_user_unique(rpt_df):
    #top_10_preds = [x[-10:] for x in top_20_preds]
    #top_5_preds = [x[-5:] for x in top_20_preds]
    ''' rpt_df = pd.DataFrame({
        'session_id': session_ids,
        'top_20_preds': top_20_preds, #这个user session的所有top preds
        'top_10_preds': top_10_preds,
        'top_5_preds': top_5_preds
    })'''
    num_rpts_20 = 0
    num_rpts_10 = 0
    num_rpts_5 = 0
    rpt_groups = rpt_df.groupby('user_id')
    for _, group in rpt_groups:
        #print(group)
        all_top_20 = np.concatenate(group.top_20_preds.values)
        all_top_10 = np.concatenate(group.top_10_preds.values)
        all_top_5 = np.concatenate(group.top_5_preds.values)
        #print(all_top_20,len(all_top_20), len(np.unique(all_top_20)))

        num_rpts_20 += (len(np.unique(all_top_20)) / len(all_top_20)/20)
        num_rpts_10 += (len(np.unique(all_top_10)) /len(all_top_10)/10)
        num_rpts_5 += (len(np.unique(all_top_5)) /len(all_top_5)/5)
    num_rpts_20 /= rpt_groups.ngroups
    num_rpts_10 /= rpt_groups.ngroups
    num_rpts_5 /= rpt_groups.ngroups
    return num_rpts_5, num_rpts_10, num_rpts_20

def calculate_user_cover_unexp(rpt_df):
    '''top_10_preds = [x[-10:] for x in top_20_preds]
    top_5_preds = [x[-5:] for x in top_20_preds]
    rpt_df = pd.DataFrame({
        'session_id': session_ids,
        'top_20_preds': top_20_preds, #这个user session的所有top preds
        'top_10_preds': top_10_preds,
        'top_5_preds': top_5_preds,
        'all_pre': all_pre
    })'''
    num_rpts_20 = 0
    num_rpts_10 = 0
    num_rpts_5 = 0
    rpt_groups = rpt_df.groupby('user_id')
    for _, group in rpt_groups:
        all_top_20 = np.concatenate(group.top_20_preds.values)
        all_top_10 = np.concatenate(group.top_10_preds.values)
        all_top_5 = np.concatenate(group.top_5_preds.values)
        all_pre = np.concatenate(group.session_ids.values)
        #print(all_top_5)
        num_rpts_20 += len(np.intersect1d(np.unique(all_top_20), np.unique(all_pre)))
        #len(np.setdiff1d(np.unique(all_top_20), np.unique(all_pre)))
        num_rpts_10 += len(np.intersect1d(np.unique(all_top_10), np.unique(all_pre)))
        num_rpts_5 +=  len(np.intersect1d(np.unique(all_top_5), np.unique(all_pre)))
    num_rpts_20 /= rpt_groups.ngroups
    num_rpts_10 /= rpt_groups.ngroups
    num_rpts_5 /= rpt_groups.ngroups
    return num_rpts_5, num_rpts_10, num_rpts_20

def eval_cov_nov_cov(all_top_20, item_num, data_path):
    all_top_1 = [x[-1] for x in all_top_20]
    all_top_5 = [x[-5:] for x in all_top_20]
    all_top_10 = [x[-10:] for x in all_top_20]
    # Evaluate coverage of all items
    all_top_1_concated = np.array(all_top_1)
    cov1 = len(np.unique(all_top_1_concated)) / item_num
    all_top_5_concated = np.concatenate(all_top_5)
    cov5 = len(np.unique(all_top_5_concated)) / item_num
    all_top_10_concated = np.concatenate(all_top_10)
    cov10 = len(np.unique(all_top_10_concated)) / item_num
    all_top_20_concated = np.concatenate(all_top_20)
    cov20 = len(np.unique(all_top_20_concated)) / item_num

    # Evaluate coverage of novel items
    less_popular_items = np.load(data_path + 'train_test/less_items.npy')
    less_pop_num = len(less_popular_items)
    novel_top_1_concated = np.intersect1d(all_top_1_concated, less_popular_items)
    nov_cov1 = len(novel_top_1_concated) / less_pop_num
    novel_top_5_concated = np.intersect1d(all_top_5_concated, less_popular_items)
    nov_cov5 = len(novel_top_5_concated) / less_pop_num
    novel_top_10_concated = np.intersect1d(all_top_10_concated, less_popular_items)
    nov_cov10 = len(novel_top_10_concated) / less_pop_num
    novel_top_20_concated = np.intersect1d(all_top_20_concated, less_popular_items)
    nov_cov20 = len(novel_top_20_concated) / less_pop_num
    return cov1, cov5, cov10, cov20, nov_cov1, nov_cov5, nov_cov10, nov_cov20


def calculate_div_reward(states, len_states, preds, div_rl_embedding, device):
    total_reward = 0
    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in range(len(states)):
        last_click = states[i][len_states[i] - 1].to(device)
        last_click_emb = div_rl_embedding(last_click)
        top_pred = preds[i].to(device)
        top_pred_emb = div_rl_embedding(top_pred)
        current_reward = 1 - cos_sim(last_click_emb, top_pred_emb)
        total_reward += current_reward
    return total_reward


def calculate_total_nov_reward(top_20, data_path):

    less_popular_items = np.load(data_path + 'less_items.npy')
    is_less_popular_20 = np.in1d(top_20, less_popular_items).reshape(-1, 20)
    is_less_popular_1 = is_less_popular_20[:, -1]
    total_nov_reward = is_less_popular_1.sum()
    return total_nov_reward


def initialize_rel_disc_matrix(size, disc_factor=0.85):
    rel_disc_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            rel_disc_matrix[i, j] = disc_factor**i * disc_factor**(np.max([1, j-i]))
    return torch.tensor(rel_disc_matrix)


def get_novelty_reward_dict(data_path):
    nov_rewards_csv = pd.read_csv(data_path + 'binary_nov_reward.csv', header=None, index_col=0, squeeze=True)
    nov_rewards_dict = nov_rewards_csv.to_dict()
    print('percentage of positive reward: ', len(nov_rewards_csv[nov_rewards_csv == 1]) / len(nov_rewards_csv))
    return nov_rewards_dict


def set_stdout(results_path, file_name, write_to_file=True):
    if write_to_file:
        print('Outputs are saved to file {}'.format(results_path + file_name))
        sys.stdout = open(results_path + file_name, 'w')
    else:
        pass


def get_stats(data_path):
    data_statis = pd.read_pickle(data_path + 'data_statis.df')  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    return state_size, item_num

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    #print(X.shape)
    return X

def calculate_div_acce_torch(sorted_list, topk, session_id, item_embeddings, div, item_num,
                             whole_item_embs, action):
    #previous session; 1-average cos;
    #whole_item_embs = l2norm(whole_item_embs, dim=-1)
    #whole_item_embs = torch.mean(whole_item_embs, dim=0).unsqueeze(0)
    #print(whole_item_embs.shape)

    action_id_embs = item_embeddings(action)
    action_id_embs = l2norm(action_id_embs, dim=-1)#.unsqueeze(-1) #bz, d, 1



    rec_list20 = sorted_list[:, -20:]
    session_id_embs = item_embeddings(session_id)
    session_id_embs = l2norm(session_id_embs, dim=-1)#.unsqueeze(-1) #bz, d, 1
    #print(session_id_embs.shape)
    mask = torch.ne(session_id, item_num).float().unsqueeze(-1)
    session_id_embs *= mask
    len_states = mask.sum(dim=1)#.squeeze(1).long()
    session_id_mean_embs = torch.sum(session_id_embs, dim=1)
    #print(session_id_mean_embs.shape, len_states.shape)
    session_id_mean_embs = session_id_mean_embs/len_states #bz, d
    #print(session_id_mean_embs.shape)
    rec_list20_embs = item_embeddings(rec_list20) #bz, 20, d -> bz, 20, 1
    rec_list20_embs = l2norm(rec_list20_embs, dim=-1)
    #print(rec_list20_embs.shape)
    rec_sims_whole = torch.mm(session_id_mean_embs, #bz, d * d, 1 -> bz, 1
                               whole_item_embs.transpose(0, 1))#.repeat(-1, 20) #bz, 20
    #print(rec_sims_whole.shape)
    #session_id_mean_embs = l2norm(session_id_mean_embs, dim=-1).unsqueeze(-1) #bz, d, 1
    #print(session_id_mean_embs.shape)
    rec_sims = torch.bmm(session_id_mean_embs.unsqueeze(1), rec_list20_embs.transpose(1, 2)).squeeze(1) #bz, 1, d * bz, d, 20->bz, 20
    #print(rec_sims.shape)
    #print(rec_sims, rec_sims_whole)

    rec_sims_true = torch.bmm(session_id_mean_embs.unsqueeze(1), action_id_embs.unsqueeze(-1)  # bz, 1, d * bz, d, 1 -> bz, 1
                            )
    rec_sims = (rec_sims_true - rec_sims)/(1-rec_sims_whole)#/( rec_sims_whole) * 100.0
    #print(rec_sims)

    for i in range(len(topk)):
        rec_sims_n = rec_sims[:, -topk[i]:]
        div[i] += torch.sum(rec_sims_n)
def calculate_unexp_acce_torch(sorted_list, topk, session_id, item_embeddings, div, item_num,
                             whole_item_embs, action):
    #previous session; 1-average cos;
    #whole_item_embs = l2norm(whole_item_embs, dim=-1)
    #whole_item_embs = torch.mean(whole_item_embs, dim=0).unsqueeze(0)
    #print(whole_item_embs.shape)

    action_id_embs = item_embeddings(action)
    action_id_embs = l2norm(action_id_embs, dim=-1)#.unsqueeze(-1) #bz, d, 1

    rec_list20 = sorted_list[:, -20:]
    session_id_embs = item_embeddings(session_id)
    session_id_embs = l2norm(session_id_embs, dim=-1)#.unsqueeze(-1) #bz, d, 1
    #print(session_id_embs.shape)
    mask = torch.ne(session_id, item_num).float().unsqueeze(-1)
    session_id_embs *= mask
    len_states = mask.sum(dim=1)#.squeeze(1).long()
    session_id_mean_embs = torch.sum(session_id_embs, dim=1)
    #print(session_id_mean_embs.shape, len_states.shape)
    session_id_mean_embs = session_id_mean_embs/len_states #bz, d
    #print(session_id_mean_embs.shape)
    rec_list20_embs = item_embeddings(rec_list20) #bz, 20, d -> bz, 20, 1
    rec_list20_embs = l2norm(rec_list20_embs, dim=-1)
    #print(rec_list20_embs.shape)
    rec_sims_whole = torch.mm(session_id_mean_embs, #bz, d * d, 1 -> bz, 1
                               whole_item_embs.transpose(0, 1))#.repeat(-1, 20) #bz, 1
    #print(rec_sims_whole.shape, whole_item_embs.shape,rec_sims_whole)
    #session_id_mean_embs = l2norm(session_id_mean_embs, dim=-1).unsqueeze(-1) #bz, d, 1
    #print(session_id_mean_embs.shape)
    rec_sims = torch.bmm(session_id_mean_embs.unsqueeze(1), rec_list20_embs.transpose(1, 2)).squeeze(1) #bz, 1, d * bz, d, 20->bz, 20
    rec_sims_true = torch.bmm(session_id_mean_embs.unsqueeze(1), action_id_embs.unsqueeze(-1)) .squeeze(1) # bz, 1, d * bz, d, 1 -> bz, 1

    #rec_sims_ = torch.le(rec_sims, rec_sims_true).float()
    rec_sims_ = torch.le(rec_sims, rec_sims_whole).float()

    #print(rec_sims.shape, rec_sims_true.shape, rec_sims_.shape, rec_sims_whole.shape)
    #print(rec_sims_)


    for i in range(len(topk)):
        rec_sims_n = rec_sims_[:, -topk[i]:]
        #print(rec_sims_n.shape, torch.sum(rec_sims_n)/topk[i])
        rec_sims_n = torch.sum(rec_sims_n, dim=1)
        rec_sims_n1 = torch.ones(rec_sims_n.shape).to(rec_sims_n.device)
        rec_sims_n = torch.where(rec_sims_n>0, rec_sims_n1, rec_sims_n)
        #print(rec_sims_n)
        div[i] += torch.sum(rec_sims_n)#/topk[i])
        #rec_sims_n = 1-torch.mean(rec_sims_n, dim=1)
        #div[i] += torch.sum(rec_sims_n)
