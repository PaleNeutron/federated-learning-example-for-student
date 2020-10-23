import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.preprocessing._data import StandardScaler

TRAINDATA_DIR = './train/'
TESTDATA_PATH = './test/testing-X.pkl'
ATTACK_TYPES = {
    'snmp': 0,
    'portmap': 1,
    'syn': 2,
    'dns': 3,
    'ssdp': 4,
    'webddos': 5,
    'mssql': 6,
    'tftp': 7,
    'ntp': 8,
    'udplag': 9,
    'ldap': 10,
    'netbios': 11,
    'udp': 12,
    'benign': 13,
}


class CompDataset(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self._data = [(x, y) for x, y in zip(X, Y)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)



def get_tag_columns(df, limit=10, force_int = True):
    '''find cols contains continuous data'''
    numerics = ['int16', 'int32', 'int64']
    ret = []
    for col in df.columns:
        if force_int and df[col].dtypes not in numerics:
            continue
        if df[col].nunique() < limit:
            ret.append(col)
    return ret


def normlize_data(df, have_target=True):
    if type(df) == np.ndarray:
        df = pd.DataFrame(df, columns=['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'SimillarHTTP', 'Inbound'])
    if have_target:
        target_label = df.columns[-1]
        y = df[target_label].values
        df = df.drop(target_label, axis=1)

    # one hot
    # tags = get_tag_columns(df, limit=6)
    tags = ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Inbound"]
    tags_df = df[tags]

    one_hot_df = pd.get_dummies(df[tags], columns=tags)
    # print(len(one_hot_df.columns))
    # print("{} lable columns: {}".format(len(one_hot_df.columns), "| ".join(one_hot_df.columns)))
    # print(one_hot_df.head())

    # Standard
    ss = StandardScaler()
    X = np.concatenate([ss.fit_transform(one_hot_df), df.drop(tags, axis=1).values], axis=1)
    if have_target:
        return X.astype("float32"), y.astype("float32")
    else:
        return X.astype("float32")
     

def extract_features(data, has_label=True):

    data['SimillarHTTP'] = 0.
    if has_label:
        return data.iloc[:, -80:-1]

    return data.iloc[:, -79:]


class UserRoundData(object):
    def __init__(self):
        self.data_dir = TRAINDATA_DIR
        self._user_datasets = []
        self.attack_types = ATTACK_TYPES
        self._load_data()

    def _get_data(self, fpath):
        if not fpath.endswith('csv'):
            return

        print('Load User Data: ', os.path.basename(fpath))
        data = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
        x = extract_features(data)
        y = np.array([
            self.attack_types[t.split('_')[-1].replace('-', '').lower()]
            for t in data.iloc[:, -1]
        ])
        # x = x.to_numpy().astype(np.float32)
        x = normlize_data(x, have_target=False)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        return (
            x,
            y,
        )

    def _load_data(self):
        _user_datasets = []
        self._user_datasets = []
        n = 0
        for root, dirs, fnames in os.walk(self.data_dir):
            for fname in fnames:
                # each file is for each user
                # user data can not be shared among users
                data = self._get_data(os.path.join(root, fname))
                if data is not None:
                    _user_datasets.append(data)
                n += 1
                if n > 20:
                    break

        for x, y in _user_datasets:
            self._user_datasets.append((
                x,
                y,
            ))

        self.n_users = len(_user_datasets)

    def all_data(self):
        X = np.concatenate([i[0] for i in self._user_datasets])
        y = np.concatenate([i[1] for i in self._user_datasets])
        return X, y

    def round_data(self, user_idx, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round.

        Args:
            user_idx: int,  in [0, self.n_users)
            n_round: int, round number
        """
        if n_round_samples == -1:
            return self._user_datasets[user_idx]

        n_samples = len(self._user_datasets[user_idx][1])
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples))

        return self._user_datasets[user_idx][0][choices], self._user_datasets[
            user_idx][1][choices]

    def uniform_random_loader(self, n_samples, batch_size=10000):
        X, Y = [], []
        n_samples_each_user = n_samples // len(self._user_datasets)
        if n_samples_each_user <= 0:
            n_samples_each_user = 1

        for idx in range(len(self._user_datasets)):
            x, y = self.round_data(user_idx=idx,
                                   n_round=0,
                                   n_round_samples=n_samples_each_user)
            X.append(x)
            Y.append(y)

        data = CompDataset(X=np.concatenate(X), Y=np.concatenate(Y))
        # balance
        # batch_size = 20
        # class_sample_count = [10, 1, 20, 3, 4] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
        # weights = 1 / torch.Tensor(class_sample_count)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
        # trainloader = data_utils.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, sampler = sampler)
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=min(batch_size, n_samples),
            shuffle=True,
        )

        return train_loader


def get_test_loader(batch_size=1000):
    if not os.path.exists(TESTDATA_PATH):
        return None
    with open(TESTDATA_PATH, 'rb') as fin:
        data = pickle.load(fin)

    test_loader = torch.utils.data.DataLoader(
        normlize_data(data['X'], have_target=False),
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader
