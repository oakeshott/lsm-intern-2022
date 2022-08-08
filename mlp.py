import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random
import numpy as np
import os
import joblib
from model import MLPClassifier

class Preprocessing:
    def __init__(self, is_train=True):
        self.transformer = dict()
        self.is_train = is_train
        if not self.is_train:
            self.transformer = self.load()

    def __call__(self, df, is_timeseries=False):
        if self.is_train:
            return self.fit_transform(df)
        else:
            return self.transform(df)

    def fit_transform(self, df):
        for column in df.columns:
            if 'cpu-util' in column:
                self.transformer[column] = MinMaxScaler()
            else:
                self.transformer[column] = StandardScaler()
            value = self.transformer[column].fit_transform(pd.DataFrame(df[column]))
            df.loc[:, column] = value
        return df

    def transform(self, df):
        for column in df.columns:
            value = self.transformer[column].transform(
                pd.DataFrame(df[column]))
            df.loc[:, column] = value
        return df

    def dump(self, filename='/tmp/mlp_transfomer.bin'):
        with open(filename, 'wb') as f:
            joblib.dump(self.transformer, f)

    def load(self, filename='/tmp/mlp_transfomer.bin'):
        with open(filename, 'rb') as f:
            data = joblib.load(f)
        return data

class NetworkMetricsDataset(Dataset):
    def __init__(self, path, metrics, device, transformer=None):
        self.path = path
        self.metrics = metrics
        self.device = device
        self.transformer = transformer

        data = []
        for metric in tqdm(self.metrics):
            df = pd.read_csv(os.path.join(self.path, metric + '.tsv'), sep="\t", index_col=0)
            df = df.fillna(0)
            df = df.sort_values("timestamp")
            df = df.set_index("timestamp")
            columns = {name: metric + '-' + name for name in df.columns}
            df.rename(columns=columns, inplace=True)
            if self.transformer:
                df = self.transformer(df)
            data.append(df)
        self.dataframe = pd.concat(data, axis=1)
        self.data = self.dataframe.values
        self.data_size = len(self.dataframe)
        self.labels = pd.read_csv(os.path.join(self.path, 'label.tsv'), sep="\t", index_col=0).set_index("timestamp").values

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        ret = self.data[idx]
        ret = torch.tensor(ret, dtype=torch.float, device=self.device)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float, device=self.device)

        return ret, label

class Environment:
    def __init__(self, model_dir, max_epoch, batchsize, seed):
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.batchsize = batchsize
        self.max_epoch = max_epoch
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = ["cpu-util", "tx-pps", "rx-pps", "network-incoming-packets-rate", "network-outgoing-packets-rate", "prefix-activity-received-current-prefixes"]
        self.events = {
                'normal': 0,
                'ixnetwork-bgp-hijacking-start': 1,
                'ixnetwork-bgp-injection-start': 2,
                'node-down': 3,
                'interface-down': 4,
                'packet-loss-delay': 5,
                }

    def train(self, dataset_path):
        transformer = Preprocessing(is_train=True)
        dataset = NetworkMetricsDataset(dataset_path, self.metrics, self.device, transformer)
        transformer.dump()

        train_indices, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=0.2,
            stratify=dataset.labels,
            random_state=self.seed,
        )
        train_dataset = Subset(dataset, train_indices)
        train_size    = len(train_dataset)
        val_dataset   = Subset(dataset, val_indices)
        val_size      = len(val_dataset)
        print(f'train size : {train_size} val size: {val_size}')
        train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize)
        val_dataloader   = DataLoader(val_dataset, batch_size=val_size)

        val_data, val_labels = iter(val_dataloader).next()
        val_data   = val_data.float().to(self.device)
        val_labels = val_labels.long().to(self.device).view(-1)

        input_dim  = list(train_dataset[0][0].shape)[-1]
        output_dim = len(self.events.keys())

        model = MLPClassifier(input_dim, output_dim).to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, self.max_epoch+1):
            running_loss = 0
            correct      = 0
            total        = 0

            # Training
            model = model.train()
            for train_data, train_labels in train_dataloader:
                train_data = train_data.float().to(self.device)
                train_labels = train_labels.long().to(self.device).view(-1)

                model.zero_grad()
                train_scores = model(train_data)
                loss = loss_function(train_scores, train_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predict = torch.max(train_scores.data, 1)
                correct += (predict == train_labels).sum().item()
                total += train_labels.size(0)

            train_loss = running_loss / len(train_dataloader)
            train_acc = correct / total

            # Check model validation
            model = model.eval()
            with torch.no_grad():
                val_scores = model(val_data)
                val_loss = loss_function(val_scores, val_labels)

                bi_scores = torch.argmax(val_scores, dim=1).to(self.device).numpy()
                y_val_scores = val_labels.to(self.device).numpy()
                val_acc = accuracy_score(y_val_scores, bi_scores)

            print(f'EPOCH: [{epoch}/{self.max_epoch}] train loss: {train_loss:.4f} train acc: {train_acc:.4f} val loss: {val_loss:.4f} val acc: {val_acc:4f}')
            # Export model
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"./{self.model_dir}/mlp_{epoch}.mdl")

    def test(self, dataset_path):
        model_paths = os.listdirs(self.model_dir)
        transformer = Preprocessing(is_train=False)
        dataset = NetworkMetricsDataset(dataset_path, metrics, self.device, transformer)

        transformer = Preprocessing(is_train=False)
        dataset = NetworkMetricsDataset(path, metrics, self.device, transformer)

        input_dim = list(dataset[0][0].shape)[-1]
        output_dim = len(self.events.keys())

        test_dataloader = DataLoader(dataset, batch_size=len(dataset))
        test_data, test_label = iter(test_dataloader).next()
        test_data = test_data.float().to(self.device)
        test_label = test_label.long().to(self.device).view(-1)

        for model_path in model_paths:
            model_path = os.path.join(self.model_dir, model_path)
            model = MLPClassifier(input_dim, output_dim).to(self.device)
            model.load_state_dict(torch.load(model_path))
            model = model.eval()
            loss_function = nn.CrossEntropyLoss()
            with torch.no_grad():
                test_scores = model(test_data)
                loss = loss_function(test_scores, test_label)
                bi_scores = torch.argmax(test_scores, dim=1).to('cpu').numpy()
                y_test_scores = test_label.to('cpu').numpy()
            print(model_path)
            print(accuracy_score(y_test_scores, bi_scores))
            print(classification_report(y_test_scores, bi_scores, target_names=list(self.events.keys())))

def main():
    parser = argparse.ArgumentParser(description="MLP Classification")
    parser.add_argument('--train', action="store_true",
                    help='is train mode')
    parser.add_argument('--test', action="store_true",
                    help='is test mode')
    parser.add_argument('--model-dir', default="models/mlp",
                    help='model dir')
    parser.add_argument('--max-epoch', type=int, default=100,
                    help='max-epoch')
    parser.add_argument('--batchsize', type=int, default=16,
                    help='batchsize')
    parser.add_argument('--seed', type=int, default=1,
                    help='seed')
    args = parser.parse_args()

    if args.train:
        dataset_path = 'dataset/train'
        env = Environment(model_dir, max_epoch, batchsize, seed)
        env.train(dataset_path)
    elif args.test:
        dataset_path = 'dataset/test'
        env = Environment(model_dir, max_epoch, batchsize, seed)
        env.test(dataset_path)

if __name__ == '__main__':
    main()
