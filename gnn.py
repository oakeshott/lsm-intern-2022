import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random
import numpy as np
import os
import joblib
import argparse
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, train_test_split_edges
import networkx as nx
from model import GCNClassifier

def collate_fn(batch):
    print(len(batch))
    print(type(batch))
class NetworkMetricsWithTopologyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [filename for filename in sorted(os.listdir(self.raw_dir))]

    @property
    def processed_file_names(self):
        return [i for i in sorted(os.listdir(self.processed_dir)) if 'data' in i]

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            g = nx.read_gpickle(raw_path)
            for n in g.nodes():
                label = g.nodes()[n]['label']
                del g.nodes()[n]['label']
            data = from_networkx(g)
            data.y = torch.tensor(label)
            data.num_nodes = len(g.nodes())
            data.edge_attr = []
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

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
        dataset = NetworkMetricsWithTopologyDataset(dataset_path)
        # dataset = [d for d in dataset]

        labels = [dataset[i].y for i in range(len(dataset))]
        train_indices, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=0.2,
            stratify=labels,
            random_state=self.seed,
            )

        train_dataset = dataset[train_indices]
        # train_dataset = [dataset[i] for i in range(len(dataset)) if i in train_indices]
        train_size    = len(train_dataset)
        val_dataset   = dataset[val_indices]
        # val_dataset = [dataset[i] for i in range(len(dataset)) if i in val_indices]
        val_size      = len(val_dataset)
        print(f'train size : {train_size} val size: {val_size}')

        train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, collate_fn=collate_fn)
        val_dataloader   = DataLoader(val_dataset, batch_size=val_size, collate_fn=collate_fn)

        input_dim = train_dataset[0].x.shape[-1]
        output_dim = len(self.events.keys())

        model         = GCNClassifier(input_dim, output_dim).to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer     = optim.Adam(model.parameters(), lr=1e-3)

        val_data = next(iter(val_dataloader))
        val_batch = val_data.batch.to(self.device)
        val_edge_index = val_data.edge_index.to(self.device)
        val_edge_attr = None
        val_labels = val_data.y.long().to(self.device).view(-1)
        val_data = val_data.x.float().to(self.device)
        for epoch in range(1, self.max_epoch+1):
            running_loss = 0
            correct      = 0
            total        = 0

            # Training
            model = model.train()
            for train_data in train_dataloader:
                train_labels = train_data.y
                x = train_data.x.float().to(self.device)
                edge_index = train_data.edge_index.to(self.device)
                batch = train_data.batch.to(self.device)
                edge_attr = None
                train_labels = train_data.y.long().to(self.device).view(-1)

                model.zero_grad()
                train_scores = model(x, edge_index, batch, edge_attr)
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
                val_scores = model(val_data, val_edge_index, val_batch, val_edge_attr)
                val_loss = loss_function(val_scores, val_labels)

                bi_scores = torch.argmax(val_scores, dim=1).to(self.device).numpy()
                y_val_scores = val_labels.to(self.device).numpy()
                val_acc = accuracy_score(y_val_scores, bi_scores)

            print(f'EPOCH: [{epoch}/{self.max_epoch}] train loss: {train_loss:.4f} train acc: {train_acc:.4f} val loss: {val_loss:.4f} val acc: {val_acc:4f}')
            # Export model
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"./{self.model_dir}/gcn_{epoch}.mdl")

    def test(self, dataset_path):
        model_paths = os.listdir(self.model_dir)
        dataset = NetworkMetricsWithTopologyDataset(dataset_path)

        input_dim = dataset[0].x.shape[-1]
        output_dim = len(self.events.keys())

        test_dataloader = DataLoader(dataset, batch_size=len(dataset))
        test_data = iter(test_dataloader).next()
        x = test_data.x.float().to(self.device)
        edge_index = test_data.edge_index.to(self.device)
        batch = test_data.batch.to(self.device)
        edge_attr = None
        test_label = test_data.y.long().to(self.device).view(-1)
        for model_path in sorted(model_paths):
            model_path = os.path.join(self.model_dir, model_path)
            model = GCNClassifier(input_dim, output_dim).to(self.device)
            model.load_state_dict(torch.load(model_path))
            model = model.eval()
            loss_function = nn.CrossEntropyLoss()
            with torch.no_grad():
                test_scores = model(x, edge_index, batch, edge_attr)
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
    parser.add_argument('--model-dir', default="models/gcn",
                    help='model dir')
    parser.add_argument('--max-epoch', type=int, default=100,
                    help='max-epoch')
    parser.add_argument('--batchsize', type=int, default=16,
                    help='batchsize')
    parser.add_argument('--seed', type=int, default=1,
                    help='seed')
    args = parser.parse_args()

    model_dir = args.model_dir
    seed      = args.seed
    batchsize = args.batchsize
    max_epoch = args.max_epoch

    if args.train:
        dataset_path = 'dataset/train/network'
        env = Environment(model_dir, max_epoch, batchsize, seed)
        env.train(dataset_path)
    if args.test:
        dataset_path = 'dataset/test/network'
        env = Environment(model_dir, max_epoch, batchsize, seed)
        env.test(dataset_path)

if __name__ == '__main__':
    main()
