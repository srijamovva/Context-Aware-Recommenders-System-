import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FrappeDataset(Dataset):
    def __init__(self, data, context_feature_names):
        self.user_ids = torch.LongTensor(data['UserID'].values)
        self.item_ids = torch.LongTensor(data['ItemID'].values)
        self.ratings = torch.FloatTensor(data['Rating'].values)
        self.context_features = [torch.LongTensor(data[name].values) for name in context_feature_names]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        rating = self.ratings[idx]
        context_features = torch.stack([feature[idx] for feature in self.context_features], dim=0)
        return user_id, item_id,  context_features, rating


class Frappe_DataLoader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_data(self, train_data, test_data):
        
        context_feature_names = train_data.columns[3:]
        train_dataset = FrappeDataset(train_data, context_feature_names)
        test_dataset = FrappeDataset(test_data, context_feature_names)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, test_dataloader




class TripAdvisorDataset(Dataset):
    def __init__(self, data):
        self.user_ids = torch.LongTensor(data['UserID'].values)
        self.item_ids = torch.LongTensor(data['ItemID'].values)
        self.ratings = torch.FloatTensor(data['Rating'].values)
        self.contexts = torch.LongTensor(data['Context'].values)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.contexts[idx], self.ratings[idx]


class Trip_Dataloader():

    def __init__(self, batch_size):

        self.batch_size = batch_size

    def load_data(self, train_data, test_data):

        self.train_data = train_data
        self.test_data = test_data

        self.train_dataset = TripAdvisorDataset(self.train_data)
        self.test_dataset = TripAdvisorDataset(self.test_data)

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, test_dataloader


    





