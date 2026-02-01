import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from data_loader import Trip_Dataloader, Frappe_DataLoader
from NeuMF import NeuMF
from NCF import NCF

np.random.seed(42)


def train_evaluate(train_data, test_data, num_users, num_items, num_context, params):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = params['epochs']
    mf_dim = params['mf_dim'] 
    mlp_layers = params['mlp_layers']
    context_dim = params['context_dim']
    dropout = params['dropout']
    ui_train = params['ui_train']

    model = NCF(num_users, num_items, num_context, context_dim, mlp_layers, dropout, ui_train, device)
    # model = NeuMF(num_users, num_items, num_context, mf_dim, context_dim, mlp_layers, dropout, ui_train, device)


    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    
    criterion = nn.MSELoss()

    model = model.to(device)
    criterion = criterion.to(device)

    model.train()
    for epoch in range(epochs):
        for user_ids, item_ids, context, ratings in train_data:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            context = context.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            outputs = model(user_ids, item_ids, context)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

    model.eval()

    total_mae = 0
    total_rmse = 0
    total_count = 0

    for user_ids, item_ids, context, ratings in test_data:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        context = context.to(device)
        ratings = ratings.to(device)

        predictions = model(user_ids, item_ids, context)

        mae = torch.sum(torch.abs(predictions - ratings))
        rmse = torch.sum((predictions - ratings) ** 2)

        total_mae += mae
        total_rmse += rmse
        total_count += predictions.size(0)

    fold_mae = total_mae / total_count
    fold_rmse = torch.sqrt(total_rmse / total_count)

    return fold_mae.item(), fold_rmse.item()



def user_item_encoder(data, dataset):
    user_encoder = {user_id: i for i, user_id in enumerate(data['UserID'].unique())}
    item_encoder = {item_id: i for i, item_id in enumerate(data['ItemID'].unique())}

    data['UserID'] = data['UserID'].apply(lambda x: user_encoder[x])
    data['ItemID'] = data['ItemID'].apply(lambda x: item_encoder[x])

    if dataset == 'trip_advisor':
        context_encoder = {context_id: i for i, context_id in enumerate(data['TripType'].unique())}
        data['Context'] = data['TripType'].apply(lambda x: context_encoder[x])

        data = data.loc[:, ['UserID', 'ItemID', 'Rating', 'Context']]


    else:
        data['Rating'] = np.log10(data['Rating'])
        data['Rating'] = data['Rating'].apply(round, args=(2,))
        # data['Context'] = data.iloc[:, 3:].applymap(str).agg('_'.join, axis=1)

        # context_encoder = {context_id: i for i, context_id in enumerate(data['Context'].unique())}
        # data['Context'] = data['Context'].apply(lambda x: context_encoder[x])


        for context in data.columns[3:-1]:
            context_encoder = {context_id: i for i, context_id in enumerate(data[context].unique())}
            data[context] = data[context].apply(lambda x: context_encoder[x])

        data = data.iloc[:, :-1]


    return data

        

def cross_validate():

    dataset = 'frappe'

    if dataset == 'trip_advisor':

        data = pd.read_csv("/scratch/mvongala/CARS/Travel_TripAdvisor_v2/Data_TripAdvisor_v2.csv")

        data = data.loc[:, ['UserID', 'ItemID', 'Rating', 'TripType', 'UserTimeZone']]


    else:
        data = pd.read_csv("/scratch/mvongala/CARS_temp/Mobile_Frappe/frappe/frappe.csv", sep = '\t')

        first_three_columns = data.columns[:3]

        new_names = {old_name: new_name for old_name, new_name in zip(first_three_columns, ['UserID', 'ItemID', 'Rating'])}

        data.rename(columns=new_names, inplace=True)

    params = {
        'learning_rate': 2e-4,
        'optimizer': 'Adam',
        'mlp_layers': [128, 64, 32, 16],
        'dropout': 0.2,
        'mf_dim': 4,
        'epochs': 100,
        'context_dim': 8,
        'ui_train': False
        
        }


    data = user_item_encoder(data, dataset)    


    num_users = len(data['UserID'].unique())
    num_items = len(data['ItemID'].unique())

    ui_encoding = 'bits'


    num_context = [len(data[context].unique()) for context in data.columns[3:]]

    num_folds = 5

    indices = np.arange(len(data))

    np.random.shuffle(indices)

    fold_indices = np.array_split(indices, num_folds)
    
    mae, rmse = 0, 0

    for fold_idx in range(num_folds):

        print(f'Fold: {fold_idx + 1}')
        val_indices = fold_indices[fold_idx]
        train_indices = np.concatenate([fold_indices[i] for i in range(num_folds) if i != fold_idx])
        train_data = data.iloc[train_indices, :].reset_index(drop = True)
        test_data = data.iloc[val_indices, :].reset_index(drop = True)


        if dataset == 'frappe':
            loader = Frappe_DataLoader(batch_size = 128)

        else:
            loader = Trip_Dataloader(batch_size = 128)

        train_data, test_data = loader.load_data(train_data, test_data)

        fold_mae, fold_rmse = train_evaluate(train_data, test_data, num_users, num_items, num_context, params)
        mae += fold_mae
        rmse += fold_rmse

    print("MAE: {:.2f} ".format(mae/num_folds))


if __name__ == "__main__":
    cross_validate()
    