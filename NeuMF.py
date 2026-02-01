import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_context, mf_dim, context_dim, mlp_layers, dropout, ui_train, device):
        super(NeuMF, self).__init__()

        self.ui_train = ui_train
        self.device = device

        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)

        self.context_embeddings = nn.ModuleList([
            nn.Embedding(num_values, context_dim) for num_values in num_context
        ])


        if self.ui_train:
            self.mlp_user_embedding = nn.Embedding(num_users, mlp_layers[0] // 2)
            self.mlp_item_embedding = nn.Embedding(num_items, mlp_layers[0] // 2)
            mlp_input_dim = mlp_layers[0] + (len(num_context) * context_dim)

        else:
            self.max_bits_user = len(bin(num_users)[2:])
            self.max_bits_item = len(bin(num_items)[2:])

            self.binary_table_users = {i: torch.tensor([int(bit) for bit in bin(i)[2:].zfill(self.max_bits_user)], dtype=torch.float32) for i in range(num_users)}
            self.binary_table_items = {i: torch.tensor([int(bit) for bit in bin(i)[2:].zfill(self.max_bits_item)], dtype=torch.float32) for i in range(num_items)}

            mlp_input_dim = self.max_bits_user + self.max_bits_item + (len(num_context) * context_dim)


        self.mlp_layers = nn.ModuleList()

        for i, out_size in enumerate(mlp_layers[1:]):
            self.mlp_layers.append(nn.Linear(mlp_input_dim, out_size))
            mlp_input_dim = out_size
        
        self.predict_layer = nn.Linear(mlp_layers[-1] + mf_dim, 1)

        self.dropout = nn.Dropout(p=dropout)

    def convert_ids_to_binary(self, ids, binary_table):
        binary_vector = torch.stack([binary_table[id_.item()] for id_ in ids])
        return binary_vector.to(self.device)

    def forward(self, user_indices, item_indices, context_indices):

        mf_user_embedding = self.mf_user_embedding(user_indices)
        mf_item_embedding = self.mf_item_embedding(item_indices)

        mf_vector = mf_user_embedding * mf_item_embedding

        if self.ui_train:
            mlp_user_embedding = self.mlp_user_embedding(user_indices)
            mlp_item_embedding = self.mlp_item_embedding(item_indices)

        else:
            mlp_user_embedding = self.convert_ids_to_binary(user_indices, self.binary_table_users)
            mlp_item_embedding = self.convert_ids_to_binary(item_indices, self.binary_table_items)


        context_embedded = [embed(context_indices[:, i]) for i, embed in enumerate(self.context_embeddings)]
        concatenated_context = torch.cat(context_embedded, dim=1)

        mlp_vector = torch.cat([mlp_user_embedding, mlp_item_embedding, concatenated_context], dim=-1)

        for layer in self.mlp_layers:
            mlp_vector = self.dropout(mlp_vector)
            mlp_vector = layer(mlp_vector)
            mlp_vector = F.relu(mlp_vector)

        vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        prediction = self.predict_layer(vector)

        return prediction.squeeze()
