import os
import numpy as np
import torch
import torch.utils.data
from dagma.linear import DagmaLinear
from sklearn.decomposition import NMF
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        minimum, maximum = (data.min(axis=0, keepdims=True), data.max(axis=0, keepdims=True)) if column_wise else (data.min(), data.max())
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
    elif normalizer == 'max11':
        minimum, maximum = (data.min(axis=0, keepdims=True), data.max(axis=0, keepdims=True)) if column_wise else (data.min(), data.max())
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
    elif normalizer == 'std':
        mean, std = (data.mean(axis=0, keepdims=True), data.std(axis=0, keepdims=True)) if column_wise else (data.mean(), data.std())
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
    elif normalizer == 'cmax':
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
    else:
        raise ValueError("Invalid normalization method")
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    T = int((24 * 60) / interval)
    test_data = data[-T * test_days:]
    val_data = data[-T * (test_days + val_days):-T * test_days]
    train_data = data[:-T * (test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = torch.cuda.is_available()
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    dataset = torch.utils.data.TensorDataset(X, Y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# initial_embedding Dagma
def generate_adj_matrix_dagma(train_data, prediction_length=1, lambda1=0.02):
    num_nodes = train_data.shape[1]
    W_est_all = np.zeros((num_nodes, num_nodes, prediction_length))

    for i in range(prediction_length):
        X = train_data[i::prediction_length]
        X = X.squeeze()
        model = DagmaLinear(loss_type='l2')
        w_est = model.fit(X, lambda1=lambda1)
        W_est_all[:, :, i] = w_est

    adj_all = np.zeros(W_est_all.shape, dtype=int)
    adj_all[W_est_all > 0] = 1
    adj = np.any(adj_all, axis=2)
    return adj.astype(int)

def generate_nmf_embeddings(adj_matrix, embedding_dim=10):
    adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
    m, p, n = torch.svd(adjacency_matrix_tensor)
    embeddings = torch.mm(m[:, :d], torch.diag(p[:d] ** 0.5))
    return torch.tensor(embeddings, dtype=torch.float32)

def get_dataloader(args, normalizer='std', tod=False, dow=False, weather=False, single=True):
    data = load_st_dataset(args.dataset)  # Shape: (T, N, D)
    
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    
    if args.test_ratio > 1:
        x_train, x_val, x_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        x_train, x_val, x_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    
    x_train, y_train = Add_Window_Horizon(x_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(x_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(x_test, args.lag, args.horizon, single)
    
    # Print dataset splits
    print(f"Train:  {x_train.shape} {y_train.shape}")
    print(f"Val:    {x_val.shape} {y_val.shape}")
    print(f"Test:   {x_test.shape} {y_test.shape}")
    
    train_dataloader = data_loader(x_train, y_train, args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    
    return train_dataloader, val_dataloader, test_dataloader, scaler


def get_train_adj_matrix_and_embeddings(dataset_name, val_ratio=0.1, test_ratio=0.2, prediction_length=12, lambda1=0.02, embedding_dim=10):
    data = load_st_dataset(dataset_name)
    data_normalized, _ = normalize_dataset(data, 'max01')
    train_data, _, _ = split_data_by_ratio(data_normalized, val_ratio, test_ratio)
    adj_matrix = generate_adj_matrix_dagma(train_data, prediction_length, lambda1)
    embeddings = generate_nmf_embeddings(adj_matrix, embedding_dim)
    return adj_matrix, embeddings

if __name__ == '__main__':
    dataset_name = 'PEMSD4'
    val_ratio = 0.2
    test_ratio = 0.2
    prediction_length = 12
    lambda1 = 0.02
    embedding_dim = 10

    adj_matrix, embeddings = get_train_adj_matrix_and_embeddings(dataset_name, val_ratio, test_ratio, prediction_length, lambda1, embedding_dim)
    print(f"Adjacency Matrix Shape: {adj_matrix.shape}")
    print(f"Embeddings Shape: {embeddings.shape}")
    np.save("adj_matrix.npy", adj_matrix)
    torch.save(embeddings, "embeddings.pt")
