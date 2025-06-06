import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        #data_path = os.path.join('data/PeMSD4/pems04.npz')
        data = np.load("/content/SGCRN_RD/data/PEMS04/pems04.npz")['data'][:, :, 0]  #onley the first dimension, traffic flow data
        #data = np.load("C:/Users/Hoda/A - Uni/thesis/SGCRN_RD/data/PEMS04/pems04.npz")['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        #data_path = os.path.join('data/PeMSD8/pems08.npz')
        data = np.load('/content/SGCRN_RD/data/PEMS08/pems08.npz')['data'][:, :, 0]  #onley the first dimension, traffic flow data

    elif dataset == 'PEMSD3':
        #data_path = os.path.join('data/PeMSD8/pems08.npz')
        data = np.load('/content/SGCRN_RD/data/PEMS03/PEMS03.npz')['data'][:, :, 0]  #onley the first dimension, traffic flow data    

    elif dataset == 'PEMSD7':
        #data_path = os.path.join('data/PeMSD8/pems08.npz')
        data = np.load('/content/SGCRN_RD/data/PEMS07/PEMS07.npz')['data'][:, :, 0]  #onley the first dimension, traffic flow data 

    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data 
