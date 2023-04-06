import numpy as np
import torch 

def split_train_val(data, ptrain=0.3, pval=0.2):
    train_mask = np.random.rand(len(data)) <= ptrain
    val_mask = np.random.rand(len(data)) >= (1-pval)
    index = np.array(range(len(data)))
    train_id, test_id = index[train_mask], index[val_mask]
    train = torch.utils.data.Subset(data, train_id)
    test = torch.utils.data.Subset(data, test_id)
    return train, test