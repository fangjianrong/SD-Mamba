import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
import os
import h5py

def load_data(dataset_name, data_path='./data'):
    dataset_name = dataset_name.upper()
    
    datasets = {
        'IP': ['Indian_pines_corrected.mat', 'Indian_pines_gt.mat'],
        'PU': ['PaviaU.mat', 'PaviaU_gt.mat'],
        'HU13': ['Houston.mat', 'Houston_gt.mat'],
        'HH': ['WHU_Hi_HongHu.mat', 'WHU_Hi_HongHu_gt.mat'],
        'HC': ['WHU_Hi_HanChuan.mat', 'WHU_Hi_HanChuan_gt.mat'],
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    files = datasets[dataset_name]
    data_file = os.path.join(data_path, files[0])
    label_file = os.path.join(data_path, files[1])

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")

    def read_content(filename):
        try:
            mat = sio.loadmat(filename)
            best_key = max((k for k in mat.keys() if not k.startswith('__')), key=lambda k: mat[k].size, default=None)
            if best_key:
                return mat[best_key]
        except NotImplementedError:
            pass

        try:
            with h5py.File(filename, 'r') as f:
                best_key = max((k for k in f.keys() if hasattr(f[k], 'size')), key=lambda k: f[k].size, default=None)
                if best_key:
                    content = f[best_key][()]
                    if content.ndim == 3:
                        return content.transpose(2, 1, 0)
                    elif content.ndim == 2:
                        return content.transpose(1, 0)
        except Exception as e:
            raise ValueError(f"Error reading {filename} with h5py: {e}")
            
        raise ValueError(f"Could not load data from {filename}.")

    data_hsi = read_content(data_file)
    labels = read_content(label_file)

    if labels.ndim == 2:
        H, W, _ = data_hsi.shape
        H_l, W_l = labels.shape
        
        if H == W_l and W == H_l:
            labels = labels.T
        elif data_hsi.shape[1] == H_l and data_hsi.shape[0] == W_l:
            data_hsi = data_hsi.transpose(1, 0, 2)
        elif H != H_l or W != W_l:
            raise ValueError(f"Shape mismatch! Data: {data_hsi.shape}, Labels: {labels.shape}")

    return data_hsi, labels

def normalize_data(X):
    min_val = np.min(X)
    max_val = np.max(X)
    return (X - min_val) / (max_val - min_val + 1e-6)

def padWithZeros(X, margin=2):
    return np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

def get_coordinates(y, removeZeroLabels=True):
    if removeZeroLabels:
        r, c = np.nonzero(y)
    else:
        r, c = np.indices(y.shape).reshape(2, -1)
    
    coords = np.column_stack((r, c))
    labels = y[r, c]
    return coords, labels

def split_coords_paper_specific(coords, labels, dataset_name):
    train_indices = []
    test_indices = []
    classes = np.unique(labels)
    
    ip_special_counts = {1: 15, 7: 15, 9: 15}
    
    for c in classes:
        if c == 0: continue 
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        
        target_count = 50 
        if dataset_name == 'IP' and c in ip_special_counts:
            target_count = ip_special_counts[c]
        
        n_samples = max(1, int(len(idx) * 0.8)) if len(idx) < target_count else target_count
             
        train_indices.extend(idx[:n_samples])
        test_indices.extend(idx[n_samples:])
        
    return (coords[train_indices], labels[train_indices], 
            coords[test_indices], labels[test_indices])

class LazyHSIDataset(data.Dataset):
    def __init__(self, padded_data, coords, labels, windowSize=11):
        self.data = np.transpose(padded_data, (2, 0, 1)) 
        self.coords = coords
        self.labels_tensor = torch.from_numpy(labels).long() - 1
        self.windowSize = windowSize
    
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = self.data[:, x : x + self.windowSize, y : y + self.windowSize]
        return torch.from_numpy(patch).float(), self.labels_tensor[idx]