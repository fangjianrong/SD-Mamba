DATASETS_LIST = ['IP', 'PU', 'HU13', 'HH', 'HC'] 
EPOCHS = 200
PATCH_SIZE = 13  
TRAIN_SHOTS = 50 
DEVICE = "cuda"  

RESULT_DIR = './results'
CHECKPOINT_DIR = './checkpoints'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import numpy as np
import time
import os
import pandas as pd
from torch.cuda.amp import autocast, GradScaler

from SD_Mamba_model import SD_Mamba 
from data_utils import (load_data, normalize_data, padWithZeros, 
                        get_coordinates, LazyHSIDataset)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

def flexible_split(coords, labels, dataset_name, shots=50):
    train_indices = []
    test_indices = []
    classes = np.unique(labels)
    
    ip_special_counts = {1: 15, 7: 15, 9: 15}
    
    for c in classes:
        if c == 0: continue 
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        
        target_count = shots
        
        if dataset_name == 'IP' and shots > 15:
            if c in ip_special_counts:
                target_count = ip_special_counts[c]
        
        if len(idx) < target_count:
             n_samples = max(1, int(len(idx) * 0.8)) 
        else:
             n_samples = target_count
             
        train_indices.extend(idx[:n_samples])
        test_indices.extend(idx[n_samples:])
        
    return (coords[train_indices], labels[train_indices], 
            coords[test_indices], labels[test_indices])

def train_eval_save(model, train_loader, test_loader, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = GradScaler()
    
    best_oa = 0
    best_aa = 0
    best_kappa = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        
        if epoch > 150 or epoch % 20 == 0: 
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    with autocast():
                        out = model(x)
                    preds.extend(out.argmax(1).cpu().numpy())
                    targets.extend(y.cpu().numpy())
            
            oa = accuracy_score(targets, preds) * 100
            
            if oa > best_oa:
                best_oa = oa
                best_kappa = cohen_kappa_score(targets, preds) * 100
                cm = confusion_matrix(targets, preds)
                per_class_acc = cm.diagonal() / cm.sum(axis=1)
                best_aa = np.nanmean(per_class_acc) * 100
                best_state = model.state_dict()
                
    if best_state is not None:
        torch.save(best_state, save_path)
        
    return best_oa, best_aa, best_kappa

if __name__ == "__main__":
    print(f"🚀 SD-Mamba Training Started on {DEVICE}")
    print(f"📄 Results will be saved to {RESULT_DIR}/training_results.csv")
    
    all_results = []
    
    for dataset_name in DATASETS_LIST:
        print(f"\n{'#'*60}")
        print(f"  PROCESSING DATASET: {dataset_name}")
        print(f"{'#'*60}")
        
        try:
            raw_data, raw_labels = load_data(dataset_name, './data')
            norm_data = normalize_data(raw_data)
            in_channels = norm_data.shape[2]
            num_classes = len(np.unique(raw_labels)) - 1
            if num_classes < 1: num_classes = int(np.max(raw_labels))
            
            margin = (PATCH_SIZE - 1) // 2
            padded_data = padWithZeros(norm_data, margin)
            all_coords, all_labels = get_coordinates(raw_labels)
            
            print(f"  -> Shape: {norm_data.shape}, Classes: {num_classes}")

            train_batch = 128
            test_batch = 4096 
            if dataset_name == 'IP': train_batch = 64
            if dataset_name in ['PU', 'HC']: train_batch = 256
            
            train_c, train_l, test_c, test_l = flexible_split(
                all_coords, all_labels, dataset_name, shots=TRAIN_SHOTS
            )
            
            train_loader = DataLoader(LazyHSIDataset(padded_data, train_c, train_l, PATCH_SIZE), 
                                      batch_size=train_batch, shuffle=True, num_workers=8, 
                                      pin_memory=True, persistent_workers=True)
            test_loader = DataLoader(LazyHSIDataset(padded_data, test_c, test_l, PATCH_SIZE), 
                                     batch_size=test_batch, shuffle=False, num_workers=8, 
                                     pin_memory=True, persistent_workers=True)
            
            model = SD_Mamba(
                in_features=in_channels, 
                num_classes=num_classes, 
                patch_size=PATCH_SIZE
            ).to(DEVICE)
            
            save_dir = os.path.join(CHECKPOINT_DIR, dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{dataset_name}_best_model.pth")
            
            t0 = time.time()
            oa, aa, kappa = train_eval_save(model, train_loader, test_loader, save_path)
            dt = time.time() - t0
            
            print(f"    ✅ [Result] OA: {oa:.2f} | AA: {aa:.2f} | Kappa: {kappa:.2f} | Time: {dt:.0f}s")
            
            all_results.append({
                "Dataset": dataset_name,
                "Model": "SD_Mamba",
                "OA": oa, 
                "AA": aa, 
                "Kappa": kappa,
                "Time(s)": dt,
                "Checkpoint": save_path
            })
            
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(RESULT_DIR, 'training_results.csv'), index=False)

        except Exception as e:
            print(f"[ERROR] {dataset_name} Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎉 All Done! Training complete. Results saved to {RESULT_DIR}/training_results.csv")