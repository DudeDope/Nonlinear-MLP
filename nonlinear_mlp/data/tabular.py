import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_adult(batch_size=256, test_size=0.2, num_workers=2):
    # Placeholder: expects preprocessed CSV or uses UCI fetch (user adapt)
    # For brevity, raise if file not found.
    path = "data/adult.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Please place a preprocessed adult.csv at data/adult.csv")
    df = pd.read_csv(path)
    y = df["label"].values
    X = df.drop(columns=["label"]).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    train = TabularDataset(X_tr, y_tr)
    test = TabularDataset(X_te, y_te)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        X.shape[1],
        len(set(y))
    )