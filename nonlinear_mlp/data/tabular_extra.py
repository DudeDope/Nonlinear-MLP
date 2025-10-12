```python
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from nonlinear_mlp.data.tabular import TabularDataset
from torch.utils.data import DataLoader

def _standard_split(X, y, test_size=0.2):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    return (TabularDataset(X_tr, y_tr), TabularDataset(X_te, y_te), X.shape[1], len(set(y)))

def load_wine_quality(batch_size=256, test_size=0.2, num_workers=2, path="data/winequality-red.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError("Download UCI Wine Quality red to data/winequality-red.csv")
    df = pd.read_csv(path, sep=';')
    # Convert quality to classification (e.g., >=6 positive else negative)
    df['label'] = (df['quality'] >= 6).astype(int)
    y = df['label'].values
    X = df.drop(columns=['quality','label']).values
    tr, te, in_dim, n_cls = _standard_split(X, y, test_size)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        in_dim,
        n_cls
    )
```