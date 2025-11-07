import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

def load_data(path="unsupervised/features.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} samples, columns = {df.columns.tolist()}")


    id_labels = df.groupby("id")["label"].first().reset_index()


    from sklearn.model_selection import train_test_split
    id_train, id_test = train_test_split(
        id_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=id_labels["label"]
    )

    df_train = df[df["id"].isin(id_train["id"])]
    df_test = df[df["id"].isin(id_test["id"])]

    print(f"Train IDs: {len(id_train)}, Test IDs: {len(id_test)}")
    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    X_train = df_train[['mean', 'std', 'trend', 'acf', 'low_ratio', 'high_ratio']].values
    y_train = df_train['label'].values
    X_test = df_test[['mean', 'std', 'trend', 'acf', 'low_ratio', 'high_ratio']].values
    y_test = df_test['label'].values

    X_train = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
    X_test = torch.tensor(X_test[:, np.newaxis, :], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test


def train_and_evaluate(model, X_train, X_test, y_train, y_test, epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []

    model.train()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train.to(device))
        loss = criterion(out, y_train.to(device))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = out.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            y_true = y_train.cpu().numpy().flatten()
            acc = (preds == y_true).mean()
            f1 = f1_score(y_true, preds, zero_division=0)
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
        print(f"Epoch {epoch + 1:02d}/{epochs} "
              f"- Loss: {loss.item():.4f} "
              f"- Acc: {acc:.3f} "
              f"- F1: {f1:.3f} "
              f"- Precision: {precision:.3f} "
              f"- Recall: {recall:.3f}")
        history.append({
            "epoch": epoch + 1,
            "loss": float(loss.item()),
            "acc": float(acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        })

    # evaluation
    model.eval()
    with torch.no_grad():
        probs = model(X_test.to(device)).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        y_true = y_test.cpu().numpy().flatten()

    auc = roc_auc_score(y_true, probs)
    f1 = f1_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)

    print("\nFinal Test Metrics:")
    print(f"AUC={auc:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    df_hist = pd.DataFrame(history)
    out_path = "result2/training_log.csv"
    df_hist.to_csv(out_path, index=False)

    return auc, f1, precision, recall, probs, y_true

def plot_roc_curve(y_true, probs, auc, save_path="result2/cnn_split_roc.png"):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (8:2 Split)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(save_path, dpi=500)
    plt.show()
    print(f"ROC curve saved to {save_path}")

def main():
    X_train, X_test, y_train, y_test = load_data("unsupervised/features.csv")

    model = CNN1D()
    auc, f1, precision, recall, probs, y_true = train_and_evaluate(
        model, X_train, X_test, y_train, y_test, epochs=50
    )

    plot_roc_curve(y_true, probs,auc)

if __name__ == "__main__":
    main()