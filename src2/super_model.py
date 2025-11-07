import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
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

def train_eval_loop(x, y, groups, epochs, lr = 1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    result = []
    all_y_true, all_y_prob, all_y_pred = [], [], []
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups)):
        model = CNN1D().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        x_train, y_train = torch.tensor(x[train_idx], dtype=torch.float32).to(device), torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1).to(device)
        x_test, y_test = torch.tensor(x[test_idx] ,dtype=torch.float32).to(device), torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1).to(device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x_test).cpu().numpy().flatten()
            y_true_tensor = y_test.detach().cpu()
            y_true = y_true_tensor.numpy().flatten()

        if len(np.unique(y_true)) < 2:
            print(f"Fold {fold + 1}: Only one class present in y_true, skipping AUC.")
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, pred)

        pred_label = (pred > 0.4).astype(int)
        f1 = f1_score(y_test.numpy(), pred_label)
        precision = precision_score(y_true, pred_label, zero_division=0)
        recall = recall_score(y_true, pred_label, zero_division=0)
        result.append((fold + 1, auc, f1, precision, recall))
        print(f'Fold {fold + 1}: AUC={auc:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}')

        all_y_true.extend(y_true.tolist())
        all_y_prob.extend(pred.tolist())
        all_y_pred.extend(pred_label.tolist())

        aucs = [r[1] for r in result if not np.isnan(r[1])]
        mean_auc = np.mean(aucs) if aucs else np.nan
        mean_f1 = np.mean([r[2] for r in result])
        mean_precision = np.mean([r[3] for r in result])
        mean_recall = np.mean([r[4] for r in result])

        try:
            global_auc = roc_auc_score(all_y_true, all_y_prob)
        except Exception:
            global_auc = np.nan

        global_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        global_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        global_recall = recall_score(all_y_true, all_y_pred, zero_division=0)

    print(f"Global AUC across all folds: {global_auc:.3f}")
    print(f"AUC={global_auc:.3f}, F1={global_f1:.3f}, "f"Precision={global_precision:.3f}, Recall={global_recall:.3f}")
    df_pred = pd.DataFrame({
        "y_true": all_y_true,
        "y_prob": all_y_prob
    })
    df_pred.to_csv("result/global_predictions.csv", index=False)

    result = { "fold_results": result,
    "fold_mean": {
        "auc": mean_auc,
        "f1": mean_f1,
        "precision": mean_precision,
        "recall": mean_recall
    },
    "global": {
        "auc": global_auc,
        "f1": global_f1,
        "precision": global_precision,
        "recall": global_recall}
    }
    return result

def save_data(result):
    fold_results = result["fold_results"]
    df_folds = pd.DataFrame(fold_results, columns=["fold", "auc", "f1", "precision", "recall"])
    df_folds.to_csv("result/super_cv20_fold_metrics.csv", index=False)

    summary = {
        "metric": ["AUC", "F1", "Precision", "Recall"],
        "fold_mean": [
            result["fold_mean"]["auc"],
            result["fold_mean"]["f1"],
            result["fold_mean"]["precision"],
            result["fold_mean"]["recall"],
        ],
        "global": [
            result["global"]["auc"],
            result["global"]["f1"],
            result["global"]["precision"],
            result["global"]["recall"],
        ],
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("result/super_cv20_summary.csv", index=False)
    print('Saved')


def main():
    df_features = pd.read_csv('unsupervised/features.csv')
    x = df_features[['mean','std','trend','acf','low_ratio','high_ratio']].values
    y = df_features['label'].values
    groups = df_features['id'].values
    x = x[:, np.newaxis, :]
    results = train_eval_loop(x, y, groups, epochs = 30)
    save_data(results)

if __name__ == '__main__':
    main()