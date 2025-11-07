import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(path="unsupervised/features.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} samples, columns = {df.columns.tolist()}")

    id_labels = df.groupby("id")["label"].first().reset_index()

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

    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    print("Logistic Regression Test Metrics:")
    print(f"AUC={auc:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

    df_metrics = pd.DataFrame({
        "AUC": [auc],
        "F1": [f1],
        "Precision": [precision],
        "Recall": [recall]
    })
    df_metrics.to_csv("result2/logistic_metrics.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", linewidth=2, color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig("result2/logistic_roc.png", dpi=300)
    plt.show()
    print("ROC curve saved to result2/logistic_roc.png")

    return auc, f1, precision, recall

def main():
    X_train, X_test, y_train, y_test = load_data("unsupervised/features.csv")
    train_logistic_regression(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()