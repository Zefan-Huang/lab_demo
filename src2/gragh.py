import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def main():
    df = pd.read_csv("result/global_predictions.csv")
    y_true = df["y_true"]
    y_prob = df["y_prob"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color="royalblue", linewidth=2, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Global ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("result2/global_ROC.png", dpi=300)
    plt.show()
    print(f"ROC saved: result2/global_ROC.png, AUC={auc:.3f}")

if __name__ == "__main__":
    main()