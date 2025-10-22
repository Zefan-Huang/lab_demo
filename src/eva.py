from numpy.matrixlib.defmatrix import matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch
from modeling import load_data, CNN
import pandas as pd

def evaluate_classification(model, test_loader, device):
    model.eval()
    y_true, y_pred, probs = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            preds = (output > 0.5).float()
            # Flatten to 1D lists for sklearn metrics
            y_true.extend(y.view(-1).cpu().numpy().tolist())
            y_pred.extend(preds.view(-1).cpu().numpy().tolist())
            probs.extend(output.view(-1).cpu().numpy().tolist())

    # Classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_v = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "precision": precision,
        "recall": recall_v,
        "f1": f1
    }
    print(f'Precision: {precision:.2f}, Recall: {recall_v:.2f}, F1: {f1:.2f}')
    return y_true, probs, metrics


def plot_roc(y_true, probs):
    import numpy as np
    y_true_arr = np.asarray(y_true).ravel()
    probs_arr = np.asarray(probs).ravel()

    auc = roc_auc_score(y_true_arr, probs_arr)
    fpr, tpr, _ = roc_curve(y_true_arr, probs_arr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    out_path = 'result/roc_curve.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'AUC: {auc:.2f}')
    print(f'ROC curve saved to {out_path}')


def main():
    train_loader, test_loader = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)

    try:
        state = torch.load('result/model.pt', map_location=device)
        model.load_state_dict(state)
        print('Loaded model weights from result/model.pt')
    except Exception as e:
        print('Warning: could not load model weights from result/model.pt:', e)
        print('Proceeding with randomly initialized model for evaluation.')

    y_true, probs, metrics = evaluate_classification(model, test_loader, device)

    plot_roc(y_true, probs)

    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    roc = roc_auc_score(y_true, probs)

    df = pd.DataFrame([{'Precision': round(precision, 2), 'Recall': round(recall,2), 'F1': round(f1,2), 'ROC': round(roc,2)}])
    df.to_csv('result/precision_recall_f1_score.csv', index=False)

if __name__ == '__main__':
    main()

