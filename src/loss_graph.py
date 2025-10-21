import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/zefanhuang/Downloads/lab_demo/src')

from modeling import load_data, train_model

def main():
    train_loader, test_loader = load_data()
    df = train_model(train_loader, test_loader)


    plt.figure(figsize=(10,4))


    plt.subplot(1,2,1)
    plt.plot(df['epoch'], df['loss'], marker='o', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)


    plt.subplot(1,2,2)
    plt.plot(df['epoch'], df['accuracy'], marker='s', color='orange', label='Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('result/training_curves.png', dpi=500)
    plt.show()
    print("Curves saved to result/training_curves.png")

if __name__ == "__main__":
    main()