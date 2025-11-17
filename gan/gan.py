import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ============================================================
# this is the data loading and preprocessing section
# ============================================================
def load_data(path="unsupervised/cleaned_MS_data.csv", seq_len=100, label_filter=0):
    df = pd.read_csv(path)
    df = df[df["label"] == label_filter].reset_index(drop=True)
    print(f"Filtered data: {len(df)} samples with label={label_filter}")


    def to_array(x):
        if isinstance(x, str):
            return np.array(eval(x), dtype=np.float32)
        elif isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=np.float32)
        else:
            return np.array([float(x)], dtype=np.float32)

    seqs = [to_array(x) for x in df["steps"]]

    normed = []
    for arr in seqs:

        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max > a_min:
            arr_n = 2.0 * (arr - a_min) / (a_max - a_min) - 1.0
        else:

            arr_n = np.zeros_like(arr, dtype=np.float32)


        if len(arr_n) >= seq_len:
            arr_n = arr_n[:seq_len]
        else:
            pad = np.zeros(seq_len - len(arr_n), dtype=np.float32)
            arr_n = np.concatenate([arr_n, pad], axis=0)

        normed.append(arr_n)

    X = torch.tensor(np.stack(normed)[:, None, 0:1], dtype=torch.float32)
    print(f"Tensor shape = {X.shape} (batch, channel, seq_len=1)")
    return X

# ============================================================
# this is the generator model
# ============================================================
class Generator(nn.Module):
    def __init__(self, noise_dim=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z).unsqueeze(1)   # [B, 1, seq_len]


# ============================================================
# this is the discriminator model
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, seq_len=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(16 * seq_len, 1),  # ← 改这里
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# ============================================================
# this is the GAN training loop
# ============================================================
def train_gan(
    X,
    noise_dim=64,
    batch_size=64,
    lr=2e-4,
    betas=(0.5, 0.999),
    epochs=100
):
    """Train GAN and return trained Generator, Discriminator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    train_loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True, drop_last=True)
    seq_len = X.shape[-1]
    G = Generator(noise_dim=noise_dim).to(device)
    D = Discriminator(seq_len=seq_len).to(device)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    fixed_noise = torch.randn(16, noise_dim, device=device)

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        for step, (real_batch,) in enumerate(train_loader):
            real = real_batch.to(device)
            b = real.size(0)
            valid = torch.ones(b, 1, device=device)
            fake_ = torch.zeros(b, 1, device=device)

            # ---- Train Discriminator ----
            z = torch.randn(b, noise_dim, device=device)
            fake = G(z).detach()
            opt_D.zero_grad()
            loss_real = criterion(D(real), valid)
            loss_fake = criterion(D(fake), fake_)
            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ---- Train Generator ----
            z = torch.randn(b, noise_dim, device=device)
            fake = G(z)
            opt_G.zero_grad()
            loss_G = criterion(D(fake), valid)
            loss_G.backward()
            opt_G.step()

            if step % 20 == 0:
                print(f"Epoch {epoch:03d}/{epochs} | Step {step:04d} | "
                      f"D_loss={loss_D.item():.4f} | G_loss={loss_G.item():.4f}")

        # after each epoch, visualize generated samples
        save_generated_outputs(G, fixed_noise, epoch)

    print("Training complete.")
    return G, D


# ============================================================
# this is the saving function (to PNG, NPY, and CSV)
# ============================================================
def save_generated_outputs(G, fixed_noise, epoch, save_prefix="gan/generated/epoch_"):
    import pathlib
    pathlib.Path("gan/generated").mkdir(parents=True, exist_ok=True)

    G.eval()
    with torch.no_grad():
        fake_vis = G(fixed_noise).cpu().numpy().reshape(len(fixed_noise), -1)  # [N, seq_len]

    # save histogram instead of line plot (since single-value)
    plt.figure(figsize=(6, 4))
    plt.hist(fake_vis.flatten(), bins=20, alpha=0.7)
    plt.title(f"Generated footsteps (epoch {epoch})")
    plt.xlabel("normalized value [-1,1]")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}{epoch:03d}.png", dpi=120)
    plt.close()

    # save npy & csv
    np.save(f"{save_prefix}{epoch:03d}.npy", fake_vis)
    pd.DataFrame(fake_vis, columns=["footstep"]).to_csv(f"{save_prefix}{epoch:03d}.csv", index=False)
    print(f"Saved results to {save_prefix}{epoch:03d}.csv")


# ============================================================
# this is the main controller (entry point)
# ============================================================
def main():
    # Step 1: load data
    X = load_data("unsupervised/cleaned_MS_data.csv")

    # Step 2: train GAN
    G, D = train_gan(X, noise_dim=64, batch_size=64, epochs=100)

    # Step 3: optionally test generator on new noise
    z = torch.randn(1, 64)
    fake_sample = G(z).detach().numpy().squeeze()
    print("Example generated sequence:", fake_sample)


# ============================================================
# this is the standard Python entry point
# ============================================================
if __name__ == "__main__":
    main()
