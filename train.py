"""
train.py
================================================================
Ana orkestrasyon dosyası / main entry point.
data_loader + models + evaluate modüllerini birleştirir.

İki fazlı boru hattı / Two-phase pipeline:
  FAZ 1 (opsiyonel) - GAN augmentation:
    Her nadir aritmi sınıfı (S, V, F, Q) için ayrı bir 1D-LSTM GAN eğitilir,
    sentetik atımlar üretilip eğitim setine eklenir. Amaç: azınlık sınıfları
    sadece KOPYALAYARAK değil, ÇEŞİTLİ gerçekçi örneklerle dengelemek.
  FAZ 2 - Sınıflandırma:
    Dengelenmiş+sentetik veriyle CNN, CNN+LSTM, CNN+LSTM+Attention eğitilir;
    her biri test edilir ve sonunda ENSEMBLE (soft voting) ile birleştirilir.

Colab kullanımı / usage:
    !python train.py                 # tam boru hattı (GAN dahil, yavaş)
    !python train.py --no-gan        # sadece sınıflandırma (hızlı)
    !python train.py --gan-epochs 1000 --clf-epochs 20
================================================================
"""

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import data_loader as dl
import models as M
import evaluate as E


# ----------------------------------------------------------------------
# Yapılandırma / Configuration
# ----------------------------------------------------------------------
class Config:
    seed = 2021
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    signal_length = dl.SIGNAL_LENGTH
    num_classes = dl.NUM_CLASSES
    batch_size = 96
    # --- GAN fazı ---
    run_gan = True
    gan_epochs = 3000           # kaynak notebook degeri; 
    gan_lr = 2e-4
    minority_classes = [1, 2, 3, 4]   # N (0) cogunluk; digerleri nadir
    synthetic_per_class = 5000
    # --- Sınıflandırıcı fazı ---
    clf_epochs = 30
    clf_lr = 1e-3
    ckpt_dir = "checkpoints"


def seed_everything(seed):
    """Tekrarlanabilirlik / reproducibility."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# FAZ 1: GAN eğitimi / GAN training
# ----------------------------------------------------------------------
def train_gan(generator, discriminator, dataloader, num_epochs, device,
              lr=2e-4, signal_length=dl.SIGNAL_LENGTH, log_every=300):
    """
    Standart GAN minimax eğitimi (Goodfellow 2014).
      D: max  log D(x) + log(1 - D(G(z)))   -> gerçeği gerçek, sahteyi sahte de
      G: max  log D(G(z))                   -> Discriminator'ı kandır

    betas=(0.5, 0.999): GAN'lerde momentumu düşürmek (0.5) eğitimi
    stabilize eden yaygın bir DCGAN pratiğidir.
    """
    netG, netD = generator.to(device), discriminator.to(device)
    optD = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optG = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    d_losses, g_losses = [], []

    for epoch in range(num_epochs):
        errD = errG = 0.0
        for data, _ in dataloader:
            real = data.to(device)
            bs = real.size(0)
            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)

            # --- Discriminator ---
            netD.zero_grad()
            lossD_real = criterion(netD(real), real_lbl)
            noise = torch.randn(bs, 1, signal_length, device=device)
            fake = netG(noise)
            lossD_fake = criterion(netD(fake.detach()), fake_lbl)  # detach: G'ye gradyan gitmesin
            lossD = lossD_real + lossD_fake
            lossD.backward(); optD.step()

            # --- Generator ---
            netG.zero_grad()
            lossG = criterion(netD(fake), real_lbl)   # sahteyi "gerçek" etiketlemeye çalış
            lossG.backward(); optG.step()
            errD, errG = lossD.item(), lossG.item()

        d_losses.append(errD); g_losses.append(errG)
        if epoch % log_every == 0:
            print(f"  [GAN] epoch {epoch} | D {errD:.4f} | G {errG:.4f} | {time.strftime('%H:%M:%S')}")
    return netG, (d_losses, g_losses)


@torch.no_grad()
def generate_synthetic(generator, n_samples, device,
                       signal_length=dl.SIGNAL_LENGTH, batch=256):
    """Eğitilmiş Generator'dan n_samples adet sentetik atım (n, 187) üretir."""
    generator.eval()
    out, remaining = [], n_samples
    while remaining > 0:
        b = min(batch, remaining)
        z = torch.randn(b, 1, signal_length, device=device)
        out.append(generator(z).squeeze(1).cpu().numpy())
        remaining -= b
    return np.concatenate(out, axis=0)[:n_samples]


def run_gan_augmentation(cfg):
    """
    Her azınlık sınıfı için GAN eğitir, sentetik atım üretir, diske kaydeder.
    Döner: {class_id: synthetic_array} sözlüğü.
    """
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    synthetic = {}
    for cls in cfg.minority_classes:
        print(f"[GAN] sinif {cls} ({dl.ID_TO_LABEL[cls]}) icin egitim...")
        loader = dl.get_gan_dataloader(cls, batch_size=cfg.batch_size)
        G = M.Generator(signal_length=cfg.signal_length)
        D = M.Discriminator(signal_length=cfg.signal_length)
        G, _ = train_gan(G, D, loader, cfg.gan_epochs, cfg.device,
                         lr=cfg.gan_lr, signal_length=cfg.signal_length)
        torch.save(G.state_dict(), os.path.join(cfg.ckpt_dir, f"generator_class{cls}.pth"))
        syn = generate_synthetic(G, cfg.synthetic_per_class, cfg.device, cfg.signal_length)
        synthetic[cls] = syn
        np.save(os.path.join(cfg.ckpt_dir, f"synthetic_class{cls}.npy"), syn)
    return synthetic


# ----------------------------------------------------------------------
# Eğitim verisi (denge + sentetik) / augmented loaders
# ----------------------------------------------------------------------
def build_augmented_loaders(cfg, synthetic=None):
    """
    Önemli: önce validation'ı ham train'den ayır (sızıntısız), SONRA train'i
    dengele ve sentetik atımları SADECE eğitim setine ekle. Test seti
    (resmi MIT-BIH test CSV'si) dokunulmadan kalır.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    train_df, test_df = dl.load_mitbih()
    base_X = train_df.iloc[:, :cfg.signal_length].values
    base_y = train_df[187].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        base_X, base_y, test_size=0.15, random_state=cfg.seed, stratify=base_y)

    # dengeleme (yalnizca train)
    tr_df = pd.DataFrame(np.column_stack([X_tr, y_tr])); tr_df[187] = tr_df[187].astype(int)
    tr_df = dl.balance_dataframe(tr_df, seed=cfg.seed)
    X_tr = tr_df.iloc[:, :cfg.signal_length].values
    y_tr = tr_df[187].values

    # GAN sentetik atımları ekle
    if synthetic:
        extra_X = np.concatenate(list(synthetic.values()), axis=0)
        extra_y = np.concatenate([np.full(len(v), c) for c, v in synthetic.items()])
        X_tr = np.concatenate([X_tr, extra_X], axis=0)
        y_tr = np.concatenate([y_tr, extra_y], axis=0)
        print(f"[DATA] +{len(extra_y)} sentetik atim eklendi.")

    tr = DataLoader(dl.ECGDataset(X_tr, y_tr, augment=True),
                    batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    vl = DataLoader(dl.ECGDataset(X_val, y_val, augment=False),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    te = DataLoader(dl.ECGDataset(test_df.iloc[:, :cfg.signal_length].values,
                                  test_df[187].values, augment=False),
                    batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    return tr, vl, te


# ----------------------------------------------------------------------
# FAZ 2: Sınıflandırıcı eğitimi / classifier training
# ----------------------------------------------------------------------
def train_classifier(model, train_loader, val_loader, num_epochs, lr, device,
                     ckpt_path="best.pth"):
    """
    AdamW + CosineAnnealingLR + CrossEntropyLoss ile eğitim.
    - CrossEntropyLoss: logit bekler (model softmax UYGULAMAZ; bkz. models.py).
    - CosineAnnealingLR: öğrenme oranını kosinüs eğrisiyle yumuşakça düşürür,
      sona doğru ince ayar yapıp daha iyi minimuma oturtur.
    - En düşük validation loss'a göre en iyi modeli kaydeder (early-stopping
      mantığı / checkpointing). val seti gerçek dağılımı yansıttığından
      seçim klinik olarak anlamlıdır.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-6)
    best_val = float("inf")

    for epoch in range(num_epochs):
        # --- train ---
        model.train(); meter = E.Meter(); meter.init_metrics()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            meter.update(logits, target, loss.item())
        tr_m = meter.get_metrics()
        # --- validation ---
        model.eval(); vmeter = E.Meter(); vmeter.init_metrics()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                loss = criterion(logits, target)
                vmeter.update(logits, target, loss.item())
        val_m = vmeter.get_metrics()
        scheduler.step()
        print(f"epoch {epoch+1}/{num_epochs} | "
              f"train loss {tr_m['loss']:.4f} f1 {tr_m['f1']:.4f} | "
              f"val loss {val_m['loss']:.4f} f1 {val_m['f1']:.4f} recall {val_m['recall']:.4f}")
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


# ----------------------------------------------------------------------
# Orkestrasyon / main
# ----------------------------------------------------------------------
def main(cfg):
    seed_everything(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    print(f"Device: {cfg.device}")

    # FAZ 1
    synthetic = run_gan_augmentation(cfg) if cfg.run_gan else None
    # Veri
    train_loader, val_loader, test_loader = build_augmented_loaders(cfg, synthetic)

    # FAZ 2: üç mimari
    specs = {
        "cnn": M.CNN(num_classes=cfg.num_classes),
        "cnn_lstm": M.CNNLSTM(num_classes=cfg.num_classes),
        "cnn_lstm_attn": M.CNNLSTMAttention(num_classes=cfg.num_classes),
    }
    trained = []
    for name, net in specs.items():
        print(f"\n=== {name} egitiliyor ===")
        ckpt = os.path.join(cfg.ckpt_dir, f"{name}.pth")
        m = train_classifier(net, train_loader, val_loader, cfg.clf_epochs,
                             cfg.clf_lr, cfg.device, ckpt)
        trained.append(m)
        yp, yt = E.predict(m, test_loader, cfg.device)
        print(f"--- {name} TEST raporu ---")
        E.print_report(yt, yp)
        E.plot_confusion_matrix(yt, yp, save_path=os.path.join(cfg.ckpt_dir, f"cm_{name}.png"))

    # Ensemble
    print("\n=== ENSEMBLE (soft voting) ===")
    yp, yt = E.predict_ensemble(trained, test_loader, cfg.device)
    E.print_report(yt, yp)
    E.plot_confusion_matrix(yt, yp, save_path=os.path.join(cfg.ckpt_dir, "cm_ensemble.png"))

    # Yorumlanabilirlik: attention örneği
    sample, lbl = next(iter(test_loader))
    path, pred = E.plot_attention(trained[2], sample[0], cfg.device,
                                  save_path=os.path.join(cfg.ckpt_dir, "attention_example.png"),
                                  label=int(lbl[0]))
    print(f"[VIZ] attention gorseli kaydedildi: {path} (tahmin={dl.ID_TO_LABEL[pred]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-gan", action="store_true", help="GAN augmentation'i atla (hizli mod)")
    p.add_argument("--gan-epochs", type=int, default=Config.gan_epochs)
    p.add_argument("--clf-epochs", type=int, default=Config.clf_epochs)
    args = p.parse_args()
    cfg = Config()
    cfg.run_gan = not args.no_gan
    cfg.gan_epochs = args.gan_epochs
    cfg.clf_epochs = args.clf_epochs
    main(cfg)
