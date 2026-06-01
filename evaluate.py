"""
evaluate.py
================================================================
Tıbbi değerlendirme metrikleri ve görselleştirme.
Medical evaluation metrics & visualization.

Neden accuracy TEK BAŞINA yeterli değil? / Why accuracy alone is misleading:
- Sınıf dengesizliği nedeniyle "her şeye Normal de" diyen bir model %80+
  accuracy alabilir ama hayati aritmileri kaçırır. Bu yüzden sınıf-bazlı
  (per-class) ve makro-ortalamalı metriklere bakıyoruz.

Klinik metrik sözlüğü / Clinical metric glossary:
- Recall (= Sensitivity / Duyarlılık): "Gerçek aritmilerin kaçını yakaladık?"
  EN KRİTİK metrik. Düşük recall = yüksek FALSE NEGATIVE = kaçırılan hasta.
- Precision (= PPV / Pozitif Kestirim Değeri): "Aritmi dediklerimizin kaçı
  gerçekten aritmi?" Düşük precision = gereksiz alarm (false positive).
- F1-score: Precision ve recall'un harmonik ortalaması; dengesiz veride
  accuracy'den çok daha bilgilendirici.
- Confusion matrix: hangi aritminin hangisiyle KARIŞTIĞINI gösterir
  (örn. V <-> F karışması klinik olarak önemli bir hata türüdür).
================================================================
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Colab'de başsız (headless) çizim için
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)

LABELS = ["N", "S", "V", "F", "Q"]


# ----------------------------------------------------------------------
# Eğitim sırasında metrik takibi / running-metric tracker
# ----------------------------------------------------------------------
class Meter:
    """
    Her batch'te metrikleri biriktirir; epoch sonunda ortalamasını verir.
    Ayrıca confusion matrix'i kümülatif olarak tutar.
    """
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.init_metrics()

    def init_metrics(self):
        self.metrics = {"loss": 0.0, "accuracy": 0.0, "f1": 0.0,
                        "precision": 0.0, "recall": 0.0}
        self.confusion = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self.steps = 0

    def update(self, logits, target, loss):
        # logits -> tahmin sınıfı / predicted class
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y = target.detach().cpu().numpy()
        self.metrics["loss"] += float(loss)
        self.metrics["accuracy"] += accuracy_score(y, pred)
        # macro: her sınıfa eşit ağırlık -> azınlık sınıfları "görünür" kılar
        self.metrics["f1"] += f1_score(y, pred, average="macro", zero_division=0)
        self.metrics["precision"] += precision_score(y, pred, average="macro", zero_division=0)
        self.metrics["recall"] += recall_score(y, pred, average="macro", zero_division=0)
        for t, p in zip(y, pred):
            self.confusion[t, p] += 1
        self.steps += 1

    def get_metrics(self):
        s = max(self.steps, 1)
        return {k: v / s for k, v in self.metrics.items()}

    def get_confusion_matrix(self):
        return self.confusion


# ----------------------------------------------------------------------
# Çıkarım / Inference
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(model, dataloader, device, return_probs=False):
    """Tüm veri kümesi üzerinde tahmin üretir. return_probs=True ise softmax olasılıkları döner."""
    model.eval()
    preds, gts = [], []
    for data, target in dataloader:
        data = data.to(device)
        logits = model(data)
        probs = torch.softmax(logits, dim=1)   # logit -> olasılık
        preds.append(probs.cpu() if return_probs else torch.argmax(probs, 1).cpu())
        gts.append(target.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(gts).numpy()
    return y_pred, y_true


@torch.no_grad()
def predict_ensemble(models_list, dataloader, device):
    """
    Birden çok modelin SOFTMAX olasılıklarını ortalar (soft voting).
    Klinik gerekçe: CNN (morfoloji), LSTM (ritim) ve Attention (odak)
    farklı hatalar yapar; ortalamaları alındığında hatalar birbirini
    dengeler -> daha güvenilir tanı. (Kaynak notebook'taki ensemble mantığı.)
    """
    probs_sum, y_true = None, None
    for m in models_list:
        p, y = predict(m, dataloader, device, return_probs=True)
        probs_sum = p if probs_sum is None else probs_sum + p
        y_true = y
    y_pred = np.argmax(probs_sum / len(models_list), axis=1)
    return y_pred, y_true


# ----------------------------------------------------------------------
# Raporlama / Reporting
# ----------------------------------------------------------------------
def print_report(y_true, y_pred, labels=LABELS):
    """Sınıf-bazlı precision/recall/F1 tablosu (klinik olarak en önemli çıktı)."""
    print(classification_report(y_true, y_pred, target_names=labels,
                                digits=4, zero_division=0))
    return classification_report(y_true, y_pred, target_names=labels,
                                 output_dict=True, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, labels=LABELS, normalize=True,
                          save_path="confusion_matrix.png"):
    """
    Confusion matrix çizer. normalize=True iken satır (gerçek sınıf) bazında
    oranlar gösterilir -> "V atımlarının %X'ini F sandık" gibi klinik hata
    örüntülerini okumak kolaylaşır.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (Actual)")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    fmt = ".2f" if normalize else "d"
    thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ----------------------------------------------------------------------
# YORUMLANABİLİRLİK: Attention görselleştirme / interpretability
# ----------------------------------------------------------------------
@torch.no_grad()
def plot_attention(model, signal, device, save_path="attention.png", label=None):
    """
    Attention ağırlıklarını EKG sinyalinin üstüne renk olarak bindirir.
    Overlays attention weights on the raw ECG beat.

 
    - Sıcak (parlak) renkler modelin tanı için "baktığı" bölgelerdir.
    - İyi eğitilmiş modelde bu odak QRS kompleksi / R-tepesi civarında
      yoğunlaşır -> modelin kararını klinik olarak anlamlı bir bölgeye
      dayandırdığını KANITLAR (kara kutu değil).
    - Attention 46 zaman adımı üzerinde üretilir; sinyalin 187 örneğine
      lineer interpolasyon ile geri ölçeklenir.

    'model' CNNLSTMAttention olmalı (forward(x, return_attn=True) destekler).
    """
    model.eval()
    if signal.dim() == 2:
        signal = signal.unsqueeze(0)        # (1, 1, 187)
    signal = signal.to(device)
    logits, weights = model(signal, return_attn=True)
    pred = int(torch.argmax(logits, dim=1).item())
    sig = signal.squeeze().cpu().numpy()
    w = weights.squeeze().cpu().numpy()
    # 46 -> 187 geri ölçekleme / upsample attention to signal length
    w_up = np.interp(np.linspace(0, 1, len(sig)),
                     np.linspace(0, 1, len(w)), w)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sig, color="black", lw=1.2, label="ECG beat")
    ax.scatter(range(len(sig)), sig, c=w_up, cmap="hot", s=18)
    title = f"Attention over heartbeat | predicted = {LABELS[pred]}"
    if label is not None:
        title += f" | true = {LABELS[int(label)]}"
    ax.set_title(title); ax.set_xlabel("Sample (time)"); ax.set_ylabel("Amplitude")
    sm = plt.cm.ScalarMappable(cmap="hot"); sm.set_array(w_up)
    fig.colorbar(sm, ax=ax, label="attention weight")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
    return save_path, pred
