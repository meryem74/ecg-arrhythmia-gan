"""
data_loader.py
================================================================
EKG (ECG) veri hazırlama modülü / ECG data preparation module.

Tıbbi arka plan / Medical background:
- Veri setindeki her SATIR, R-tepesine (R-peak) göre hizalanmış ve
  sabit 187 örneğe yeniden örneklenmiş TEK bir kalp atımıdır (single heartbeat).
  Each row is one R-peak-aligned heartbeat resampled to a fixed 187 samples.
- Bir atım, PQRST kompleksini içerir: P dalgası (atriyal depolarizasyon),
  QRS kompleksi (ventriküler depolarizasyon - en keskin/baskın bileşen),
  T dalgası (ventriküler repolarizasyon).
  A beat contains the PQRST complex: P wave, QRS complex (sharpest peak), T wave.
- Sütun 0..186 = sinyal genliği (amplitude), sütun 187 = sınıf etiketi (AAMI).

Sınıflar / Classes (AAMI standardı):
    0 = N  -> Normal beat (+ dal blokları / bundle branch blocks)
    1 = S  -> Supraventricular ectopic (örn. atriyal prematüre / APC)
    2 = V  -> Ventricular ectopic (PVC - prematüre ventriküler kontraksiyon)
    3 = F  -> Fusion beat (ventriküler + normal füzyon)
    4 = Q  -> Unknown / Paced (pacemaker atımları, sınıflandırılamayan)
================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ----------------------------------------------------------------------
# Sabitler / Constants
# ----------------------------------------------------------------------
DATA_DIR = "data"  # Colab'de veriyi indirdiğin klasör / your downloaded data folder
MITBIH_TRAIN = os.path.join(DATA_DIR, "mitbih_train.csv")
MITBIH_TEST = os.path.join(DATA_DIR, "mitbih_test.csv")
PTBDB_NORMAL = os.path.join(DATA_DIR, "ptbdb_normal.csv")
PTBDB_ABNORMAL = os.path.join(DATA_DIR, "ptbdb_abnormal.csv")

SIGNAL_LENGTH = 187   # Bir atımdaki örnek sayısı / samples per heartbeat
NUM_CLASSES = 5
SEED = 2021


ID_TO_LABEL = {
    0: "N - Normal beat",
    1: "S - Supraventricular ectopic",
    2: "V - Ventricular ectopic (PVC)",
    3: "F - Fusion beat",
    4: "Q - Unknown / Paced",
}


# ----------------------------------------------------------------------
# Augmentation: Gaussian gürültü / Gaussian noise
# ----------------------------------------------------------------------
def add_gaussian_noise(signal: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Sinyale küçük Gauss gürültüsü ekler.
    Adds small Gaussian noise to the signal.

    Tıbbi gerekçe / Rationale:
    - Gerçek EKG kayıtları taban hattı kayması (baseline wander), kas
      artefaktı ve elektrot gürültüsü içerir. Eğitimde kontrollü gürültü
      eklemek modeli bu artefaktlara karşı DAYANIKLI (robust) yapar ve
      ezberlemeyi (overfitting) azaltır; PQRST'nin gerçek morfolojisini
      öğrenmeye zorlar.
    - NOT: Kaynak notebook sigma=0.5 kullanmıştı; sinyaller [0,1] aralığına
      normalize olduğu için bu çok agresiftir ve QRS'i boğabilir. Daha
      gerçekçi/savunulabilir bir varsayılan olarak sigma=0.05 seçtik.
    - SADECE eğitim setine uygulanır; validation/test'e ASLA dokunmayız
      (sızıntısız, dürüst değerlendirme için).
    """
    return signal + np.random.normal(0.0, sigma, size=signal.shape)


# ----------------------------------------------------------------------
# CSV okuma / Loading
# ----------------------------------------------------------------------
def load_mitbih():
    """MIT-BIH train/test CSV'lerini okur. Son sütun (187) tamsayı etikettir."""
    if not os.path.exists(MITBIH_TRAIN):
        raise FileNotFoundError(
            f"{MITBIH_TRAIN} bulunamadi. 'data/' klasor adlarini kontrol et."
        )
    tr = pd.read_csv(MITBIH_TRAIN, header=None)
    te = pd.read_csv(MITBIH_TEST, header=None)
    tr[187] = tr[187].astype(int)
    te[187] = te[187].astype(int)
    return tr, te


def load_ptbdb():
    """
    PTBDB (ikili: normal vs abnormal MI) okur ve birleştirir.
    Binary myocardial-infarction dataset; opsiyonel ikinci görev için.
    Etiket: 0 = normal, 1 = abnormal (anormal/MI).
    """
    normal = pd.read_csv(PTBDB_NORMAL, header=None)
    abnormal = pd.read_csv(PTBDB_ABNORMAL, header=None)
    df = pd.concat([normal, abnormal], axis=0).reset_index(drop=True)
    df[187] = df[187].astype(int)
    return df


# ----------------------------------------------------------------------
# Sınıf dengeleme / Class balancing
# ----------------------------------------------------------------------
def balance_dataframe(df: pd.DataFrame, target_per_class: int = 20000, seed: int = SEED):
    """
    Her sınıfı 'target_per_class' örneğe eşitler.
    Balances every class to 'target_per_class' samples.

    Tıbbi gerekçe / Why this matters clinically:
    - Ham MIT-BIH'te N (Normal) atımlar ~%82'dir; V, S, F, Q nadirdir.
      Dengesiz veriyle eğitilen model "her şeye Normal de" diyerek yüksek
      accuracy alır ama KLİNİK OLARAK KRİTİK aritmileri (örn. PVC) kaçırır
      -> yüksek false-negative, hasta için tehlikeli.
    - Çoğunluk sınıf (N) alt-örneklenir (downsample), azınlıklar üst-örneklenir
      (upsample). İdeal olarak azınlıklar GAN ile sentetik üretilir (bkz. train.py),
      böylece tekrar yerine ÇEŞİTLİLİK kazanırız.
    """
    frames = []
    for cls in range(NUM_CLASSES):
        sub = df[df[187] == cls]
        if len(sub) == 0:
            continue
        # Sınıf hedeften küçükse tekrar ile çoğalt (replace=True), büyükse seç.
        replace = len(sub) < target_per_class
        frames.append(
            resample(sub, replace=replace, n_samples=target_per_class,
                     random_state=seed + cls)
        )
    out = pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# PyTorch Dataset
# ----------------------------------------------------------------------
class ECGDataset(Dataset):
    """
    Tek bir atımı (1, 187) tensörü olarak döndürür.
    Returns one heartbeat as a (1, 187) tensor -> (channel=1, length=187).

    Kanal boyutu = 1 çünkü tek-kanallı (single-lead) sinyal; Conv1d için
    (batch, channels, length) formatı gerekir. The leading 1 is the input
    channel expected by nn.Conv1d.
    """
    def __init__(self, signals: np.ndarray, labels: np.ndarray,
                 augment: bool = False, sigma: float = 0.05):
        self.signals = signals.astype("float32")
        self.labels = labels.astype("int64")
        self.augment = augment   # sadece train'de True
        self.sigma = sigma

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx].copy()
        if self.augment:
            sig = add_gaussian_noise(sig, self.sigma)
        # (187,) -> (1, 187): Conv1d'nin beklediği kanal eksenini ekliyoruz.
        sig = torch.FloatTensor(sig).unsqueeze(0)
        target = torch.tensor(self.labels[idx], dtype=torch.long)
        return sig, target


# ----------------------------------------------------------------------
# Sınıflandırıcı için DataLoader'lar
# ----------------------------------------------------------------------
def get_classifier_dataloaders(batch_size: int = 96, balance: bool = True,
                               augment_train: bool = True, val_size: float = 0.15,
                               seed: int = SEED):
    """
    Sınıflandırma için (train, val, test) DataLoader üçlüsü döndürür.

    Önemli / Important:
    - Dengeleme ve gürültü SADECE eğitim setine uygulanır.
    - Validation, dengelenmemiş train'den stratify ile ayrılır (gerçek
      dağılımı temsil etsin diye); test seti dokunulmadan kalır.
    - Resmi MIT-BIH test CSV'si değerlendirme için ayrı tutulur.
    """
    train_df, test_df = load_mitbih()

    # Validation'i ham (dengesiz) train'den ayir -> gercek dagilimi yansitsin.
    base_X = train_df.iloc[:, :SIGNAL_LENGTH].values
    base_y = train_df[187].values
    X_tr_raw, X_val, y_tr_raw, y_val = train_test_split(
        base_X, base_y, test_size=val_size, random_state=seed, stratify=base_y
    )

    # Dengelemeyi YALNIZCA train kismina uygula (sizinti olmasin diye
    # val ayrildiktan SONRA).
    if balance:
        tr_df = pd.DataFrame(np.column_stack([X_tr_raw, y_tr_raw]))
        tr_df[187] = tr_df[187].astype(int)
        tr_df = balance_dataframe(tr_df, seed=seed)
        X_tr = tr_df.iloc[:, :SIGNAL_LENGTH].values
        y_tr = tr_df[187].values
    else:
        X_tr, y_tr = X_tr_raw, y_tr_raw

    X_te = test_df.iloc[:, :SIGNAL_LENGTH].values
    y_te = test_df[187].values

    ds_tr = ECGDataset(X_tr, y_tr, augment=augment_train)
    ds_val = ECGDataset(X_val, y_val, augment=False)
    ds_te = ECGDataset(X_te, y_te, augment=False)

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------
# GAN için tek-sınıf DataLoader
# ----------------------------------------------------------------------
def get_gan_dataloader(class_id: int, batch_size: int = 96, seed: int = SEED):
    """
    GAN'i TEK bir sınıf üzerinde eğitmek için o sınıfın gerçek atımlarını verir.
    Returns only one class's real beats to train a class-specific GAN.

    Neden tek sınıf? / Why per-class:
    - Her aritmi sınıfının kendine özgü morfolojisi var (örn. PVC'de geniş,
      bizar QRS; füzyonda karma şekil). GAN'i sınıf-özel eğitmek, o sınıfa
      ait gerçekçi sentetik atımlar üretip dengelemeyi sağlar.
    - drop_last=True: GAN eğitiminde sabit batch boyutu daha kararlı.
    """
    train_df, _ = load_mitbih()
    sub = train_df[train_df[187] == class_id]
    if len(sub) == 0:
        raise ValueError(f"class_id={class_id} icin ornek bulunamadi.")
    X = sub.iloc[:, :SIGNAL_LENGTH].values
    y = sub[187].values
    ds = ECGDataset(X, y, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=2, drop_last=True)


if __name__ == "__main__":
    # Hizli kendi-kendine test / quick self-test
    tr, te = load_mitbih()
    print("MIT-BIH train shape:", tr.shape, "| test shape:", te.shape)
    print("Ham sinif dagilimi / raw distribution:\n", tr[187].value_counts().sort_index())
    trl, vl, tel = get_classifier_dataloaders(batch_size=32)
    xb, yb = next(iter(trl))
    print("Batch sinyal:", xb.shape, "| etiket:", yb.shape)  # beklenen: [32,1,187], [32]
