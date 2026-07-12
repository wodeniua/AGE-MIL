# datasets/xiangya_dual.py
import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_clinical_excel(excel_path: str):
    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]
    df["编号"] = df["编号"].astype(str).str.strip()

    feat_cols = ["年龄（穿刺时）", "穿刺前TPSA", "临床T分期", "活检ISUP分组", "系统活检Gleanson评分", "阳性核心百分比；；"]

    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat_cols] = df[feat_cols].fillna(0.0)

    clinical_dict = {}
    for _, r in df.iterrows():
        clinical_dict[r["编号"]] = r[feat_cols].to_numpy(dtype=np.float32)

    return clinical_dict, len(feat_cols)

# ----------------------------
# H5 Reader (features + coords)
# ----------------------------
def read_h5_features_coords(path: str):
    """
    Read (features, coords) from an .h5 file.
    Keys: ['coords', 'features'].
    - TRITAN features: usually (D,) or (1,D); occasionally (N,D)
    - CONCH features : usually (N,D); occasionally (D,)
    """
    with h5py.File(path, "r") as f:
        if "features" not in f:
            raise KeyError(f"[H5] {path} missing key 'features'")
        feats = f["features"][:]
        coords = f["coords"][:] if "coords" in f else None
    return feats, coords


def split_dataset_xiangya_dual(conf):
    """
    Xiangya patient-level split builder for dual-stream.
    CSV must have columns: patient_id, filename, label, dataset (train/val/test)

    Returns:
      train_split, train_names, val_split, val_names, test_split, test_names

    split dict structure:
      {patient_id: {"label": int,
                    "slides": [{"tritan_path": "...", "conch_path": "..."}, ...]}}
    """
    split_file_path = f'./splits/{conf.dataset}/xiangya_labels_trainval_fold_{conf.seed}.csv'
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"The file {split_file_path} does not exist.")

    data = pd.read_csv(split_file_path)
    
    clinical_dict, clinical_dim = None, 0
    if hasattr(conf, "clinical_excel_path") and conf.clinical_excel_path:
        if os.path.exists(conf.clinical_excel_path):
            clinical_dict, clinical_dim = load_clinical_excel(conf.clinical_excel_path)

    required_columns = ["patient_id", "filename", "label", "dataset"]
    if not all(c in data.columns for c in required_columns):
        raise ValueError(f"CSV missing columns. Need {required_columns}, got {list(data.columns)}")

    if not hasattr(conf, "tritan_feat_dir") or not hasattr(conf, "conch_feat_dir"):
        raise AttributeError("conf must have: tritan_feat_dir, conch_feat_dir")

    train_names = data[data["dataset"] == "train"]["patient_id"].unique().tolist()
    val_names   = data[data["dataset"] == "val"]["patient_id"].unique().tolist()
    test_names  = data[data["dataset"] == "test"]["patient_id"].unique().tolist()

    patient_groups = data.groupby("patient_id")
    train_split, val_split, test_split = {}, {}, {}

    def _build(names, out_split):
        for patient_id in names:
            if patient_id not in patient_groups.groups:
                continue
            patient_data = patient_groups.get_group(patient_id)
            label = int(patient_data["label"].iloc[0])

            slides = []
            for _, row in patient_data.iterrows():
                fn = row["filename"]

                tritan_path = os.path.join(conf.tritan_feat_dir, fn)
                conch_path  = os.path.join(conf.conch_feat_dir,  fn)

                # both must exist
                if not os.path.exists(tritan_path):
                    continue
                if not os.path.exists(conch_path):
                    continue
                
                slides.append({"tritan_path": tritan_path, "conch_path": conch_path})

            if len(slides) == 0:
                continue
            
            clinical_vec = None
            if clinical_dict is not None:
                pid_str = str(patient_id).strip()
                clinical_vec = clinical_dict.get(pid_str, np.zeros((clinical_dim,), dtype=np.float32))

            # out_split[patient_id] = {"label": label, "slides": slides}
            out_split[patient_id] = {"label": label, "slides": slides, "clinical": clinical_vec}


    _build(train_names, train_split)
    _build(val_names, val_split)
    _build(test_names, test_split)

    return train_split, train_names, val_split, val_names, test_split, test_names


class PatientDualStreamDataset(Dataset):
    """
    Each sample = one patient with multiple slides.

    NO sampling, NO padding, NO masks.

    Outputs:
      - tritan_feats:    [N1, D]  all slide-level vectors for this patient
      - conch_patches:   [N2, D]  all patch embeddings across all slides (concatenated)
      - label:           scalar (0/1)
    """
    def __init__(self, split_dict):
        self.items = list(split_dict.items())

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _ensure_1d_slide_feat(t: torch.Tensor, tritan_path: str) -> torch.Tensor:
        """
        TRITAN is expected to be one slide vector [D].
        Accepts:
          - [D]
          - [1, D]
          - [N, D] (fallback: mean over N)
        """
        if t.ndim == 1:
            return t
        if t.ndim == 2:
            if t.shape[0] == 1:
                return t.squeeze(0)
            return t.mean(dim=0)
        raise ValueError(f"TRITAN features has unexpected shape: {tuple(t.shape)} in {tritan_path}")

    @staticmethod
    def _ensure_2d_patch_feat(p: torch.Tensor, conch_path: str) -> torch.Tensor:
        """
        CONCH is expected to be patch matrix [N, D].
        Accepts:
          - [N, D]
          - [D] (treated as 1 patch)
        """
        if p.ndim == 1:
            return p.unsqueeze(0)
        if p.ndim == 2:
            return p
        raise ValueError(f"CONCH features has unexpected shape: {tuple(p.shape)} in {conch_path}")

    def __getitem__(self, idx):
        patient_id, rec = self.items[idx]
        label = int(rec["label"])
        slides = rec["slides"]
        if rec.get("clinical", None) is None:
            clinical = torch.zeros(0, dtype=torch.float32)   # [0]
        else:
            clinical = torch.from_numpy(rec["clinical"]).float()  # [C]

        tritan_list = []
        conch_list = []

        for s in slides:
            # -------- TRITAN: slide-level vector --------
            t_np, _ = read_h5_features_coords(s["tritan_path"])
            t = torch.from_numpy(t_np).float()
            t = self._ensure_1d_slide_feat(t, s["tritan_path"])
            tritan_list.append(t)

            # -------- CONCH: patch matrix --------
            p_np, _ = read_h5_features_coords(s["conch_path"])
            p = torch.from_numpy(p_np).float()
            p = self._ensure_2d_patch_feat(p, s["conch_path"])
            conch_list.append(p)

        tritan_feats = torch.stack(tritan_list, dim=0)    # [N1, D]
        conch_patches = torch.cat(conch_list, dim=0)      # [N2, D]

        return {
            "patient_id": patient_id,
            "label": torch.tensor(label, dtype=torch.long),
            "tritan_feats": tritan_feats,      # [N1, D]
            "conch_patches": conch_patches,    # [N2, D]
            "clinical": clinical,
        }


def collate_patient_dual_list(batch):
    """
    No padding, no masks.

    Returns:
      - patient_id:    List[str]
      - label:         LongTensor [B]
      - tritan_feats:  List[FloatTensor], each [Ni_slides, D]
      - conch_patches: List[FloatTensor], each [Ni_patches, D]
      - n_slides:      LongTensor [B]
      - n_patches:     LongTensor [B]
    """
    patient_ids = [x["patient_id"] for x in batch]
    labels = torch.stack([x["label"] for x in batch], dim=0)  # [B]
    clinical_list = [x["clinical"] for x in batch]
    if clinical_list[0].numel() == 0:
        clinical = torch.zeros((len(batch), 0), dtype=torch.float32)  # [B,0]
    else:
        clinical = torch.stack(clinical_list, dim=0)  # [B,C]


    tritan_list = [x["tritan_feats"] for x in batch]
    conch_list  = [x["conch_patches"] for x in batch]

    n_slides  = torch.tensor([t.shape[0] for t in tritan_list], dtype=torch.long)  # [B]
    n_patches = torch.tensor([p.shape[0] for p in conch_list], dtype=torch.long)   # [B]

    return {
        "patient_id": patient_ids,
        "label": labels,
        "tritan_feats": tritan_list,
        "conch_patches": conch_list,
        "n_slides": n_slides,
        "n_patches": n_patches,
        "clinical": clinical,
    }


def dual_npy_feat_dataset(conf):
    """
    Keep the same call site:
      train_data, val_data, test_data = dual_npy_feat_dataset(conf)

    Returns Dataset objects (not tensors).
    """
    train_split, _, val_split, _, test_split, _ = split_dataset_xiangya_dual(conf)

    train_data = PatientDualStreamDataset(train_split)
    val_data   = PatientDualStreamDataset(val_split)
    test_data  = PatientDualStreamDataset(test_split)

    return train_data, val_data, test_data
