#!/usr/bin/env python3
"""Generate Phase2_EmbeddingFix_ReExtract.ipynb
Extracts embeddings from BOTH layers in one run:
  - v2: downsamples[2] (intermediate, 256ch) 
  - v3: downsamples[-1] (deepest, 320ch) with fixed patch aggregation
"""
import json

cells = []
def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [s], "id": f"md{len(cells)}"})
def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "source": [s], "outputs": [], "execution_count": None, "id": f"c{len(cells)}"})

md("""# Embedding Re-Extraction — v2 (intermediate) + v3 (deepest, fixed)
**No training.** Inference only. Extracts from TWO layers per fold to isolate layer-choice vs bug-fix effects.""")

code("""!pip install -q monai
print('monai installed ✅')""")

code("""import warnings
warnings.filterwarnings('ignore', message='.*Num foregrounds 0.*')
warnings.filterwarnings('ignore', message='.*non-tuple sequence.*')
warnings.filterwarnings('ignore', message='.*axcodes.*length.*')

import os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

from monai.networks.nets import DynUNet, DenseNet121
from monai.data import DataLoader, CacheDataset
from monai.inferers import sliding_window_inference
import monai.transforms as T
from monai.transforms.compose import MapTransform
from monai.utils import ensure_tuple_rep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')""")

code("""CONFIG = {
    'seg_params': {
        'spatial_dims': 3, 'in_channels': 4, 'out_channels': 3,
        'kernel_size': [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        'strides': [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        'upsample_kernel_size': [[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        'deep_supervision': True, 'deep_supr_num': 3,
        'filters': [32, 64, 128, 256, 320], 'res_block': True, 'trans_bias': True,
    },
    'patch_size': [64, 64, 64],
}

DATA_ROOT = Path('/kaggle/input/datasets/mohamedmohamed23/cyprus-proteas-brain-mets')
CKPT_ROOT = Path('/kaggle/input/datasets/mohamedmohamed23/metseg-all-checkpoints')
OUTPUT_ROOT = Path('/kaggle/working')

print(f'Data: {DATA_ROOT} (exists={DATA_ROOT.exists()})')
print(f'Ckpts: {CKPT_ROOT} (exists={CKPT_ROOT.exists()})')
for f in sorted(CKPT_ROOT.rglob('*.pth')):
    print(f'  {f.name} ({f.stat().st_size/1e6:.0f} MB)')""")

code("""class ConvertToMultiChannelBratsMetsd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            result = [
                (img == 1) | (img == 3) | (img == 2),
                (img == 1) | (img == 3),
                img == 3,
            ]
            d[key] = torch.stack(result, dim=0).float() if isinstance(img, torch.Tensor) else np.stack(result, axis=0).astype(np.float32)
        return d
print('Label converter ✅')""")

code("""SYMLINK_DIR = Path('/kaggle/working/nifti_links')

def resolve_path(root, rel):
    p = root / rel
    if p.exists(): return str(p)
    gz = str(p) + '.gz'
    if Path(gz).exists(): return gz
    nii_gz = str(p).replace('.nii.gz', '.nii_gz')
    if Path(nii_gz).exists():
        link = SYMLINK_DIR / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists(): os.symlink(nii_gz, str(link))
        return str(link)
    parent = p.parent
    if parent.exists():
        target = p.name
        for f in parent.iterdir():
            if f.name.lower() == target.lower(): return str(f)
            nii_gz_f = str(f).replace('.nii_gz', '.nii.gz')
            if Path(nii_gz_f).name.lower() == target.lower():
                link = SYMLINK_DIR / rel
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists(): os.symlink(str(f), str(link))
                return str(link)
    raise FileNotFoundError(f'Not found: {rel}')

splits_file = None
for name in ['data_splits.json', 'cv_splits_3fold.json', 'cv_splits.json']:
    candidate = DATA_ROOT / name
    if candidate.exists(): splits_file = candidate; break
if splits_file is None:
    for f in DATA_ROOT.rglob('*splits*.json'):
        splits_file = f; break

with open(splits_file) as f:
    raw_splits = json.load(f)
all_splits = raw_splits.get('3fold', raw_splits)
if 'fold_0' not in all_splits:
    all_splits = {k: v for k, v in raw_splits.items() if k.startswith('fold_')}
print(f'Splits: {splits_file.name}, folds: {list(all_splits.keys())}')

def get_all_dicts():
    all_d, seen = [], set()
    for fk in all_splits:
        for scan in all_splits[fk]['train_scans'] + all_splits[fk]['test_scans']:
            key = (scan['patient_dir'], scan['visit'])
            if key not in seen:
                seen.add(key)
                try:
                    all_d.append({
                        'image': [resolve_path(DATA_ROOT, scan['t1']),
                                  resolve_path(DATA_ROOT, scan['t1c']),
                                  resolve_path(DATA_ROOT, scan['t2']),
                                  resolve_path(DATA_ROOT, scan['fla'])],
                        'label': resolve_path(DATA_ROOT, scan['mask']),
                        'patient_dir': scan['patient_dir'],
                        'visit': scan['visit'],
                    })
                except FileNotFoundError: pass
    return all_d

val_transforms = T.Compose([
    T.LoadImaged(keys=['image', 'label']),
    T.EnsureChannelFirstd(keys=['image', 'label']),
    T.EnsureTyped(keys=['image', 'label']),
    T.Orientationd(keys=['image', 'label'], axcodes='RAS'),
    T.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=True),
    T.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ConvertToMultiChannelBratsMetsd(keys=['label']),
    T.EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
])

all_dicts = get_all_dicts()
print(f'Scans: {len(all_dicts)}')
print('Data + transforms ✅')""")

code("""def create_segmenter():
    return DynUNet(**CONFIG['seg_params'])

def load_fold_checkpoint(model, fold):
    patterns = [CKPT_ROOT / f'metseg_fold{fold}_best.pth',
                CKPT_ROOT / f'checkpoints/metseg_fold{fold}_best.pth']
    for f in CKPT_ROOT.rglob(f'*fold{fold}_best*.pth'):
        patterns.append(f)
    ckpt_path = None
    for p in patterns:
        if p.exists(): ckpt_path = p; break
    if ckpt_path is None:
        print(f'  ❌ Fold {fold} not found!'); return False
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'seg_state_dict' in ckpt:
        model.load_state_dict(ckpt['seg_state_dict'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    dice = ckpt.get('best_dice', '?')
    epoch = ckpt.get('epoch', '?')
    print(f'  ✅ Fold {fold}: {ckpt_path.name} (Dice={dice}, ep={epoch})')
    return True

seg = create_segmenter()
print(f'DynUNet: {sum(p.numel() for p in seg.parameters()):,} params')
n_ds = len(seg.downsamples)
for i in range(n_ds):
    print(f'  downsamples[{i}]: filters[{i}]={CONFIG["seg_params"]["filters"][i]}ch')
print(f'  downsamples[-1] = downsamples[{n_ds-1}]')
del seg
print('Model ✅')""")

md("""## Extraction: Both Layers in One Pass

For each scan, hooks on BOTH layers fire during `sliding_window_inference`.  
We accumulate patches separately for each layer, then average → one embedding per layer per scan.""")

code("""# ── Multi-layer extraction with fixed patch aggregation ──

def extract_both_layers(model, fold):
    \"\"\"Extract from downsamples[2] (v2) AND downsamples[-1] (v3) in one forward pass.\"\"\"
    model.eval(); model.to(device)
    
    # Storage for both hooks
    patches_layer2 = []
    patches_deepest = []
    
    def hook_layer2(module, input, output):
        feat = output[0] if isinstance(output, (list, tuple)) else output
        pooled = F.adaptive_avg_pool3d(feat.detach(), 1).flatten()
        patches_layer2.append(pooled.cpu())
    
    def hook_deepest(module, input, output):
        feat = output[0] if isinstance(output, (list, tuple)) else output
        pooled = F.adaptive_avg_pool3d(feat.detach(), 1).flatten()
        patches_deepest.append(pooled.cpu())
    
    # Register BOTH hooks
    h2 = model.downsamples[2].register_forward_hook(hook_layer2)
    hd = model.downsamples[-1].register_forward_hook(hook_deepest)
    print(f'  Hooks: downsamples[2] + downsamples[-1] ✅')
    
    all_dicts = get_all_dicts()
    print(f'  Extracting from {len(all_dicts)} scans...')
    emb_ds = CacheDataset(all_dicts, val_transforms, cache_rate=0.3, num_workers=0)
    emb_loader = DataLoader(emb_ds, batch_size=1, shuffle=False, num_workers=0)
    
    emb_v2 = {}  # intermediate layer
    emb_v3 = {}  # deepest layer (fixed)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(emb_loader, desc=f'Fold {fold}')):
            images = batch['image'].to(device)
            patient = batch['patient_dir'][0]; visit = batch['visit'][0]
            key = f'{patient}__{visit}'
            
            patches_layer2.clear()
            patches_deepest.clear()
            
            _ = sliding_window_inference(images, CONFIG['patch_size'], 4,
                                         model, overlap=0.5, mode='gaussian')
            
            if len(patches_layer2) > 0:
                emb_v2[key] = torch.stack(patches_layer2).mean(dim=0).numpy()
            if len(patches_deepest) > 0:
                emb_v3[key] = torch.stack(patches_deepest).mean(dim=0).numpy()
            
            if i < 3:
                n2 = emb_v2[key].shape[0] if key in emb_v2 else 0
                n3 = emb_v3[key].shape[0] if key in emb_v3 else 0
                print(f'    {key}: layer2={n2}-dim, deepest={n3}-dim')
    
    h2.remove(); hd.remove()
    
    # Save v2 (intermediate)
    v2_dir = OUTPUT_ROOT / 'embeddings_v2'; v2_dir.mkdir(parents=True, exist_ok=True)
    np.savez(v2_dir / f'cnn_metseg_embeddings_v2_fold{fold}.npz', **emb_v2)
    d2 = list(emb_v2.values())[0].shape[0]
    print(f'  v2: {len(emb_v2)} × {d2}-dim saved')
    
    # Save v3 (deepest, fixed)
    v3_dir = OUTPUT_ROOT / 'embeddings_v3'; v3_dir.mkdir(parents=True, exist_ok=True)
    np.savez(v3_dir / f'cnn_metseg_embeddings_v3_fold{fold}.npz', **emb_v3)
    d3 = list(emb_v3.values())[0].shape[0]
    print(f'  v3: {len(emb_v3)} × {d3}-dim saved')
    
    return emb_v2, emb_v3

print('Dual-layer extraction ready ✅')""")

code("""print('='*60)
print('  EXTRACTING v2 + v3 EMBEDDINGS (3 FOLDS × 2 LAYERS)')
print('='*60)

results_v2, results_v3 = {}, {}
for fold in [0, 1, 2]:
    print(f'\\n── Fold {fold} ──')
    model = create_segmenter()
    loaded = load_fold_checkpoint(model, fold)
    if not loaded:
        print(f'  Skipping fold {fold}'); continue
    ev2, ev3 = extract_both_layers(model, fold)
    results_v2[fold] = ev2
    results_v3[fold] = ev3
    del model
    torch.cuda.empty_cache()

print(f'\\n✅ Done!')
for label, d in [('v2', 'embeddings_v2'), ('v3', 'embeddings_v3')]:
    edir = OUTPUT_ROOT / d
    for f in sorted(edir.glob('*.npz')):
        print(f'  {f.name} ({f.stat().st_size/1024:.0f} KB)')""")

code("""print('\\n📊 Quality Comparison: v2 (intermediate) vs v3 (deepest, fixed)')
print('='*60)

for fold in sorted(results_v2.keys()):
    for label, embs in [('v2', results_v2[fold]), ('v3', results_v3[fold])]:
        keys = sorted(embs.keys())
        X = np.array([embs[k] for k in keys])
        norms = np.linalg.norm(X, axis=1)
        cos_sims = []
        for i in range(min(20, len(X)-1)):
            cos = np.dot(X[i], X[i+1]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[i+1]) + 1e-8)
            cos_sims.append(cos)
        print(f'  Fold {fold} {label}: {X.shape[1]}-dim | norms [{norms.min():.4f}, {norms.max():.4f}] std={norms.std():.4f} | cos={np.mean(cos_sims):.4f}')
    print()

print('📥 Download BOTH folders:')
print(f'  {OUTPUT_ROOT / "embeddings_v2"}')
print(f'  {OUTPUT_ROOT / "embeddings_v3"}')""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {"accelerator": "gpu", "isGpuEnabled": True}
    },
    "cells": cells
}

out = '/home/moamed/canada_me/explainable_diseas/implementation_cyprus/notebooks/Phase2_EmbeddingFix_ReExtract.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'Notebook: {out}')
print(f'Cells: {len(cells)}')
