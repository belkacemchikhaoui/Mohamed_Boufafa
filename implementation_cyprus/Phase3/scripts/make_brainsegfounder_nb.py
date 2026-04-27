#!/usr/bin/env python3
"""Generate Phase 3 Activity 1B — BrainSegFounder Training notebook (.ipynb)."""
import json, sys

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source.split("\n"), "outputs": [], "execution_count": None})

# ════════════════════════════════════════════════════════════
# Cell 0: Title
# ════════════════════════════════════════════════════════════
md("""# Phase 3 — Experiment B: BrainSegFounder
## Swin UNETR + BrainSegFounder Weights (42k Brain MRI)

**Paper:** BrainSegFounder (Medical Image Analysis, 2024)

**Key Advantages over Experiment A (BraTS-2021 SSL):**
- Pretrained on **42,470 UK Biobank + BraTS brain MRI** (vs 1,251)
- **35× more pretraining data** → richer representations
- Full SwinUNETR fine-tuned on BraTS segmentation → better initialization
- Same architecture (drop-in replacement of weights)

**Architecture:** MONAI `SwinUNETR` — 62M parameters
**Strategy:** ONE fold per Kaggle session → relaunch auto-continues""")

# ════════════════════════════════════════════════════════════
# Cell 1: Config
# ════════════════════════════════════════════════════════════
code("""# ╔════════════════════════════════════════════════════════════╗
# ║  Swin UNETR — FOLD-BY-FOLD (auto-detects completed)      ║
# ╚════════════════════════════════════════════════════════════╝

MODE = 'train_all'   # 'quick_test' | 'train_all'
CV_TYPE = '3fold'

# ╔════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — Swin UNETR                            ║
# ╚════════════════════════════════════════════════════════════╝

CONFIG = {
    # Swin UNETR architecture
    'in_channels': 4,                  # T1, T1c, T2, FLAIR
    'out_channels': 3,                 # WT, TC, ET
    'feature_size': 48,                # Swin UNETR base channel dim (C=48)
    'use_checkpoint': True,            # Gradient checkpointing (saves ~2 GB VRAM)
    'spatial_dims': 3,
    'drop_rate': 0.0,
    'attn_drop_rate': 0.0,
    'dropout_path_rate': 0.0,

    # Training
    'patch_size': [96, 96, 96],        # Full volume crops (matches img_size)
    'num_samples': 2,                  # Crops per volume (reduced for memory)
    'pos_neg_ratio': [2, 1],
    'lr': 1e-4,                        # Lower than Met-Seg (Swin is pretrained)
    'weight_decay': 1e-5,
    'cache_rate': 0.3,                 # Lower cache (Swin uses more memory)
    'batch_size': 1,                   # Swin UNETR is large — batch=1 on T4
    'num_workers': 2,

    # LR schedule: warmup → cosine → fine-tune
    'warmup_epochs': 5,
    'cosine_epochs': 45,               # Cosine decay from lr to 1e-6
    'finetune_epochs': 10,             # Fine-tune at 1e-5
}

if MODE == 'quick_test':
    CONFIG['epochs'] = 10
    CONFIG['patience'] = 10
    CONFIG['val_interval'] = 2
else:
    CONFIG['epochs'] = 60
    CONFIG['patience'] = 20
    CONFIG['val_interval'] = 5

REGION_NAMES = ['WT', 'TC', 'ET']

print(f'Mode: {MODE} | CV: {CV_TYPE}')
print(f'Epochs: {CONFIG["epochs"]} | LR: {CONFIG["lr"]} (AdamW, warmup→cosine→finetune)')
print(f'Batch: {CONFIG["batch_size"]} | Samples/vol: {CONFIG["num_samples"]} | Val every {CONFIG["val_interval"]} ep')
print(f'Swin UNETR: feature_size={CONFIG["feature_size"]} | patch={CONFIG["patch_size"]}')
print(f'Gradient checkpointing: {CONFIG["use_checkpoint"]}')
print(f'Strategy: ONE fold per session → stop → relaunch auto-continues')""")

# ════════════════════════════════════════════════════════════
# Cell 2: Install deps
# ════════════════════════════════════════════════════════════
code("""# Install dependencies
import subprocess, sys
for pkg in ['monai[all]', 'nibabel']:
    try:
        __import__(pkg.split('[')[0])
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
print('Dependencies installed ✅')""")

# ════════════════════════════════════════════════════════════
# Cell 3: Imports + GPU check
# ════════════════════════════════════════════════════════════
code("""import warnings
warnings.filterwarnings('ignore', message='.*Num foregrounds 0.*')
warnings.filterwarnings('ignore', message='.*non-tuple sequence for multidimensional indexing.*')
warnings.filterwarnings('ignore', message='.*axcodes.*length is smaller.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, time, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.data import DataLoader, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import monai.transforms as T
from monai.transforms.compose import MapTransform
from monai.utils import ensure_tuple_rep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM: {vram:.1f} GB')
    if vram < 14:
        print('⚠️ VRAM < 14 GB — reducing num_samples to 1')
        CONFIG['num_samples'] = 1""")

# ════════════════════════════════════════════════════════════
# Cell 4: Load pretrained weights
# ════════════════════════════════════════════════════════════
code("""# ── Find BrainSegFounder pretrained weights ──

OUTPUT_ROOT = Path('/kaggle/working/phase3_brainsegfounder_outputs')
WEIGHTS_DIR = OUTPUT_ROOT / 'pretrained_weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

import glob

def find_bsf_weights():
    \"\"\"Search for BrainSegFounder weights in Kaggle input datasets.\"\"\"
    search_dirs = [
        '/kaggle/input/datasets/boufafamoamed/brainsegfounder-weights',
        '/kaggle/input/brainsegfounder-weights',
        '/kaggle/input/brainsegfounder',
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, '**', '*.pt'), recursive=True):
                if os.path.getsize(f) > 100_000_000:
                    return f
            for f in glob.glob(os.path.join(d, '**', '*.pth'), recursive=True):
                if os.path.getsize(f) > 100_000_000:
                    return f
    # Search ALL input directories
    for f in glob.glob('/kaggle/input/**/*.pt', recursive=True):
        if 'brainseg' in f.lower() and os.path.getsize(f) > 100_000_000:
            return f
    return None

BSF_WEIGHTS = find_bsf_weights()

if BSF_WEIGHTS:
    print(f'✅ BrainSegFounder weights found: {BSF_WEIGHTS}')
    print(f'   Size: {os.path.getsize(BSF_WEIGHTS) / 1024**2:.1f} MB')
else:
    print('⚠️ BrainSegFounder weights NOT found!')
    print('   Downloading from HuggingFace...')
    import urllib.request
    url = 'https://huggingface.co/smilelab/BrainSegFounder/resolve/main/model_weights_BRATS-finetune.pt'
    dst = str(WEIGHTS_DIR / 'model_brainsegfounder_brats_finetune.pt')
    try:
        urllib.request.urlretrieve(url, dst)
        BSF_WEIGHTS = dst
        print(f'✅ Downloaded: {os.path.getsize(dst) / 1024**2:.1f} MB')
    except Exception as e:
        print(f'❌ Download failed: {e}')
        print('   Please add "brainsegfounder-weights" dataset to this notebook')
        BSF_WEIGHTS = None""")

# ════════════════════════════════════════════════════════════
# Cell 5: Data loading (SAME as Phase 2)
# ════════════════════════════════════════════════════════════
code("""# ── Data path resolution (SAME as Phase 2) ──

DATA_ROOT = Path('/kaggle/input/datasets/boufafamoamed/cyprus-proteas-brain-mets')

# Fallback: search all input dirs for data_splits.json
if not DATA_ROOT.exists():
    for candidate in Path('/kaggle/input').iterdir():
        if (candidate / 'data_splits.json').exists():
            DATA_ROOT = candidate
            break
        # Check subdirs
        for sub in candidate.iterdir():
            if sub.is_dir() and (sub / 'data_splits.json').exists():
                DATA_ROOT = sub
                break

SYMLINK_DIR = Path('/kaggle/working/nifti_links')

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

# Load splits (SAME splits as Phase 2!)
splits_file = None
for name in ['data_splits.json', 'cv_splits_3fold.json', 'cv_splits.json']:
    candidate = DATA_ROOT / name
    if candidate.exists():
        splits_file = candidate
        break
if splits_file is None:
    for f in DATA_ROOT.rglob('*splits*.json'):
        splits_file = f; break

with open(splits_file) as f:
    raw_splits = json.load(f)

if '3fold' in raw_splits:
    all_splits = raw_splits['3fold']
elif 'fold_0' in raw_splits:
    all_splits = raw_splits
else:
    raise ValueError(f'Cannot find fold data. Keys: {list(raw_splits.keys())}')

print(f'DATA_ROOT: {DATA_ROOT}')
print(f'Loaded splits: {splits_file.name}')
print(f'Folds: {list(all_splits.keys())}')""")

# ════════════════════════════════════════════════════════════
# Cell 6: Build scan dicts — SEPARATE MODALITY KEYS
# ════════════════════════════════════════════════════════════
code("""# ── Scan dicts: separate keys per modality (t1, t1c, t2, fla) ──
# This avoids MONAI multi-file loading issues in newer versions.
# Each modality is loaded independently with full MetaTensor metadata,
# then merged via ConcatItemsd → 'image' (4 channels).

MOD_KEYS = ['t1', 't1c', 't2', 'fla']

def build_scan_dicts(scans, label='subset'):
    dicts, skips = [], 0
    for scan in scans:
        try:
            dicts.append({
                't1':  resolve_path(DATA_ROOT, scan['t1']),
                't1c': resolve_path(DATA_ROOT, scan['t1c']),
                't2':  resolve_path(DATA_ROOT, scan['t2']),
                'fla': resolve_path(DATA_ROOT, scan['fla']),
                'label': resolve_path(DATA_ROOT, scan['mask']),
                'patient_dir': scan['patient_dir'], 'visit': scan['visit'],
            })
        except FileNotFoundError: skips += 1
    if skips: print(f'  Skipped {skips} in {label}')
    return dicts

def get_fold_dicts(fold):
    fd = all_splits[f'fold_{fold}']
    return build_scan_dicts(fd['train_scans'], f'fold{fold}_train'), build_scan_dicts(fd['test_scans'], f'fold{fold}_val')

def get_all_dicts():
    all_d, seen = [], set()
    for fk in all_splits:
        for scan in all_splits[fk]['train_scans'] + all_splits[fk]['test_scans']:
            key = (scan['patient_dir'], scan['visit'])
            if key not in seen:
                seen.add(key)
                try:
                    all_d.append({
                        't1':  resolve_path(DATA_ROOT, scan['t1']),
                        't1c': resolve_path(DATA_ROOT, scan['t1c']),
                        't2':  resolve_path(DATA_ROOT, scan['t2']),
                        'fla': resolve_path(DATA_ROOT, scan['fla']),
                        'label': resolve_path(DATA_ROOT, scan['mask']),
                        'patient_dir': scan['patient_dir'], 'visit': scan['visit']})
                except FileNotFoundError: pass
    return all_d

train_dicts, val_dicts = get_fold_dicts(0)
print(f'Fold 0: {len(train_dicts)} train | {len(val_dicts)} val')
print(f'  Keys per sample: {list(train_dicts[0].keys())}')""")

# ════════════════════════════════════════════════════════════
# Cell 7: Label conversion
# ════════════════════════════════════════════════════════════
code("""# ── Label conversion: Cyprus {0,1,2,3} → BraTS [WT, TC, ET] ──

class ConvertToMultiChannelBratsMetsd(MapTransform):
    \"\"\"Convert labels to 3-channel [WT, TC, ET].\"\"\"    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            result = [
                (img == 1) | (img == 3) | (img == 2),  # WT
                (img == 1) | (img == 3),                # TC
                img == 3,                                # ET
            ]
            d[key] = torch.stack(result, dim=0).float() if isinstance(img, torch.Tensor) else np.stack(result, axis=0).astype(np.float32)
        return d

print('Label mapping (Cyprus → BraTS):')
print('  Channel 0 (WT) = labels 1+2+3 (all tumor)')
print('  Channel 1 (TC) = labels 1+3 (tumor core)')
print('  Channel 2 (ET) = label 3 (enhancing only)')""")

# ════════════════════════════════════════════════════════════
# Cell 8: Transforms — separate load + ConcatItemsd
# ════════════════════════════════════════════════════════════
code("""# ── Transforms: Load each modality separately, then concat ──
# This is the ROBUST approach for newer MONAI (1.5+) versions.
# Each modality is loaded as its own MetaTensor with correct affine,
# then ConcatItemsd merges them into a single 4-channel 'image'.

patch = CONFIG['patch_size']
ALL_KEYS = MOD_KEYS + ['label']  # ['t1','t1c','t2','fla','label']

train_transforms = T.Compose([
    # Step 1: Load each modality + label separately
    T.LoadImaged(keys=ALL_KEYS),
    T.EnsureChannelFirstd(keys=ALL_KEYS),
    T.EnsureTyped(keys=ALL_KEYS),
    # Step 2: Merge 4 modalities → 'image' (4, H, W, D)
    T.ConcatItemsd(keys=MOD_KEYS, name='image', dim=0),
    T.DeleteItemsd(keys=MOD_KEYS),  # Clean up individual modality keys
    # Step 3: Standard preprocessing
    T.Orientationd(keys=['image', 'label'], axcodes='RAS'),
    T.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=True),
    T.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ConvertToMultiChannelBratsMetsd(keys=['label']),
    # Step 4: Light augmentation
    T.RandFlipd(keys=['image', 'label'], spatial_axis=[0], prob=0.5),
    T.RandFlipd(keys=['image', 'label'], spatial_axis=[1], prob=0.5),
    T.RandFlipd(keys=['image', 'label'], spatial_axis=[2], prob=0.5),
    T.RandScaleIntensityd(keys='image', factors=0.1, prob=0.3),
    T.RandShiftIntensityd(keys='image', offsets=0.1, prob=0.3),
    T.SpatialPadd(keys=['image', 'label'], spatial_size=ensure_tuple_rep(patch, 3)),
    T.RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label',
        spatial_size=ensure_tuple_rep(patch, 3),
        pos=CONFIG['pos_neg_ratio'][0], neg=CONFIG['pos_neg_ratio'][1],
        num_samples=CONFIG['num_samples'], image_key='image', image_threshold=0),
    T.EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
])

val_transforms = T.Compose([
    T.LoadImaged(keys=ALL_KEYS),
    T.EnsureChannelFirstd(keys=ALL_KEYS),
    T.EnsureTyped(keys=ALL_KEYS),
    T.ConcatItemsd(keys=MOD_KEYS, name='image', dim=0),
    T.DeleteItemsd(keys=MOD_KEYS),
    T.Orientationd(keys=['image', 'label'], axcodes='RAS'),
    T.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=True),
    T.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ConvertToMultiChannelBratsMetsd(keys=['label']),
    T.EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
])

# Quick test: verify the transform pipeline produces correct shapes
test_data = train_transforms(train_dicts[0])
if isinstance(test_data, list):
    # num_samples > 1 returns a list
    print(f'  Train output: list of {len(test_data)} crops')
    print(f'  Image shape: {test_data[0]["image"].shape}')  # Should be (4, 96, 96, 96)
    print(f'  Label shape: {test_data[0]["label"].shape}')  # Should be (3, 96, 96, 96)
else:
    print(f'  Image shape: {test_data["image"].shape}')
    print(f'  Label shape: {test_data["label"].shape}')

assert test_data[0]['image'].shape[0] == 4, f'Expected 4 channels, got {test_data[0]["image"].shape[0]}'
print(f'  ✅ 4-channel verification passed!')
print(f'  Light augmentation (flips + mild intensity + shift)')
print(f'  {CONFIG["num_samples"]} random crops per volume, patch {patch}')""")


# ════════════════════════════════════════════════════════════
# Cell 9: Model creation + weight loading
# ════════════════════════════════════════════════════════════
code("""# ── Swin UNETR + BrainSegFounder weight loading ──
# BrainSegFounder BRATS-finetune.pt = FULL SwinUNETR state_dict
# (in_channels=4, out_channels=3, feature_size=48) — drop-in!

import inspect
_swin_sig = inspect.signature(SwinUNETR.__init__)
_has_img_size = 'img_size' in _swin_sig.parameters

def create_model():
    \"\"\"Create Swin UNETR model.\"\"\"
    kwargs = dict(
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['out_channels'],
        feature_size=CONFIG['feature_size'],
        use_checkpoint=CONFIG['use_checkpoint'],
        spatial_dims=CONFIG['spatial_dims'],
        drop_rate=CONFIG['drop_rate'],
        attn_drop_rate=CONFIG['attn_drop_rate'],
        dropout_path_rate=CONFIG['dropout_path_rate'],
    )
    if _has_img_size:
        kwargs['img_size'] = tuple(CONFIG['patch_size'])
    return SwinUNETR(**kwargs)

def load_bsf_weights(model):
    \"\"\"Load BrainSegFounder weights (full state_dict).\"\"\"
    if not BSF_WEIGHTS:
        print('  ⚠️ No BrainSegFounder weights')
        return model
    sd = torch.load(BSF_WEIGHTS, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    if isinstance(sd, dict) and 'model_state_dict' in sd: sd = sd['model_state_dict']
    try:
        model.load_state_dict(sd, strict=True)
        print('  ✅ BrainSegFounder FULL model loaded (encoder+decoder)')
    except RuntimeError:
        mdict = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in mdict and v.shape == mdict[k].shape}
        mdict.update(matched)
        model.load_state_dict(mdict)
        print(f'  ✅ BrainSegFounder partial: {len(matched)}/{len(mdict)} layers')
    return model

test_model = create_model()
n_params = sum(p.numel() for p in test_model.parameters())
print(f'SwinUNETR (BrainSegFounder): {n_params:,} params ({n_params/1e6:.1f}M)')
load_bsf_weights(test_model)
del test_model; torch.cuda.empty_cache()""")

# ════════════════════════════════════════════════════════════
# Cell 10: Embedding extraction hook
# ════════════════════════════════════════════════════════════
code("""# ── Embedding hook for Swin UNETR bottleneck ──
# Swin UNETR encoder produces features at 5 stages.
# We hook the LAST encoder stage (768-dim) for embeddings.

_embedding_storage = {}

def _hook_fn(module, input, output):
    \"\"\"Capture output of the bottleneck layer.\"\"\"
    if isinstance(output, (list, tuple)):
        feat = output[-1]
    else:
        feat = output
    _embedding_storage['feat'] = feat.detach()

def register_embedding_hook(model):
    \"\"\"Register hook on Swin UNETR encoder bottleneck.\"\"\"
    target = None
    if hasattr(model, 'swinViT'):
        if hasattr(model.swinViT, 'norm'):
            target = model.swinViT.norm
        elif hasattr(model.swinViT, 'layers4'):
            target = model.swinViT.layers4
    
    if target is not None:
        hook = target.register_forward_hook(_hook_fn)
        print(f'  Hook registered on {target.__class__.__name__} ✅')
        return hook
    else:
        print('  Using forward-pass extraction (no hook needed)')
        return None

print('Embedding extraction ready')
print('  Method: Swin UNETR encoder bottleneck → GAP → 768-dim')""")

# ════════════════════════════════════════════════════════════
# Cell 11: LR schedule + Training function
# ════════════════════════════════════════════════════════════
code("""def get_lr_for_epoch(epoch, config):
    \"\"\"Warmup → Cosine decay → Fine-tune schedule.\"\"\"
    warmup = config['warmup_epochs']
    cosine_end = warmup + config['cosine_epochs']
    base_lr = config['lr']
    
    if epoch < warmup:
        return 1e-6 + (base_lr - 1e-6) * (epoch / warmup)
    elif epoch < cosine_end:
        progress = (epoch - warmup) / config['cosine_epochs']
        return 1e-6 + (base_lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        return 1e-5

def train_fold(fold):
    \"\"\"Train one fold of Swin UNETR.\"\"\"
    
    ckpt_dir = OUTPUT_ROOT / 'checkpoints'; ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f'bsf_fold{fold}_best.pth'
    latest_path = ckpt_dir / f'bsf_fold{fold}_latest.pth'
    
    # ── Step 1: Check if fold is COMPLETE ──
    if best_path.exists() and latest_path.exists():
        latest_ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
        if latest_ckpt.get('epoch', -1) >= CONFIG['epochs'] - 1:
            bd = latest_ckpt.get('best_dice', 0)
            print(f'  ✅ Fold {fold} already complete (Dice={bd:.4f}), skipping')
            model = create_model().to(device)
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt['model_state_dict'])
            return model, bd, best_ckpt.get('metrics_history', {}), True
    
    # ── Step 2: Build data loaders ──
    train_dicts, val_dicts = get_fold_dicts(fold)
    print(f'  Train: {len(train_dicts)} | Val: {len(val_dicts)}')
    
    train_ds = CacheDataset(train_dicts, train_transforms, cache_rate=CONFIG['cache_rate'], num_workers=CONFIG['num_workers'])
    val_ds = CacheDataset(val_dicts, val_transforms, cache_rate=1.0, num_workers=CONFIG['num_workers'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=CONFIG['num_workers'])
    
    # ── Step 3: Create model and load pretrained weights ──
    model = create_model().to(device)
    
    load_bsf_weights(model)
    print('  Loaded BrainSegFounder weights ✅')
    
    # ── Step 4: Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'], betas=(0.9, 0.999))
    
    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True,
                         squared_pred=True, smooth_nr=0, smooth_dr=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    dice_metric = DiceMetric(include_background=True, reduction='mean_batch')
    
    best_dice, patience_ctr = 0.0, 0
    start_epoch = 0
    metrics_log = {'train_loss': [], 'val_dice': [], 'val_per_region': [], 'lr': []}
    
    # ── Step 5: Resume from checkpoint ──
    if latest_path.exists():
        print(f'  🔄 Found latest checkpoint, attempting resume...')
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        metrics_log = ckpt.get('metrics_history', metrics_log)
        patience_ctr = ckpt.get('patience_ctr', 0)
        print(f'  🔄 Resuming fold {fold} from epoch {start_epoch} (best: {best_dice:.4f})')
    
    # ── Step 6: GPU info ──
    print(f'  Using {torch.cuda.get_device_name(0)}')
    
    t0 = time.time()
    
    # ── Step 7: Training loop ──
    for epoch in range(start_epoch, CONFIG['epochs']):
        # LR schedule
        current_lr = get_lr_for_epoch(epoch, CONFIG)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
        
        model.train()
        epoch_loss, n_steps = 0.0, 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            n_steps += 1
        
        avg_loss = epoch_loss / max(n_steps, 1)
        metrics_log['train_loss'].append(avg_loss)
        metrics_log['lr'].append(current_lr)
        
        # ── Validation ──
        if (epoch + 1) % CONFIG['val_interval'] == 0 or epoch == CONFIG['epochs'] - 1:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for vb in val_loader:
                    vi = vb['image'].to(device)
                    vl = vb['label'].to(device)
                    with torch.amp.autocast('cuda'):
                        vo = sliding_window_inference(vi, CONFIG['patch_size'], 4, model,
                                                      overlap=0.5, mode='gaussian')
                    vp = (torch.sigmoid(vo) > 0.5).float()
                    dice_metric(vp, vl)
            
            dv = dice_metric.aggregate()
            md = dv.mean().item()
            pr = [dv[i].item() for i in range(len(REGION_NAMES))]
            metrics_log['val_dice'].append(md)
            metrics_log['val_per_region'].append(pr)
            
            elapsed = (time.time() - t0) / 60
            rs = ' '.join([f'{n}={v:.3f}' for n, v in zip(REGION_NAMES, pr)])
            print(f'Epoch {epoch:3d}/{CONFIG["epochs"]-1} | Loss: {avg_loss:.4f} | Dice: {md:.4f} ({rs}) | LR: {current_lr:.1e} | {elapsed:.1f} min')
            
            if md > best_dice:
                best_dice = md; patience_ctr = 0
                torch.save({
                    'epoch': epoch, 'best_dice': best_dice,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics_history': metrics_log,
                    'config': CONFIG,
                }, best_path)
                print(f'  ✅ New best! Saved.')
            else:
                patience_ctr += 1
                if patience_ctr >= CONFIG['patience']:
                    print(f'  Early stopping at epoch {epoch}.'); break
        else:
            elapsed = (time.time() - t0) / 60
            print(f'Epoch {epoch:3d}/{CONFIG["epochs"]-1} | Loss: {avg_loss:.4f} | LR: {current_lr:.1e} | {elapsed:.1f} min')
        
        # Save LATEST checkpoint every epoch
        torch.save({
            'epoch': epoch, 'best_dice': best_dice,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics_history': metrics_log,
            'patience_ctr': patience_ctr,
            'config': CONFIG,
        }, latest_path)
    
    elapsed = (time.time() - t0) / 60
    print(f'\\n  Fold {fold} complete: Best Dice = {best_dice:.4f} | Time = {elapsed:.1f} min')
    
    # ── Save figures ──
    fig_dir = OUTPUT_ROOT / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(metrics_log['train_loss']); axes[0].set_title(f'Loss (Fold {fold})'); axes[0].set_xlabel('Epoch')
    if metrics_log['val_dice']:
        axes[1].plot(metrics_log['val_dice'], label='Mean Dice', marker='o'); axes[1].legend()
    axes[1].set_title(f'Val Dice (Fold {fold})'); axes[1].set_xlabel('Validation Step')
    if metrics_log['lr']:
        axes[2].plot(metrics_log['lr']); axes[2].set_title(f'LR Schedule (warmup→cosine→finetune)')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR')
    plt.tight_layout(); plt.savefig(fig_dir / f'bsf_fold{fold}_training_curves.png', dpi=150); plt.close()
    
    met_dir = OUTPUT_ROOT / 'metrics'; met_dir.mkdir(parents=True, exist_ok=True)
    with open(met_dir / f'bsf_fold{fold}_metrics.json', 'w') as f_out:
        json.dump(metrics_log, f_out, indent=2)
    
    return model, best_dice, metrics_log, False""")

# ════════════════════════════════════════════════════════════
# Cell 12: Embedding extraction
# ════════════════════════════════════════════════════════════
code("""def extract_embeddings(model, fold):
    \"\"\"Extract 768-dim embeddings from Swin UNETR encoder for all scans.
    
    Key: Resize full volumes to 96³ before encoder pass to avoid OOM.
    Full volumes (~240×240×155) are too large for T4 VRAM.
    \"\"\"
    # Flush GPU memory first
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    model.eval()
    model.to(device)
    
    all_dicts = get_all_dicts()
    print(f'  Extracting from {len(all_dicts)} scans...')
    
    # Use val_transforms (no augmentation) but we'll resize before encoder pass
    emb_ds = CacheDataset(all_dicts, val_transforms, cache_rate=CONFIG['cache_rate'], num_workers=CONFIG['num_workers'])
    emb_loader = DataLoader(emb_ds, batch_size=1, shuffle=False, num_workers=CONFIG['num_workers'])
    
    target_size = ensure_tuple_rep(CONFIG['patch_size'], 3)  # (96, 96, 96)
    embeddings = {}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(emb_loader, desc='Extracting embeddings')):
            images = batch['image']  # (1, 4, H, W, D) — full volume
            patient = batch['patient_dir'][0]
            visit = batch['visit'][0]
            key = f'{patient}__{visit}'
            
            # Resize to 96³ to fit in VRAM (same as training patch size)
            images = F.interpolate(images.float(), size=target_size, mode='trilinear', align_corners=False)
            images = images.to(device)
            
            with torch.amp.autocast('cuda'):
                encoder_outputs = model.swinViT(images, model.normalize)
                bottleneck = encoder_outputs[-1]  # deepest features
                emb = F.adaptive_avg_pool3d(bottleneck, 1).flatten()
            
            embeddings[key] = emb.cpu().numpy()
            
            # Free memory periodically
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
    
    if len(embeddings) == 0:
        print('  ⚠️ No embeddings extracted!')
        return
    
    emb_dir = OUTPUT_ROOT / 'embeddings'; emb_dir.mkdir(parents=True, exist_ok=True)
    emb_dim = list(embeddings.values())[0].shape[0]
    
    np.savez(emb_dir / f'bsf_embeddings_fold{fold}.npz', **embeddings)
    
    meta = {k: {'patient_dir': k.split('__')[0], 'visit': k.split('__')[1]} for k in embeddings}
    with open(emb_dir / f'bsf_embeddings_fold{fold}_meta.json', 'w') as f_out:
        json.dump(meta, f_out, indent=2)
    
    norms = [np.linalg.norm(v) for v in embeddings.values()]
    print(f'  ✅ {len(embeddings)} embeddings × {emb_dim}-dim saved')
    print(f'     Norm range: [{min(norms):.4f}, {max(norms):.4f}] std={np.std(norms):.4f}')
    
    n_bad = sum(1 for v in embeddings.values() if np.any(np.isnan(v)) or np.any(np.isinf(v)))
    if n_bad > 0:
        print(f'  ⚠️ {n_bad} embeddings have NaN/Inf!')
    else:
        print(f'     No NaN/Inf ✅')""")

# ════════════════════════════════════════════════════════════
# Cell 13: MAIN — fold-by-fold execution
# ════════════════════════════════════════════════════════════
code("""# ╔════════════════════════════════════════════════════════════╗
# ║  MAIN — FOLD-BY-FOLD (one fold per session)              ║
# ║  Relaunch auto-detects completed folds and continues     ║
# ╚════════════════════════════════════════════════════════════╝

import torch, shutil
n_gpus = torch.cuda.device_count()
print(f'GPUs available: {n_gpus}')
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)')

# ── Recover checkpoints/embeddings from uploaded datasets ──
# On Kaggle, each version starts fresh. Previous fold outputs
# must be uploaded as datasets. This block searches /kaggle/input/
# for checkpoint .pth files and embedding .npz files, and copies
# them to the working directory so fold detection works.

ckpt_dir = OUTPUT_ROOT / 'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)
emb_dir_recovery = OUTPUT_ROOT / 'embeddings'
emb_dir_recovery.mkdir(parents=True, exist_ok=True)
fig_dir_recovery = OUTPUT_ROOT / 'figures'
fig_dir_recovery.mkdir(parents=True, exist_ok=True)
met_dir_recovery = OUTPUT_ROOT / 'metrics'
met_dir_recovery.mkdir(parents=True, exist_ok=True)

recovered = 0
for input_dir in Path('/kaggle/input').iterdir():
    if not input_dir.is_dir():
        continue
    # Search for checkpoint files
    for pth_file in input_dir.rglob('bsf_fold*_best.pth'):
        dest = ckpt_dir / pth_file.name
        if not dest.exists():
            shutil.copy2(str(pth_file), str(dest))
            print(f'  📦 Recovered: {pth_file.name}')
            recovered += 1
    for pth_file in input_dir.rglob('bsf_fold*_latest.pth'):
        dest = ckpt_dir / pth_file.name
        if not dest.exists():
            shutil.copy2(str(pth_file), str(dest))
            print(f'  📦 Recovered: {pth_file.name}')
            recovered += 1
    # Search for embedding files
    for npz_file in input_dir.rglob('bsf_embeddings_fold*.npz'):
        dest = emb_dir_recovery / npz_file.name
        if not dest.exists():
            shutil.copy2(str(npz_file), str(dest))
            print(f'  📦 Recovered: {npz_file.name}')
            recovered += 1
    for json_file in input_dir.rglob('bsf_embeddings_fold*_meta.json'):
        dest = emb_dir_recovery / json_file.name
        if not dest.exists():
            shutil.copy2(str(json_file), str(dest))
            recovered += 1
    # Search for figures and metrics
    for fig_file in input_dir.rglob('bsf_fold*_training_curves.png'):
        dest = fig_dir_recovery / fig_file.name
        if not dest.exists():
            shutil.copy2(str(fig_file), str(dest))
            recovered += 1
    for met_file in input_dir.rglob('bsf_fold*_metrics.json'):
        dest = met_dir_recovery / met_file.name
        if not dest.exists():
            shutil.copy2(str(met_file), str(dest))
            recovered += 1

if recovered > 0:
    print(f'  📦 Total recovered from previous versions: {recovered} files')
else:
    print(f'  No previous fold outputs found in /kaggle/input/')

# ── Find the NEXT fold to train ──

completed_folds = {}
target_fold = None
for f in [0, 1, 2]:
    best_p = ckpt_dir / f'bsf_fold{f}_best.pth'
    latest_p = ckpt_dir / f'bsf_fold{f}_latest.pth'
    if best_p.exists() and latest_p.exists():
        ckpt = torch.load(latest_p, map_location='cpu', weights_only=False)
        if ckpt.get('epoch', -1) >= CONFIG['epochs'] - 1:
            completed_folds[f] = ckpt.get('best_dice', 0)
            print(f'  ✅ Fold {f}: COMPLETE (Dice={completed_folds[f]:.4f})')
            continue
        else:
            print(f'  🔄 Fold {f}: PARTIAL (epoch {ckpt.get("epoch", 0)}/{CONFIG["epochs"]-1})')
    else:
        print(f'  🆕 Fold {f}: NOT STARTED')
    if target_fold is None:
        target_fold = f

if target_fold is None:
    print('\\n' + '=' * 60)
    print('  🎉 ALL 3 FOLDS COMPLETE!')
    print('=' * 60)
    for f, d in completed_folds.items():
        print(f'  Fold {f}: Dice={d:.4f}')
    mean_dice = sum(completed_folds.values()) / len(completed_folds)
    std_dice = np.std(list(completed_folds.values()))
    print(f'  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}')
    
    # Extract embeddings for all folds
    for f in [0, 1, 2]:
        emb_path = OUTPUT_ROOT / f'embeddings/bsf_embeddings_fold{f}.npz'
        if not emb_path.exists():
            print(f'\\n  Extracting embeddings for fold {f}...')
            model = create_model().to(device)
            best_ckpt = torch.load(ckpt_dir / f'bsf_fold{f}_best.pth', map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt['model_state_dict'])
            extract_embeddings(model, f)
            del model; torch.cuda.empty_cache()
else:
    # -- Extract embeddings for any recovered folds missing embeddings --
    for cf in list(completed_folds.keys()):
        emb_chk = OUTPUT_ROOT / f'embeddings/bsf_embeddings_fold{cf}.npz'
        if not emb_chk.exists():
            print(f'  Fold {cf} recovered but missing embeddings — extracting now...')
            emb_model = create_model().to(device)
            bc = torch.load(ckpt_dir / f'bsf_fold{cf}_best.pth', map_location=device, weights_only=False)
            emb_model.load_state_dict(bc['model_state_dict'])
            del bc; torch.cuda.empty_cache()
            extract_embeddings(emb_model, cf)
            del emb_model
            import gc; gc.collect(); torch.cuda.empty_cache()
    
    print(f'\\n  ▶▶ Training FOLD {target_fold} this session')
    print('=' * 60)
    print(f'  Training Fold {target_fold} | Swin UNETR | {CONFIG["epochs"]} epochs')
    print('=' * 60)
    
    model, best_dice, metrics, was_skipped = train_fold(target_fold)
    
    # ── Free training memory before embedding extraction ──
    # Training objects hold >5GB of VRAM — must release
    import gc
    try: del optimizer
    except: pass
    try: del scaler
    except: pass
    try: del loss_fn
    except: pass
    try: del train_ds, val_ds, train_loader, val_loader
    except: pass
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  GPU memory cleared for embedding extraction')
    
    # Reload best model for extraction
    best_path = ckpt_dir / f'bsf_fold{target_fold}_best.pth'
    if best_path.exists():
        # Create fresh model to avoid any residual training state
        del model; gc.collect(); torch.cuda.empty_cache()
        model = create_model().to(device)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        del ckpt; gc.collect(); torch.cuda.empty_cache()
    print(f'\\n  Extracting embeddings for fold {target_fold}...')
    extract_embeddings(model, target_fold)
    
    completed_folds[target_fold] = best_dice
    remaining = [f for f in [0,1,2] if f not in completed_folds]
    
    print('\\n' + '=' * 60)
    print(f'  ✅ Fold {target_fold} DONE — Dice={best_dice:.4f}')
    print(f'  Completed: {list(completed_folds.keys())} | Remaining: {remaining}')
    if remaining:
        print(f'  → Relaunch notebook to train fold {remaining[0]}')
    else:
        print(f'  🎉 ALL FOLDS COMPLETE!')
        mean_dice = sum(completed_folds.values()) / len(completed_folds)
        print(f'  Mean Dice: {mean_dice:.4f}')
    print('=' * 60)

print(f'\\n  Files saved to {OUTPUT_ROOT}:')
for p in sorted(OUTPUT_ROOT.rglob('*')):
    if p.is_file():
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f'    {p.relative_to(OUTPUT_ROOT)} ({size_mb:.1f} MB)')
total_mb = sum(p.stat().st_size for p in OUTPUT_ROOT.rglob('*') if p.is_file()) / (1024*1024)
print(f'\\n  Total output size: {total_mb:.1f} MB / 19,000 MB limit')""")

# ════════════════════════════════════════════════════════════
# Cell 14: 3D Inference Visualization
# ════════════════════════════════════════════════════════════
code("""# ╔════════════════════════════════════════════════════════════╗
# ║  3D INFERENCE VISUALIZATION — Ground Truth vs Prediction  ║
# ╚════════════════════════════════════════════════════════════╝

from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_3d_tumor(ax, mask_3d, color, alpha=0.5, title=''):
    \"\"\"Render a binary 3D mask as a surface mesh.\"\"\"
    if mask_3d.sum() < 10:
        ax.set_title(f'{title}\\n(empty)')
        return
    try:
        verts, faces, _, _ = marching_cubes(mask_3d, level=0.5, step_size=2)
        mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0.1)
        mesh.set_facecolor(color)
        mesh.set_edgecolor([c * 0.7 for c in color[:3]] + [0.3])
        ax.add_collection3d(mesh)
        ax.set_xlim(0, mask_3d.shape[0])
        ax.set_ylim(0, mask_3d.shape[1])
        ax.set_zlim(0, mask_3d.shape[2])
    except Exception as e:
        ax.set_title(f'{title}\\n(render failed: {e})')

def visualize_3d_comparison(model, val_loader, fold, n_samples=3):
    \"\"\"Create 3D rendering comparison: Ground Truth vs Swin UNETR Prediction.\"\"\"
    model.eval()
    fig_dir = OUTPUT_ROOT / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {
        'WT': (0.2, 0.7, 0.3, 0.3),
        'TC': (0.9, 0.8, 0.2, 0.5),
        'ET': (0.9, 0.2, 0.2, 0.7),
    }
    
    samples_done = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if samples_done >= n_samples:
                break
            
            image = batch['image'].to(device)
            label = batch['label']
            
            with torch.amp.autocast('cuda'):
                pred = sliding_window_inference(image, CONFIG['patch_size'], 4, model,
                                                 overlap=0.5, mode='gaussian')
            pred_bin = (torch.sigmoid(pred) > 0.5).float().cpu()
            
            gt = label[0].numpy()
            pr = pred_bin[0].numpy()
            
            if gt.sum() < 50:
                continue
            
            sample_dice = []
            for c in range(3):
                intersection = (gt[c] * pr[c]).sum()
                union = gt[c].sum() + pr[c].sum()
                d = (2 * intersection / (union + 1e-8))
                sample_dice.append(d)
            mean_d = sum(sample_dice) / 3
            
            fig = plt.figure(figsize=(16, 8))
            fig.suptitle(f'Sample {idx} | Dice: WT={sample_dice[0]:.3f} TC={sample_dice[1]:.3f} ET={sample_dice[2]:.3f} | Mean={mean_d:.3f}', fontsize=14, fontweight='bold')
            
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Ground Truth', fontsize=13, fontweight='bold')
            for c, (name, color) in enumerate(colors.items()):
                render_3d_tumor(ax1, gt[c], color, alpha=color[3])
            ax1.view_init(elev=25, azim=135)
            ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
            ax1.grid(False)
            ax1.xaxis.pane.fill = False; ax1.yaxis.pane.fill = False; ax1.zaxis.pane.fill = False
            
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title('BrainSegFounder Prediction', fontsize=13, fontweight='bold')
            for c, (name, color) in enumerate(colors.items()):
                render_3d_tumor(ax2, pr[c], color, alpha=color[3])
            ax2.view_init(elev=25, azim=135)
            ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
            ax2.grid(False)
            ax2.xaxis.pane.fill = False; ax2.yaxis.pane.fill = False; ax2.zaxis.pane.fill = False
            
            plt.tight_layout()
            save_path = fig_dir / f'bsf_fold{fold}_3d_sample{idx}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.show()
            print(f'  Saved: {save_path.name}')
            
            samples_done += 1
    
    print(f'\\n  ✅ {samples_done} 3D visualizations saved to {fig_dir}')

# ── Run inference on the completed fold ──
print('\\n' + '=' * 60)
print('  3D INFERENCE VISUALIZATION')
print('=' * 60)

ckpt_dir = OUTPUT_ROOT / 'checkpoints'
vis_fold = None
for f in [0, 1, 2]:
    best_p = ckpt_dir / f'bsf_fold{f}_best.pth'
    latest_p = ckpt_dir / f'bsf_fold{f}_latest.pth'
    if best_p.exists() and latest_p.exists():
        ckpt = torch.load(latest_p, map_location='cpu', weights_only=False)
        if ckpt.get('epoch', -1) >= CONFIG['epochs'] - 1:
            vis_fold = f

if vis_fold is not None:
    best_path = ckpt_dir / f'bsf_fold{vis_fold}_best.pth'
    if best_path.exists():
        print(f'  Loading best model from fold {vis_fold}...')
        vis_model = create_model().to(device)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        vis_model.load_state_dict(ckpt['model_state_dict'])
        print(f'  Best Dice: {ckpt.get("best_dice", "?"):.4f}')
        
        _, vis_val_dicts = get_fold_dicts(vis_fold)
        vis_ds = CacheDataset(vis_val_dicts, val_transforms, cache_rate=1.0, num_workers=0)
        vis_loader = DataLoader(vis_ds, batch_size=1, shuffle=False, num_workers=0)
        
        visualize_3d_comparison(vis_model, vis_loader, vis_fold, n_samples=3)
        
        del vis_model
        torch.cuda.empty_cache()
    else:
        print(f'  ⚠️ No best checkpoint found for fold {vis_fold}')
else:
    print('  ⚠️ No completed folds found for visualization')""")

# ════════════════════════════════════════════════════════════
# Assemble notebook
# ════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

from pathlib import Path as _Path
out_path = _Path(__file__).parent.parent / "notebooks" / "Phase3_A1B_BrainSegFounder_Training.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"✅ Generated: {out_path}")
print(f"   Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'code')} code, {sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")
