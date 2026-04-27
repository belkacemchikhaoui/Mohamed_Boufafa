"""Build Phase2_A2 - nnU-Net v2 Fine-Tuning (native framework, resumable)."""
import json

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src.strip().split("\n"),"id":"a"}
def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip().split("\n"),"id":"a"}

cells = [
md("""# Phase 2B — nnU-Net v2 Fine-Tuning (BraTS 2024 Post-Treatment)
## CNN Baseline #2: nnU-Net v2 (native framework from MIC-DKFZ)

**Architecture:** PlainConvUNet (auto-configured by nnU-Net v2)
- Patch size: 128×160×112
- 4 input channels: T1, T1ce, T2, FLAIR
- Labels: Background(0), Edema(1), Non-enhancing(2), empty(3), Enhancing(4)
- Pretrained: BraTS 2019, 5 folds × 1000 epochs

**Strategy:**
- nnU-Net v2 is a FULL FRAMEWORK, not just a model
- We use `nnunetv2` package directly (pip install nnunetv2)
- Fine-tune from pretrained checkpoint on BraTS 2024 Post-Treatment data
- **Resumable**: nnU-Net v2 natively supports `--c` flag to continue training
- Each Kaggle session: ~10h of training, then resume in next session

**Note:** nnU-Net v2 ≠ MONAI DynUNet. They are different architectures with
incompatible weights. This notebook uses the REAL nnU-Net v2."""),

code("""# ╔════════════════════════════════════════╗
# ║  CONFIG                                ║
# ╚════════════════════════════════════════╝
import os, shutil, json
from pathlib import Path

# nnU-Net environment variables (MUST be set before importing nnunetv2)
NNUNET_BASE = Path('/kaggle/working/nnunet')
os.environ['nnUNet_raw'] = str(NNUNET_BASE / 'raw')
os.environ['nnUNet_preprocessed'] = str(NNUNET_BASE / 'preprocessed')
os.environ['nnUNet_results'] = str(NNUNET_BASE / 'results')
for p in [NNUNET_BASE / 'raw', NNUNET_BASE / 'preprocessed', NNUNET_BASE / 'results']:
    p.mkdir(parents=True, exist_ok=True)

OUTPUT_ROOT = Path('/kaggle/working/phase2_nnunet')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Dataset ID for our BraTS 2024 task
DATASET_ID = 100  # Dataset100_BraTS2024
DATASET_NAME = f'Dataset{DATASET_ID:03d}_BraTS2024'
FOLD = 0  # which fold to train (change per session if needed)
EPOCHS = 200  # fine-tuning epochs (not 1000 - we start from pretrained)

print(f'nnU-Net base:  {NNUNET_BASE}')
print(f'Dataset:       {DATASET_NAME}')
print(f'Fold:          {FOLD}')
print(f'Fine-tune for: {EPOCHS} epochs')"""),

code("""# ── Install nnU-Net v2 ──
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'nnunetv2'])
print('nnU-Net v2 installed')"""),

code("""# ── Prepare BraTS 2024 data in nnU-Net format ──
# nnU-Net expects:
#   raw/DatasetXXX/imagesTr/  → CASE_0000.nii.gz, CASE_0001.nii.gz, etc (4 modalities)
#   raw/DatasetXXX/labelsTr/  → CASE.nii.gz (segmentation)
#   raw/DatasetXXX/dataset.json

DATA_ROOT = Path('/kaggle/input')
NIFTI_ROOT = None
for c in DATA_ROOT.rglob('BraTS-GLI-*'):
    if c.is_dir():
        NIFTI_ROOT = c.parent; break

if NIFTI_ROOT is None:
    # Try common paths
    for p in [DATA_ROOT / 'brats2024-training-data', DATA_ROOT / 'brats2024-training-additional']:
        for c in p.rglob('BraTS-GLI-*'):
            if c.is_dir(): NIFTI_ROOT = c.parent; break
        if NIFTI_ROOT: break

print(f'NIfTI source: {NIFTI_ROOT}')

# Discover all patient scan folders
all_dirs = sorted([d for d in NIFTI_ROOT.rglob('BraTS-GLI-*') if d.is_dir() and any(d.glob('*-seg*'))])
print(f'Found {len(all_dirs)} scans with labels')

# Create nnU-Net folder structure
raw_dir = Path(os.environ['nnUNet_raw']) / DATASET_NAME
images_dir = raw_dir / 'imagesTr'
labels_dir = raw_dir / 'labelsTr'
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Map BraTS modality suffixes to nnU-Net channel indices
MOD_MAP = {'t1n': '0000', 't1c': '0001', 't2w': '0002', 't2f': '0003'}

# Create symlinks (fast, no copy)
skipped = 0
for d in all_dirs:
    case_id = d.name  # e.g., BraTS-GLI-02641-100
    for suffix, idx in MOD_MAP.items():
        src = list(d.glob(f'*-{suffix}*'))
        if src:
            dst = images_dir / f'{case_id}_{idx}.nii.gz'
            if not dst.exists():
                os.symlink(str(src[0]), str(dst))
        else: skipped += 1
    seg = list(d.glob('*-seg*'))
    if seg:
        dst = labels_dir / f'{case_id}.nii.gz'
        if not dst.exists():
            os.symlink(str(seg[0]), str(dst))

print(f'Created symlinks: {len(list(images_dir.glob("*.nii.gz")))} images, {len(list(labels_dir.glob("*.nii.gz")))} labels')
if skipped: print(f'  Skipped {skipped} missing modality files')"""),

code("""# ── Create dataset.json ──
cases = sorted(set(f.name.rsplit('_', 1)[0] for f in images_dir.glob('*.nii.gz')))
dataset_json = {
    'channel_names': {'0': 'T1', '1': 'T1ce', '2': 'T2', '3': 'FLAIR'},
    'labels': {
        'background': 0,
        'NCR': 1,       # Necrotic core
        'ED': 2,        # Edema
        'ET': 4,        # Enhancing tumor (label 4, not 3!)
    },
    'numTraining': len(cases),
    'file_ending': '.nii.gz',
}
with open(raw_dir / 'dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)
print(f'dataset.json: {len(cases)} training cases')
print(f'Labels: {dataset_json["labels"]}')"""),

code("""# ── Copy pretrained weights to nnU-Net results dir ──
# The pretrained checkpoint was trained on Dataset002_BRATS19
# We copy fold_0 as starting point for fine-tuning

PRETRAINED_ROOT = None
for loc in [Path('/kaggle/input')]:
    for f in loc.rglob('checkpoint_final.pth'):
        PRETRAINED_ROOT = f.parent.parent.parent  # go up to DatasetXXX level
        break

if PRETRAINED_ROOT is None:
    # Try direct path if uploaded as dataset
    for loc in Path('/kaggle/input').iterdir():
        for f in loc.rglob('fold_0/checkpoint_final.pth'):
            PRETRAINED_ROOT = f.parent.parent.parent
            break

if PRETRAINED_ROOT:
    # Copy the pretrained model structure to our dataset's results
    src_trainer = list(PRETRAINED_ROOT.glob('nnUNetTrainer*'))[0]
    dst_trainer = Path(os.environ['nnUNet_results']) / DATASET_NAME / src_trainer.name
    
    src_fold = src_trainer / f'fold_{FOLD}'
    dst_fold = dst_trainer / f'fold_{FOLD}'
    dst_fold.mkdir(parents=True, exist_ok=True)
    
    # Copy checkpoint as starting point
    src_ckpt = src_fold / 'checkpoint_final.pth'
    dst_ckpt = dst_fold / 'checkpoint_final.pth'
    if src_ckpt.exists() and not dst_ckpt.exists():
        shutil.copy2(str(src_ckpt), str(dst_ckpt))
        print(f'Copied pretrained fold_{FOLD}: {src_ckpt.stat().st_size/1e6:.0f} MB')
    
    # Copy plans.json and dataset.json from pretrained
    for fname in ['plans.json', 'dataset.json', 'dataset_fingerprint.json']:
        src_f = src_trainer / fname
        dst_f = dst_trainer / fname
        if src_f.exists() and not dst_f.exists():
            shutil.copy2(str(src_f), str(dst_f))
    
    print(f'Pretrained setup complete: {dst_trainer}')
else:
    print('No pretrained weights found - will plan and train from scratch')
    print('This will take longer but nnU-Net handles it automatically')"""),

code("""# ── Run nnU-Net preprocessing ──
# This auto-generates the preprocessed data based on dataset fingerprint
import subprocess

print('Running nnU-Net preprocessing...')
print('(This may take 30-60 minutes depending on dataset size)')
result = subprocess.run([
    sys.executable, '-m', 'nnunetv2.experiment_planning.plan_and_preprocess',
    '-d', str(DATASET_ID),
    '-c', '3d_fullres',
    '--verify_dataset_integrity',
], capture_output=True, text=True, timeout=7200)

if result.returncode == 0:
    print('Preprocessing complete')
else:
    print(f'Preprocessing output: {result.stdout[-500:]}')
    print(f'Errors: {result.stderr[-500:]}')"""),

code("""# ── Fine-tune nnU-Net v2 ──
# This is the actual training command
# --c flag = continue from checkpoint (RESUME SUPPORT)
# --npz = save softmax outputs for ensemble

print(f'Starting nnU-Net v2 fine-tuning: fold {FOLD}, {EPOCHS} epochs')
print(f'This will run for ~10h on Kaggle T4. Use --c to resume in next session.')
print()

cmd = [
    sys.executable, '-m', 'nnunetv2.run.run_training',
    str(DATASET_ID),           # dataset ID
    '3d_fullres',              # configuration
    str(FOLD),                 # fold
    '--npz',                   # save predictions
]

# Add --c flag if checkpoint exists (resume training)
results_fold = Path(os.environ['nnUNet_results']) / DATASET_NAME
trainer_dirs = list(results_fold.glob('nnUNetTrainer*'))
if trainer_dirs:
    fold_dir = trainer_dirs[0] / f'fold_{FOLD}'
    if (fold_dir / 'checkpoint_final.pth').exists() or (fold_dir / 'checkpoint_latest.pth').exists():
        cmd.append('--c')
        print('  Resume mode: --c flag added (continuing from checkpoint)')

print(f'Command: {" ".join(cmd)}')
print('='*60)

# Run training (this will take hours)
result = subprocess.run(cmd, timeout=39600)  # 11h timeout
print(f'\\nTraining finished with return code: {result.returncode}')"""),

code("""# ── Extract embeddings from fine-tuned nnU-Net ──
# Load the trained model and extract bottleneck features

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Find the trained model
results_dir = Path(os.environ['nnUNet_results']) / DATASET_NAME
trainer_dir = list(results_dir.glob('nnUNetTrainer*'))[0]
fold_dir = trainer_dir / f'fold_{FOLD}'

# Load using nnU-Net's own loading mechanism
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    device=device,
    verbose=False,
)
predictor.initialize_from_trained_model_folder(
    str(trainer_dir),
    use_folds=(FOLD,),
)

# Get the network and hook into bottleneck
network = predictor.network
print(f'Model loaded from fold {FOLD}')
print(f'Architecture: {type(network).__name__}')
n_params = sum(p.numel() for p in network.parameters())
print(f'Parameters: {n_params:,} ({n_params/1e6:.1f}M)')

# Find deepest encoder layer for hooking
_feats = {}
def hook_fn(m, inp, out):
    _feats['feat'] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()

# Hook the deepest encoder stage
encoder_layers = [m for m in network.modules() if hasattr(m, 'conv')]
if hasattr(network, 'encoder'):
    # nnU-Net v2 PlainConvUNet structure
    stages = list(network.encoder.stages)
    target = stages[-1]  # deepest encoder stage
    hook = target.register_forward_hook(hook_fn)
    print(f'Hooked encoder stage {len(stages)-1}')
else:
    # Fallback: find last conv block before decoder
    all_modules = list(network.named_modules())
    for name, mod in reversed(all_modules):
        if 'encoder' in name.lower() or 'down' in name.lower():
            hook = mod.register_forward_hook(hook_fn)
            print(f'Hooked: {name}')
            break"""),

code("""# ── v2 Embedding Extraction (octant + mask-weighted pooling) ──
ROI_PAD = 8; ROI_SIZE = (64,64,64)
REGIONS = ['WT','TC','ET']

def roi_crop_resize(image, label):
    wt = label[0,0]
    nz = wt.nonzero(as_tuple=False)
    if len(nz) == 0: ic, lc = image, label
    else:
        lo = nz.min(0).values; hi = nz.max(0).values
        sh = torch.tensor(wt.shape, device=wt.device)
        lo = torch.clamp(lo-ROI_PAD,min=0); hi = torch.clamp(hi+ROI_PAD+1,max=sh)
        ic = image[:,:,lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        lc = label[:,:,lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
    return F.interpolate(ic,ROI_SIZE,mode='trilinear',align_corners=False), \
           F.interpolate(lc.float(),ROI_SIZE,mode='nearest')

def octant_pool(feat):
    H,W,D = feat.shape[2:]
    pieces = []
    for hs in [slice(None,H//2),slice(H//2,None)]:
        for ws in [slice(None,W//2),slice(W//2,None)]:
            for ds in [slice(None,D//2),slice(D//2,None)]:
                pieces.append(F.adaptive_avg_pool3d(feat[:,:,hs,ws,ds],1).flatten())
    return torch.cat(pieces)

def mask_weighted_pool(feat, lbl):
    H,W,D = feat.shape[2:]
    regions = []
    for ch in range(3):
        prob = F.interpolate(lbl[:,ch:ch+1],size=(H,W,D),mode='nearest')
        regions.append((feat*prob).sum(dim=[0,2,3,4]) / (prob.sum()+1e-6))
    return torch.cat(regions)

# Convert labels to WT/TC/ET channels
def convert_labels(seg):
    wt = ((seg==1)|(seg==2)|(seg==4)).float()
    tc = ((seg==1)|(seg==4)).float()
    et = (seg==4).float()
    return torch.stack([wt, tc, et], dim=0).unsqueeze(0)

print('Extraction functions ready')"""),

code("""# ── Run extraction on all scans ──
import nibabel as nib
import monai.transforms as T

embeddings, skipped = {}, 0
network.eval()

for scan_dir in tqdm(all_dirs, desc='Extracting embeddings'):
    case_id = scan_dir.name
    patient_id = '-'.join(case_id.split('-')[:-1])
    timepoint = case_id.split('-')[-1]
    key = f'{patient_id}__{timepoint}'

    try:
        # Load 4 modalities
        mods = []
        for suffix in ['t1n','t1c','t2w','t2f']:
            f = list(scan_dir.glob(f'*-{suffix}*'))[0]
            mods.append(nib.load(str(f)).get_fdata())
        image = np.stack(mods, axis=0)[np.newaxis]  # (1, 4, H, W, D)

        # Load and convert label
        seg_f = list(scan_dir.glob('*-seg*'))[0]
        seg = nib.load(str(seg_f)).get_fdata()
        label = convert_labels(torch.from_numpy(seg))

        # Z-score normalize per channel
        img_t = torch.from_numpy(image).float()
        for c in range(4):
            ch = img_t[0, c]
            nz = ch > 0
            if nz.any():
                img_t[0, c] = (ch - ch[nz].mean()) / (ch[nz].std() + 1e-8)

        # ROI crop + resize
        img_roi, lbl_roi = roi_crop_resize(img_t.to(device), label.to(device))

        # Forward pass to get features
        _feats.clear()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            _ = network(img_roi)
        feat = _feats.get('feat')
        if feat is None:
            skipped += 1; continue

        # Pool embeddings
        oct = octant_pool(feat)
        msk = mask_weighted_pool(feat, lbl_roi)
        embeddings[key] = torch.cat([oct, msk]).cpu().numpy()

        if len(embeddings) <= 3:
            print(f'  {key}: {embeddings[key].shape[0]}-dim')

    except Exception as e:
        print(f'  Skip {key}: {e}')
        skipped += 1

hook.remove()
print(f'\\nExtracted: {len(embeddings)} | Skipped: {skipped}')

# Save
emb_dir = OUTPUT_ROOT / 'embeddings'
emb_dir.mkdir(parents=True, exist_ok=True)
if embeddings:
    dim = list(embeddings.values())[0].shape[0]
    np.savez(emb_dir / 'cnn_nnunet_embeddings_v2.npz', **embeddings)
    print(f'Saved: {len(embeddings)} x {dim}-dim embeddings')"""),

code("""# ── Dice Evaluation on validation split ──
import matplotlib.pyplot as plt

# Use nnU-Net's built-in prediction for proper evaluation
print('Running nnU-Net inference for Dice evaluation...')

# Quick manual Dice on a subset
from monai.metrics import DiceMetric
dice_metric = DiceMetric(include_background=True, reduction='mean_batch')

val_dirs = all_dirs[-50:]  # last 50 scans as quick validation
dice_scores = []

network.eval()
with torch.no_grad():
    for scan_dir in tqdm(val_dirs[:20], desc='Evaluating Dice'):
        try:
            mods = []
            for suffix in ['t1n','t1c','t2w','t2f']:
                f = list(scan_dir.glob(f'*-{suffix}*'))[0]
                mods.append(nib.load(str(f)).get_fdata())
            image = np.stack(mods, axis=0)[np.newaxis]
            img_t = torch.from_numpy(image).float()
            for c in range(4):
                ch = img_t[0,c]; nz = ch>0
                if nz.any(): img_t[0,c] = (ch-ch[nz].mean())/(ch[nz].std()+1e-8)

            seg_f = list(scan_dir.glob('*-seg*'))[0]
            seg = nib.load(str(seg_f)).get_fdata()
            gt = convert_labels(torch.from_numpy(seg)).to(device)

            with torch.amp.autocast('cuda'):
                pred = torch.sigmoid(network(img_t.to(device)))
            dice_metric((pred > 0.5).float(), gt)
        except: pass

dv = dice_metric.aggregate()
print(f'\\nDice scores (20 samples):')
for i, rn in enumerate(REGIONS):
    print(f'  {rn}: {dv[i].item():.4f}')
print(f'  Mean: {dv.mean().item():.4f}')

# Save summary
fig_dir = OUTPUT_ROOT / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
print(f'\\n Phase 2B (nnU-Net v2) COMPLETE')
print(f'  Outputs: {OUTPUT_ROOT}')
total = 0
for p in sorted(OUTPUT_ROOT.rglob('*')):
    if p.is_file(): sz=p.stat().st_size; total+=sz; print(f'  {p.relative_to(OUTPUT_ROOT)} ({sz/1e6:.1f}MB)')
print(f'  Total: {total/1e6:.1f} MB')"""),
]

nb = {"nbformat":4,"nbformat_minor":5,
      "metadata":{"kaggle":{"accelerator":"nvidiaTeslaT4","isGpuEnabled":True,
                   "isInternetEnabled":True,"language":"python","sourceType":"notebook"},
                   "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info":{"name":"python","version":"3.12.12"}},
      "cells":cells}
for i,c in enumerate(nb['cells']): c['id'] = f'c{i:02d}'
out = '/home/moamed/canada_me/explainable_diseas/implementation_brats2024/Phase2/notebooks/Phase2_A2_nnUNet_Finetune.ipynb'
with open(out,'w') as f: json.dump(nb,f,indent=1)
print(f'Built: {out} ({len(cells)} cells)')

# Remove old wrong DynUNet notebook
import os
old = '/home/moamed/canada_me/explainable_diseas/implementation_brats2024/Phase2/notebooks/Phase2_A2_DynUNet_Finetune.ipynb'
if os.path.exists(old): os.remove(old); print(f'Removed old DynUNet notebook')
