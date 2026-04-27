"""
Standalone builder for Phase2_A1_SegResNet_Finetune.ipynb
Run: python3 patch_segresnet_notebook.py
Upload the output .ipynb to Kaggle and run in a FRESH session.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / 'notebooks' / 'Phase2_A1_SegResNet_Finetune.ipynb'

# ── helpers ──────────────────────────────────────────────────
def md(src, cid):
    return {"cell_type":"markdown","metadata":{},"source":src,"id":cid}

def code(src, cid):
    # strip leading newline if present
    src = src.lstrip('\n')
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src,"id":cid}

# ═══════════════════════════════════════════════════════════════
# CELL 01 — Config
# ═══════════════════════════════════════════════════════════════
C01 = """
from pathlib import Path
import os, warnings
warnings.filterwarnings('ignore')

MODEL_NAME  = 'segresnet'
PATCH       = [128, 128, 128]
REGIONS     = ['WT', 'TC', 'ET', 'RC']   # BraTS 2024 Post-Treatment
OUTPUT_ROOT = Path('/kaggle/working/phase2_segresnet')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f'Model: {MODEL_NAME} | Patch: {PATCH} | Regions: {REGIONS}')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 02 — Imports
# ═══════════════════════════════════════════════════════════════
C02 = """
import subprocess, sys, json, time, math, os, shutil
import numpy as np
import torch
import torch.nn.functional as F

try:
    import monai
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'monai[all]', '-q'])
    import monai

import monai.transforms as T
from monai.data import Dataset, CacheDataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.transforms import MapTransform

set_determinism(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'MONAI {monai.__version__} | PyTorch {torch.__version__} | Device: {device}')
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)} | Total memory: {total_mem:.1f} GB')

# Disk space check — PersistentDataset on 297 scans = ~60GB, kills Kaggle (20GB limit)
usage = shutil.disk_usage('/kaggle/working')
free_gb = usage.free / 1e9
print(f'Disk free: {free_gb:.1f} GB (need ~2GB for checkpoints)')
if free_gb < 5:
    print('WARNING: low disk space! Clear /kaggle/working or reduce data.')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 03 — Label mapping (BraTS 2024 Post-Treatment)
# ═══════════════════════════════════════════════════════════════
C03 = """
# BraTS 2024 Post-Treatment: 0=BG, 1=NCR, 2=ED, 3=RC, 4=ET
# WT = 1+2+3+4 | TC = 1+3+4 | ET = 4 | RC = 3  ->  4 output channels

class ConvertToMultiChannelBratsPostTreatmentd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            result = [
                (img==1)|(img==2)|(img==3)|(img==4),  # WT
                (img==1)|(img==3)|(img==4),             # TC
                img==4,                                  # ET
                img==3,                                  # RC
            ]
            d[key] = (torch.stack(result, 0).float()
                      if isinstance(img, torch.Tensor)
                      else np.stack(result, 0).astype(np.float32))
        return d

ConvertToMultiChannelBratsGliomad = ConvertToMultiChannelBratsPostTreatmentd
print('Label: WT=1+2+3+4 | TC=1+3+4 | ET=4 | RC=3  ->  4 channels OK')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 04 — Data loading (path discovery, no HDD paths)
# ═══════════════════════════════════════════════════════════════
C04 = """
# .nii_gz -> .nii.gz symlink resolver
SYMLINK_DIR = Path('/kaggle/working/nifti_links')

def setup_nii_gz_symlinks(data_dir):
    count = 0
    for nii_gz in Path(data_dir).rglob('*.nii_gz'):
        real_name = nii_gz.name.replace('.nii_gz', '.nii.gz')
        link = SYMLINK_DIR / nii_gz.parent.name / real_name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(str(nii_gz), str(link))
            count += 1
    return count

# Step 1: Create symlinks
DATA_ROOT = Path('/kaggle/input')
for ds_dir in DATA_ROOT.iterdir():
    if not ds_dir.is_dir(): continue
    if list(ds_dir.rglob('*.nii_gz')):
        n = setup_nii_gz_symlinks(ds_dir)
        if n: print(f'  Created {n} symlinks in {ds_dir.name}')

# Step 2: Find NIFTI_ROOT
NIFTI_ROOT = None
for search_root in [SYMLINK_DIR, DATA_ROOT]:
    if not search_root.exists(): continue
    for c in search_root.rglob('BraTS-GLI-*'):
        if c.is_dir():
            NIFTI_ROOT = c.parent
            break
    if NIFTI_ROOT: break

if NIFTI_ROOT is None:
    raise RuntimeError('No BraTS-GLI-* folders found — check dataset attachments')
print(f'NIFTI_ROOT: {NIFTI_ROOT}')

# Step 3: Split metadata ONLY from scan_index (never file paths — those are local HDD paths!)
split_map, scan_meta = {}, {}
for f in DATA_ROOT.rglob('scan_index.json'):
    si = json.load(open(f))
    for s in si.get('training_scans', []):
        pid, sid = s['patient_id'], s['scan_id']
        split = s.get('split')
        if split: split_map[pid] = split
        scan_meta[sid] = {'patient_id': pid, 'timepoint': s.get('timepoint','100'), 'split': split}
    print(f'  Split metadata: {len(split_map)} patients from scan_index.json')
    break

# Step 4: Discover file paths fresh from disk (machine-independent)
all_dirs = sorted([d for d in NIFTI_ROOT.iterdir() if d.is_dir() and 'BraTS-GLI' in d.name])
training_scans = []
for d in all_dirs:
    files = {m: list(d.glob(f'*-{m}*')) for m in ['t1n','t1c','t2w','t2f']}
    seg   = list(d.glob('*-seg*'))
    if not (all(files[m] for m in files) and seg): continue
    name  = d.name
    pid   = name.rsplit('-', 1)[0]
    tp    = name.rsplit('-', 1)[1] if '-' in name else '100'
    split = scan_meta.get(name, {}).get('split') or split_map.get(pid)
    training_scans.append({
        'scan_id': name, 'patient_id': pid, 'timepoint': tp,
        't1n': str(files['t1n'][0]), 't1c': str(files['t1c'][0]),
        't2w': str(files['t2w'][0]), 't2f': str(files['t2f'][0]),
        'seg': str(seg[0]), 'split': split,
    })
print(f'Total scans discovered: {len(training_scans)}')

# Step 5: Train/val split
if any(s['split'] for s in training_scans):
    train_scans = [s for s in training_scans if s.get('split') == 'train']
    val_scans   = [s for s in training_scans if s.get('split') == 'val']
    known_train = {s['patient_id'] for s in train_scans}
    known_val   = {s['patient_id'] for s in val_scans}
    for s in training_scans:
        if s.get('split'): continue
        if   s['patient_id'] in known_train: train_scans.append(s)
        elif s['patient_id'] in known_val:   val_scans.append(s)
        else: train_scans.append(s)
else:
    from collections import defaultdict
    pts = defaultdict(list)
    for s in training_scans: pts[s['patient_id']].append(s)
    pids = sorted(pts.keys())
    n80  = int(0.8 * len(pids))
    train_pids = set(pids[:n80]); val_pids = set(pids[n80:])
    train_scans = [s for s in training_scans if s['patient_id'] in train_pids]
    val_scans   = [s for s in training_scans if s['patient_id'] in val_pids]

tp = len({s['patient_id'] for s in train_scans})
vp = len({s['patient_id'] for s in val_scans})
print(f'Train: {len(train_scans)} scans ({tp} patients)')
print(f'Val:   {len(val_scans)} scans ({vp} patients)')
print('All paths resolved live from disk - no local HDD paths')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 05 — Transforms
# ═══════════════════════════════════════════════════════════════
C05 = """
patch = [128, 128, 128]
train_transforms = T.Compose([
    T.LoadImaged(keys=['image','label']),
    T.EnsureChannelFirstd(keys=['image','label']),
    T.EnsureTyped(keys=['image','label']),
    T.Orientationd(keys=['image','label'], axcodes='RAS'),
    T.CropForegroundd(keys=['image','label'], source_key='image', allow_smaller=True),
    T.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ConvertToMultiChannelBratsGliomad(keys=['label']),
    T.RandFlipd(keys=['image','label'], spatial_axis=[0], prob=0.5),
    T.RandFlipd(keys=['image','label'], spatial_axis=[1], prob=0.5),
    T.RandFlipd(keys=['image','label'], spatial_axis=[2], prob=0.5),
    T.RandScaleIntensityd(keys='image', factors=0.1, prob=0.3),
    T.RandShiftIntensityd(keys='image', offsets=0.1, prob=0.3),
    T.SpatialPadd(keys=['image','label'], spatial_size=patch),
    T.RandCropByPosNegLabeld(keys=['image','label'], label_key='label',
        spatial_size=patch, pos=2, neg=1, num_samples=2, image_key='image', image_threshold=0),
    T.EnsureTyped(keys=['image','label'], dtype=torch.float32),
])
val_transforms = T.Compose([
    T.LoadImaged(keys=['image','label']),
    T.EnsureChannelFirstd(keys=['image','label']),
    T.EnsureTyped(keys=['image','label']),
    T.Orientationd(keys=['image','label'], axcodes='RAS'),
    T.CropForegroundd(keys=['image','label'], source_key='image', allow_smaller=True),
    T.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ConvertToMultiChannelBratsGliomad(keys=['label']),
    T.EnsureTyped(keys=['image','label'], dtype=torch.float32),
])
print('Transforms ready')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 06 — Build dicts with gzip validation
# NOTE: use integer comparison for gzip magic to avoid byte literal issues
# ═══════════════════════════════════════════════════════════════
C06 = """
import nibabel as nib

def is_valid_gzip(path):
    # Check gzip magic bytes (0x1f 0x8b) and file size > 1KB
    try:
        with open(str(path), 'rb') as f:
            h = f.read(10)
            if len(h) < 10 or h[0] != 31 or h[1] != 139:
                return False
            f.seek(0, 2)
            return f.tell() > 1024
    except Exception:
        return False

def validate_scan(s):
    # Fast gzip header check + nibabel header (no full decompress)
    try:
        for key in ['t1n', 't1c', 't2w', 't2f', 'seg']:
            if not is_valid_gzip(s[key]):
                return False
            _ = nib.load(s[key]).shape
        return True
    except Exception:
        return False

def build_dicts(scan_list):
    dicts, bad = [], []
    for s in scan_list:
        if not validate_scan(s):
            bad.append(s['scan_id'])
            continue
        dicts.append({
            'image': [s['t1n'], s['t1c'], s['t2w'], s['t2f']],
            'label': s['seg'],
            'patient_id': s['patient_id'],
            'timepoint':  s['timepoint'],
        })
    if bad:
        print(f'  Skipped {len(bad)} corrupted scans: {bad[:3]}{"..." if len(bad)>3 else ""}')
    return dicts

print('Validating scans (gzip header check)...')
train_dicts = build_dicts(train_scans)
val_dicts   = build_dicts(val_scans)
print(f'Train dicts: {len(train_dicts)} | Val dicts: {len(val_dicts)}')
if train_dicts:
    print(f'  Sample: {train_dicts[0]["image"][0]}')
    print(f'  Exists: {Path(train_dicts[0]["image"][0]).exists()}')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 07 — Model
# ═══════════════════════════════════════════════════════════════
C07 = """
from monai.networks.nets import SegResNet
import urllib.request, zipfile

model = SegResNet(
    spatial_dims=3, in_channels=4, out_channels=4,
    init_filters=32, blocks_down=(1,2,2,4), blocks_up=(1,1,1), dropout_prob=0.2
)
print(f'SegResNet: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

# PyTorch 2.6+ defaults weights_only=True which blocks numpy in checkpoints
# Try safe load first, then allow numpy globals, then fall back to weights_only=False
def safe_torch_load(path):
    import numpy
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        pass
    try:
        safe = [numpy.core.multiarray.scalar, numpy.dtype, numpy.ndarray]
        with torch.serialization.safe_globals(safe):
            return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        pass
    return torch.load(path, map_location='cpu', weights_only=False)

# Search ALL of /kaggle/input for pretrained weights (.pt / .pth)
print('Searching for pretrained weights in /kaggle/input ...')
ckpt_found = None
for ext in ['*.pt', '*.pth']:
    for ckpt in Path('/kaggle/input').rglob(ext):
        ckpt_found = ckpt
        break
    if ckpt_found: break

if ckpt_found:
    print(f'  Found: {ckpt_found}')
    try:
        state = safe_torch_load(ckpt_found)
        state = state.get('state_dict', state.get('model', state))
        own   = model.state_dict()
        compat = {k: v for k,v in state.items() if k in own and own[k].shape==v.shape}
        model.load_state_dict({**own, **compat})
        print(f'Pretrained: loaded {len(compat)}/{len(own)} layers from {ckpt_found.name}')
    except Exception as e:
        print(f'Could not load weights: {e} - training from scratch')
else:
    print('No pretrained weights found - training from scratch')

model = model.to(device)
"""

# ═══════════════════════════════════════════════════════════════
# CELL 08 — Training
# KEY FIXES:
#   1. iter/next loop catches DataLoader WORKER exceptions (EOFError)
#      - plain `for batch in loader` re-raises BEFORE the try block!
#   2. SmartCacheDataset: 5% RAM cache = ~9GB safe for Kaggle T4
#   3. Auto-resume from checkpoint
# ═══════════════════════════════════════════════════════════════
C08 = """
from torch.cuda.amp import GradScaler, autocast

CKPT_DIR    = OUTPUT_ROOT / 'checkpoints'; CKPT_DIR.mkdir(exist_ok=True)
BEST_PATH   = CKPT_DIR / 'segresnet_best.pth'
LATEST_PATH = CKPT_DIR / 'segresnet_latest.pth'

def get_lr(ep, total, base=1e-4):
    warm = 5
    if ep < warm: return base * (ep+1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (ep-warm)/(total-warm)))

def safe_loader_iter(loader):
    # iter/next catches DataLoader WORKER exceptions (EOFError, gzip truncation)
    # 'for batch in loader' re-raises worker errors BEFORE entering the loop body
    it = iter(loader)
    SKIP_KEYS = ['EOFError','gzip','Compressed file','end-of-stream','worker','corrupt']
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as e:
            if any(k in str(e) for k in SKIP_KEYS):
                continue   # skip bad file, keep going
            raise          # re-raise real PyTorch errors

def train_model(model, lr=1e-4, epochs=30, patience=10, val_interval=4):
    # Auto-resume
    start_ep, best_dice, metrics_log = 0, 0.0, {'dice':[], 'per_region':[], 'loss':[]}
    if LATEST_PATH.exists():
        lc = torch.load(LATEST_PATH, map_location='cpu')
        model.load_state_dict(lc['model'])
        start_ep   = lc.get('epoch', 0) + 1
        best_dice  = lc.get('best_dice', 0)
        metrics_log = lc.get('metrics', metrics_log)
        print(f'Resumed from epoch {start_ep-1}, best_dice={best_dice:.4f}')
        if start_ep >= epochs:
            return model, best_dice, metrics_log

    loss_fn     = DiceLoss(to_onehot_y=False, sigmoid=True, smooth_nr=0, smooth_dr=1e-5)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler      = GradScaler()
    dice_metric = DiceMetric(include_background=True, reduction='mean_batch')
    no_improve  = 0
    t0          = time.time()

    # Dataset strategy:
    # Train: CacheDataset 5% cache (~66 scans in RAM) - stable API across all MONAI versions
    # Val:   Plain Dataset ONLY - no disk cache (PersistentDataset = ~60GB -> kills Kaggle disk)
    try:
        from monai.data import CacheDataset
        train_ds = CacheDataset(train_dicts, train_transforms, cache_rate=0.05, num_workers=4)
        print('Using CacheDataset (5% RAM cache)')
    except Exception:
        train_ds = Dataset(train_dicts, train_transforms)
        print('Using plain Dataset (CacheDataset unavailable)')
    val_ds = Dataset(val_dicts, val_transforms)  # plain disk read - NO storage bloat
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f'Train: {len(train_loader)} batches | Val: {len(val_loader)} batches')
    print(f'Expected ~26 min/epoch x {epochs} epochs ~ {epochs*26/60:.1f}h total')

    for ep in range(start_ep, epochs):
        model.train()
        cur_lr = get_lr(ep, epochs, lr)
        for pg in optimizer.param_groups: pg['lr'] = cur_lr

        ep_loss, n_ok, n_bad = 0.0, 0, 0
        for batch in safe_loader_iter(train_loader):
            try:
                imgs = batch['image'].to(device)
                lbls = batch['label'].to(device)
                optimizer.zero_grad()
                with autocast():
                    loss = loss_fn(model(imgs), lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                ep_loss += loss.item(); n_ok += 1
            except Exception:
                n_bad += 1
        avg_loss = ep_loss / max(n_ok, 1)
        bad_str  = f' | skipped {n_bad}' if n_bad else ''
        metrics_log['loss'].append(avg_loss)

        if (ep+1) % val_interval == 0 or ep == epochs-1:
            model.eval(); dice_metric.reset()
            with torch.no_grad():
                for vb in safe_loader_iter(val_loader):
                    try:
                        vo = sliding_window_inference(vb['image'].to(device), PATCH, 4, model, overlap=0.5)
                        dice_metric((torch.sigmoid(vo)>0.5).float(), vb['label'].to(device))
                    except Exception:
                        pass
            dv = dice_metric.aggregate(); md = dv.mean().item()
            pr = [round(dv[i].item(),4) for i in range(4)]
            metrics_log['dice'].append(md); metrics_log['per_region'].append(pr)
            tag = ' NEW BEST' if md > best_dice else ''
            print(f'Ep {ep:3d} | L={avg_loss:.4f} | Dice={md:.4f} WT={pr[0]:.3f} TC={pr[1]:.3f} ET={pr[2]:.3f} RC={pr[3]:.3f} | {(time.time()-t0)/60:.1f}m{tag}{bad_str}')
            if md > best_dice:
                best_dice = md; no_improve = 0
                torch.save({'model': model.state_dict(), 'epoch': ep, 'best_dice': best_dice}, BEST_PATH)
            else:
                no_improve += val_interval
        else:
            print(f'Ep {ep:3d} | L={avg_loss:.4f} | LR={cur_lr:.2e} | {(time.time()-t0)/60:.1f}m{bad_str}')

        torch.save({'model': model.state_dict(), 'epoch': ep,
                    'best_dice': best_dice, 'metrics': metrics_log}, LATEST_PATH)
        if no_improve >= patience:
            print(f'Early stop at ep {ep}'); break

    print(f'Done. Best Dice = {best_dice:.4f}')
    return model, best_dice, metrics_log

print('Fine-tuning SegResNet on BraTS 2024 Post-Treatment...')
# 30 epochs x ~26min = ~13h -- use auto-resume for SegResNet (2 sessions)
model, best_dice, metrics = train_model(model, lr=1e-4, epochs=30, patience=10)
"""

# ═══════════════════════════════════════════════════════════════
# CELL 09 — Embedding extraction
# ═══════════════════════════════════════════════════════════════
C09 = """
ROI_PAD  = 8
ROI_SIZE = (64, 64, 64)

def roi_crop_resize(image, label):
    wt = label[0, 0]
    nz = wt.nonzero(as_tuple=False)
    if len(nz) == 0:
        ic, lc = image, label
    else:
        lo = nz.min(0).values; hi = nz.max(0).values
        sh = torch.tensor(wt.shape, device=wt.device)
        lo = torch.clamp(lo - ROI_PAD, min=0)
        hi = torch.clamp(hi + ROI_PAD + 1, max=sh)
        ic = image[:, :, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        lc = label[:, :, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return (F.interpolate(ic, ROI_SIZE, mode='trilinear', align_corners=False),
            F.interpolate(lc.float(), ROI_SIZE, mode='nearest'))

def octant_pool(feat):
    H, W, D = feat.shape[2:]
    return torch.cat([feat[:,:,hs,ws,ds].mean(dim=[0,2,3,4])
                      for hs in [slice(None,H//2),slice(H//2,None)]
                      for ws in [slice(None,W//2),slice(W//2,None)]
                      for ds in [slice(None,D//2),slice(D//2,None)]])

def mask_pool(feat, lbl):
    H, W, D = feat.shape[2:]
    return torch.cat([(feat * F.interpolate(lbl[:,c:c+1], (H,W,D), mode='nearest')).sum([0,2,3,4]) /
                      (F.interpolate(lbl[:,c:c+1], (H,W,D), mode='nearest').sum() + 1e-6)
                      for c in range(4)])

def extract_embeddings(model):
    model.eval()
    _f = {}
    hook = None
    for name, mod in model.named_modules():
        if hasattr(mod, 'convs') and 'down_layers' in name:
            hook = mod.register_forward_hook(lambda m,i,o: _f.update({'enc': o.detach()}))
            break

    emb_dir = OUTPUT_ROOT / 'embeddings'; emb_dir.mkdir(exist_ok=True)
    embs, ids, tps = [], [], []
    ds     = Dataset(train_dicts + val_dicts, val_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad():
        for i, batch in enumerate(safe_loader_iter(loader)):
            try:
                img = batch['image'].to(device)
                lbl = batch['label'].to(device)
                _f.clear(); _ = model(img)
                if 'enc' not in _f: continue
                ic, lc = roi_crop_resize(img, lbl)
                fr = F.interpolate(_f['enc'], ROI_SIZE, mode='trilinear', align_corners=False)
                embs.append(torch.cat([octant_pool(fr), mask_pool(fr, lc)]).cpu().numpy())
                ids.append(batch['patient_id'][0])
                tps.append(batch['timepoint'][0])
            except Exception:
                continue
            if (i+1) % 100 == 0: print(f'  {i+1}/{len(ds)} extracted')

    if hook: hook.remove()
    arr = np.array(embs)
    out = emb_dir / 'cnn_segresnet_embeddings_v2.npz'
    np.savez_compressed(out, embeddings=arr, patient_ids=ids, timepoints=tps)
    print(f'Embeddings saved: {arr.shape} -> {out}')
    return arr

import gc
print('Cleaning VRAM before embedding extraction...')
if 'optimizer' in globals(): del optimizer
if 'scaler' in globals(): del scaler
gc.collect()
torch.cuda.empty_cache()

print('Extracting embeddings...')
embeddings = extract_embeddings(model)
"""

# ═══════════════════════════════════════════════════════════════
# CELL 10 — Summary
# ═══════════════════════════════════════════════════════════════
C10 = """
import json as _j
summary = {
    'model': MODEL_NAME, 'best_dice': float(best_dice),
    'regions': REGIONS, 'label': '{0=BG,1=NCR,2=ED,3=RC,4=ET}',
    'train_scans': len(train_dicts), 'val_scans': len(val_dicts),
}
(OUTPUT_ROOT / 'summary.json').write_text(_j.dumps(summary, indent=2))
print('=' * 50)
print(f'  SegResNet done  |  Best Dice: {best_dice:.4f}')
print(f'  Outputs: {OUTPUT_ROOT}')
print('=' * 50)
print('Next: run Phase2_B1_CNN_Embedding_Comparison.ipynb')
"""

# ═══════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════
cells = [
    md("# Phase 2A — SegResNet Fine-Tuning (BraTS 2024 Post-Treatment)\n"
       "## 4 output channels: WT / TC / ET / RC\n"
       "Label 3 = Resection Cavity (RC) — present in 75% of post-treatment scans.", "c00"),
    code(C01, "c01"),
    code(C02, "c02"),
    code(C03, "c03"),
    code(C04, "c04"),
    code(C05, "c05"),
    code(C06, "c06"),
    code(C07, "c07"),
    code(C08, "c08"),
    code(C09, "c09"),
    code(C10, "c10"),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.12.0"}
    },
    "cells": cells,
}

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Written: {NB_PATH}")
print(f"Cells: {len(cells)}")
