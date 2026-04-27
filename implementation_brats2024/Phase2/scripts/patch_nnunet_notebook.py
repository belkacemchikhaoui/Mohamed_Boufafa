"""
Standalone builder for Phase2_A2_nnUNet_Finetune.ipynb
Run: python3 patch_nnunet_notebook.py
Upload the output .ipynb to Kaggle (fresh session).

Pretrained ckpt expected at:
  /kaggle/input/.../nnunet_v2/Dataset002_BRATS19/
    nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth
Target BraTS 2021 performance (single fold):
  WT=0.9005  TC=0.8673  ET=0.8509
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / 'notebooks' / 'Phase2_A2_nnUNet_Finetune.ipynb'

def md(src, cid):
    return {"cell_type":"markdown","metadata":{},"source":src,"id":cid}

def code(src, cid):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.lstrip('\n'),"id":cid}

# ═══════════════════════════════════════════════════════════════
# CELL 01 — Config
# ═══════════════════════════════════════════════════════════════
C01 = """
from pathlib import Path
import os, warnings
warnings.filterwarnings('ignore')

MODEL_NAME  = 'nnunet'
PATCH       = [128, 128, 128]
# 3-channel output matching BraTS 2021 pretrained checkpoint (no RC)
# Output order: WT, TC, ET  (same region-based sigmoid as nnUNet)
REGIONS     = ['WT', 'TC', 'ET']
OUTPUT_ROOT = Path('/kaggle/working/phase2_nnunet')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f'Model: {MODEL_NAME} | Patch: {PATCH} | Regions: {REGIONS}')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 02 — Imports  (install nnunetv2 if needed)
# ═══════════════════════════════════════════════════════════════
C02 = """
import subprocess, sys, json, time, math, os, shutil
import numpy as np
import torch
import torch.nn.functional as F

# Install nnunetv2 (needed to reconstruct exact architecture from plans)
try:
    import nnunetv2
    try:
        ver = nnunetv2.__version__
    except AttributeError:
        import importlib.metadata
        try: ver = importlib.metadata.version('nnunetv2')
        except Exception: ver = 'installed'
    print(f'nnunetv2 {ver} ready')
except ImportError:
    print('Installing nnunetv2 ...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nnunetv2', '-q'])
    import nnunetv2
    print('nnunetv2 installed')

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

# Disk space check
usage = shutil.disk_usage('/kaggle/working')
free_gb = usage.free / 1e9
print(f'Disk free: {free_gb:.1f} GB (need ~2GB for checkpoints)')
if free_gb < 5:
    print('WARNING: low disk space!')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 03 — Label mapping (3-channel, NO RC)
# BraTS 2024 labels: 0=BG, 1=NCR, 2=ED, 3=RC, 4=ET
# We treat RC (label 3) as part of TC / WT  (compatible with BraTS 2021 ckpt)
# Output channels: WT=1+2+3+4 | TC=1+3+4 | ET=4
# ═══════════════════════════════════════════════════════════════
C03 = """
# BraTS 2024 Post-Treatment: 0=BG, 1=NCR, 2=ED, 3=RC, 4=ET
# 3-channel output  (RC merged into WT+TC, matching BraTS 2021 checkpoint format)
# WT = 1+2+3+4  (whole tumor, includes RC)
# TC = 1+3+4    (tumor core = NCR+RC+ET)
# ET = 4        (enhancing tumor only)
# This is compatible with nnUNet BraTS 2021 pretrained weights.

class ConvertToMultiChannelBrats3Chd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            result = [
                (img==1)|(img==2)|(img==3)|(img==4),  # WT (all non-BG)
                (img==1)|(img==3)|(img==4),             # TC (NCR+RC+ET)
                img==4,                                  # ET
            ]
            d[key] = (torch.stack(result, 0).float()
                      if isinstance(img, torch.Tensor)
                      else np.stack(result, 0).astype(np.float32))
        return d

print('Label: WT=1+2+3+4 | TC=1+3+4 | ET=4  (3 channels, no RC)')
print('Compatible with BraTS 2021 nnUNet checkpoint (WT=0.900 TC=0.867 ET=0.851)')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 04 — Data loading (symlinks + fresh path discovery)
# Same robust approach as SegResNet — no HDD paths ever
# ═══════════════════════════════════════════════════════════════
C04 = """
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

DATA_ROOT = Path('/kaggle/input')
for ds_dir in DATA_ROOT.iterdir():
    if not ds_dir.is_dir(): continue
    if list(ds_dir.rglob('*.nii_gz')):
        n = setup_nii_gz_symlinks(ds_dir)
        if n: print(f'  Created {n} symlinks in {ds_dir.name}')

NIFTI_ROOT = None
for search_root in [SYMLINK_DIR, DATA_ROOT]:
    if not search_root.exists(): continue
    for c in search_root.rglob('BraTS-GLI-*'):
        if c.is_dir():
            NIFTI_ROOT = c.parent
            break
    if NIFTI_ROOT: break

if NIFTI_ROOT is None:
    raise RuntimeError('No BraTS-GLI-* folders found - check dataset attachments')
print(f'NIFTI_ROOT: {NIFTI_ROOT}')

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

all_dirs = sorted([d for d in NIFTI_ROOT.iterdir() if d.is_dir() and 'BraTS-GLI' in d.name])
training_scans = []
for d in all_dirs:
    files = {m: list(d.glob(f'*-{m}*')) for m in ['t1n','t1c','t2w','t2f']}
    seg   = list(d.glob('*-seg*'))
    if not (all(files[m] for m in files) and seg): continue
    name = d.name; pid = name.rsplit('-',1)[0]; tp = name.rsplit('-',1)[1] if '-' in name else '100'
    split = scan_meta.get(name, {}).get('split') or split_map.get(pid)
    training_scans.append({
        'scan_id': name, 'patient_id': pid, 'timepoint': tp,
        't1n': str(files['t1n'][0]), 't1c': str(files['t1c'][0]),
        't2w': str(files['t2w'][0]), 't2f': str(files['t2f'][0]),
        'seg': str(seg[0]), 'split': split,
    })
print(f'Total scans: {len(training_scans)} from {NIFTI_ROOT}')

if any(s['split'] for s in training_scans):
    train_scans = [s for s in training_scans if s.get('split')=='train']
    val_scans   = [s for s in training_scans if s.get('split')=='val']
    kt = {s['patient_id'] for s in train_scans}; kv = {s['patient_id'] for s in val_scans}
    for s in training_scans:
        if s.get('split'): continue
        if s['patient_id'] in kt: train_scans.append(s)
        elif s['patient_id'] in kv: val_scans.append(s)
        else: train_scans.append(s)
else:
    from collections import defaultdict
    pts = defaultdict(list)
    for s in training_scans: pts[s['patient_id']].append(s)
    pids = sorted(pts.keys()); n80 = int(0.8*len(pids))
    tp_set = set(pids[:n80]); vp_set = set(pids[n80:])
    train_scans = [s for s in training_scans if s['patient_id'] in tp_set]
    val_scans   = [s for s in training_scans if s['patient_id'] in vp_set]

print(f'Train: {len(train_scans)} scans | Val: {len(val_scans)} scans')
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
    ConvertToMultiChannelBrats3Chd(keys=['label']),
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
    ConvertToMultiChannelBrats3Chd(keys=['label']),
    T.EnsureTyped(keys=['image','label'], dtype=torch.float32),
])
print('Transforms ready (3-channel: WT/TC/ET)')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 06 — Gzip validation + build dicts
# ═══════════════════════════════════════════════════════════════
C06 = """
import nibabel as nib

def is_valid_gzip(path):
    # Check gzip magic bytes (31, 139) and minimum size
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
    try:
        for key in ['t1n','t1c','t2w','t2f','seg']:
            if not is_valid_gzip(s[key]): return False
            _ = nib.load(s[key]).shape
        return True
    except Exception:
        return False

def build_dicts(scan_list):
    dicts, bad = [], []
    for s in scan_list:
        if not validate_scan(s):
            bad.append(s['scan_id']); continue
        dicts.append({
            'image': [s['t1n'],s['t1c'],s['t2w'],s['t2f']],
            'label': s['seg'],
            'patient_id': s['patient_id'],
            'timepoint':  s['timepoint'],
        })
    if bad: print(f'  Skipped {len(bad)} corrupted: {bad[:3]}{"..." if len(bad)>3 else ""}')
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
# CELL 07 — Load nnUNet model from pretrained checkpoint
# Strategy:
#   1. Search /kaggle/input for nnUNetPlans.json + checkpoint_final.pth
#   2. Use nnunetv2 to build EXACT matching architecture → load all weights
#   3. Fallback: MONAI DynUNet with similar spec if nnunetv2 fails
# ═══════════════════════════════════════════════════════════════
C07 = """
print('='*55)
print('  Loading nnUNet v2 pretrained model')
print('  Target: WT=0.9005  TC=0.8673  ET=0.8509 (BraTS 2021)')
print('='*55)

# ── Step 1: Find checkpoint + plans ──
ckpt_path  = None
plans_path = None
for p in Path('/kaggle/input').rglob('checkpoint_final.pth'):
    ckpt_path = p; break
for fname in ['plans.json', 'nnUNetPlans.json']:  # handles both naming conventions
    for p in Path('/kaggle/input').rglob(fname):
        plans_path = p; break
    if plans_path: break

print(f'Checkpoint: {ckpt_path}')
print(f'Plans:      {plans_path}')

model = None

def safe_torch_load(path):
    # PyTorch 2.6+ blocks numpy in checkpoints with weights_only=True
    # Try weights_only=True first, then add numpy safe globals, then fall back
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
    # Final fallback - weights_only=False (checkpoint is our own trusted upload)
    return torch.load(path, map_location='cpu', weights_only=False)

# -- Step 2: Build PlainConvUNet directly from plans.json (bypasses get_network_from_plans API changes) --
if ckpt_path and plans_path:
    try:
        import json as _j, torch.nn as nn

        plans = _j.load(open(plans_path))
        cfg   = plans['configurations']['3d_fullres']

        # Read exact architecture from plans
        arch_class = cfg.get('UNet_class_name', 'PlainConvUNet')
        n_stages   = len(cfg['conv_kernel_sizes'])
        base_f     = cfg.get('UNet_base_num_features', 32)
        max_f      = cfg.get('unet_max_num_features', 320)
        features   = [min(base_f * (2**i), max_f) for i in range(n_stages)]
        strides    = cfg['pool_op_kernel_sizes']          # already includes [1,1,1] as first
        kernels    = cfg['conv_kernel_sizes']
        n_enc      = cfg.get('n_conv_per_stage_encoder', [2]*n_stages)
        n_dec      = cfg.get('n_conv_per_stage_decoder', [2]*(n_stages-1))
        print(f'  {arch_class} | {n_stages} stages | features: {features}')
        print(f'  strides: {strides}')

        # Import PlainConvUNet - try multiple paths (changed across nnunetv2 versions)
        PlainConvUNet = None
        for imp in [
            ('dynamic_network_architectures.architectures.unet', 'PlainConvUNet'),
            ('nnunetv2.architectures.neural_network',            'PlainConvUNet'),
            ('nnunetv2.nets.UNet',                               'PlainConvUNet'),
        ]:
            try:
                mod = __import__(imp[0], fromlist=[imp[1]])
                PlainConvUNet = getattr(mod, imp[1])
                print(f'  Imported from {imp[0]}')
                break
            except Exception:
                continue

        if PlainConvUNet is None:
            raise ImportError('Could not import PlainConvUNet from any known path')

        model = PlainConvUNet(
            input_channels          = 4,         # T1N, T1C, T2W, T2F
            n_stages                = n_stages,
            features_per_stage      = features,
            conv_op                 = nn.Conv3d,
            kernel_sizes            = kernels,
            strides                 = strides,
            n_conv_per_stage        = n_enc,
            num_classes             = 3,          # WT, TC, ET
            n_conv_per_stage_decoder= n_dec,
            conv_bias               = False,
            norm_op                 = nn.InstanceNorm3d,
            norm_op_kwargs          = {'eps': 1e-05, 'affine': True},
            dropout_op              = None,
            dropout_op_kwargs       = None,
            nonlin                  = nn.LeakyReLU,
            nonlin_kwargs           = {'inplace': True},
            deep_supervision        = False,
        )

        # Load pretrained weights (safe_torch_load handles PyTorch 2.6 weights_only change)
        ckpt   = safe_torch_load(ckpt_path)
        state  = ckpt.get('network_weights', ckpt.get('state_dict', ckpt))
        own    = model.state_dict()
        compat = {k: v for k,v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict({**own, **compat}, strict=False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Pretrained: {len(compat)}/{len(own)} layers loaded | {n_params/1e6:.1f}M params')
        if len(compat) < len(own)//2:
            print('  WARNING: <50% layers matched - weight keys may differ')

    except Exception as e:
        print(f'  PlainConvUNet build failed: {e}')
        print('  Falling back to MONAI DynUNet ...')
        model = None

# -- Step 3: Fallback -- MONAI DynUNet (training from scratch) --
if model is None:
    from monai.networks.nets import DynUNet
    kernels  = [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
    strides  = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
    model = DynUNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        kernel_size=kernels, strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name='INSTANCE', deep_supervision=False, res_block=True,
    )
    print(f'  DynUNet: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params (training from scratch)')

model = model.to(device)
print(f'Model on {device}')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 08 — Fine-tuning
# All SegResNet fixes applied:
#   - iter/next catches DataLoader worker EOFError
#   - SmartCacheDataset with try/except fallback
#   - Auto-resume from checkpoint
#   - 3-channel Dice metric
#   - Lower LR for fine-tuning (1e-4 with warmup)
# ═══════════════════════════════════════════════════════════════
C08 = """
from torch.cuda.amp import GradScaler, autocast

CKPT_DIR    = OUTPUT_ROOT / 'checkpoints'; CKPT_DIR.mkdir(exist_ok=True)
BEST_PATH   = CKPT_DIR / 'nnunet_best.pth'
LATEST_PATH = CKPT_DIR / 'nnunet_latest.pth'

def get_lr(ep, total, base=1e-4):
    warm = 5
    if ep < warm: return base * (ep+1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (ep-warm) / max(total-warm, 1)))

def safe_loader_iter(loader):
    # MUST use iter/next  - 'for batch in loader' re-raises worker exceptions
    # BEFORE entering the loop body, bypassing any inner try/except
    it = iter(loader)
    SKIP = ['EOFError','gzip','Compressed file','end-of-stream','worker','corrupt']
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as e:
            if any(k in str(e) for k in SKIP):
                continue   # skip bad file, keep training
            raise

def train_model(model, lr=1e-4, epochs=30, patience=10, val_interval=4):
    start_ep, best_dice, mlog = 0, 0.0, {'dice':[],'per_region':[],'loss':[]}
    if LATEST_PATH.exists():
        lc = torch.load(LATEST_PATH, map_location='cpu')
        model.load_state_dict(lc['model'])
        start_ep  = lc.get('epoch',0) + 1
        best_dice = lc.get('best_dice', 0)
        mlog      = lc.get('metrics', mlog)
        print(f'Resumed from epoch {start_ep-1}, best_dice={best_dice:.4f}')
        if start_ep >= epochs:
            return model, best_dice, mlog

    loss_fn     = DiceLoss(to_onehot_y=False, sigmoid=True, smooth_nr=0, smooth_dr=1e-5)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler      = GradScaler()
    dice_metric = DiceMetric(include_background=True, reduction='mean_batch')
    no_improve  = 0
    t0          = time.time()

    # Dataset strategy:
    # Train: CacheDataset 5% cache (~66 scans in RAM) — stable API across all MONAI versions
    # Val:   Plain Dataset ONLY — no disk cache (PersistentDataset = ~60GB -> kills Kaggle disk)
    try:
        from monai.data import CacheDataset
        train_ds = CacheDataset(train_dicts, train_transforms, cache_rate=0.05, num_workers=4)
        print('Using CacheDataset (5% RAM cache)')
    except Exception:
        train_ds = Dataset(train_dicts, train_transforms)
        print('Using plain Dataset (CacheDataset unavailable)')
    val_ds = Dataset(val_dicts, val_transforms)  # plain disk read - no storage bloat

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f'Train: {len(train_loader)} batches | Val: {len(val_loader)} batches')
    print(f'Expected ~20 min/epoch x {epochs} epochs ~ {epochs*20/60:.1f}h total')

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
        mlog['loss'].append(avg_loss)

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
            pr = [round(dv[i].item(),4) for i in range(3)]  # WT, TC, ET
            mlog['dice'].append(md); mlog['per_region'].append(pr)
            tag = ' NEW BEST' if md > best_dice else ''
            print(f'Ep {ep:3d} | L={avg_loss:.4f} | Dice={md:.4f} WT={pr[0]:.3f} TC={pr[1]:.3f} ET={pr[2]:.3f} | {(time.time()-t0)/60:.1f}m{tag}{bad_str}')
            if md > best_dice:
                best_dice = md; no_improve = 0
                torch.save({'model': model.state_dict(), 'epoch': ep, 'best_dice': best_dice}, BEST_PATH)
            else:
                no_improve += val_interval
        else:
            print(f'Ep {ep:3d} | L={avg_loss:.4f} | LR={cur_lr:.2e} | {(time.time()-t0)/60:.1f}m{bad_str}')

        torch.save({'model': model.state_dict(), 'epoch': ep,
                    'best_dice': best_dice, 'metrics': mlog}, LATEST_PATH)
        if no_improve >= patience:
            print(f'Early stop at ep {ep}'); break

    print(f'Done. Best Mean Dice = {best_dice:.4f}')
    print(f'  Target: WT=0.900 TC=0.867 ET=0.851 (BraTS 2021 benchmark)')
    return model, best_dice, mlog

print('Fine-tuning nnUNet on BraTS 2024 Post-Treatment...')
# 30 epochs x ~20min = ~10h -- leaves ~2h for embedding extraction + visualization
model, best_dice, metrics = train_model(model, lr=1e-4, epochs=30, patience=10)
"""

# ═══════════════════════════════════════════════════════════════
# CELL 09 — 3D Inference Visualization (GT vs Pred on T1c)
# Adapted from Cyprus Phase2_C1 notebook
# ═══════════════════════════════════════════════════════════════
C09_VIZ = """
import matplotlib
matplotlib.use('Agg')   # headless Kaggle
import matplotlib.pyplot as plt

fig_dir = OUTPUT_ROOT / 'figures'; fig_dir.mkdir(exist_ok=True)

def visualize_predictions(model, n_samples=3):
    model.eval()
    # Use first n_samples from val_dicts (already validated - no corrupt files)
    vis_dicts = val_dicts[:n_samples]
    if not vis_dicts:
        print('  No val scans for visualization'); return

    vis_ds     = Dataset(vis_dicts, val_transforms)
    vis_loader = DataLoader(vis_ds, batch_size=1, shuffle=False, num_workers=2)

    for i, batch in enumerate(vis_loader):
        try:
            vi = batch['image'].to(device)
            vl = batch['label'].to(device)
            with torch.no_grad():
                vo = sliding_window_inference(vi, PATCH, 4, model, overlap=0.5)
            pred = (torch.sigmoid(vo) > 0.5).float().cpu().numpy()[0]  # (3,H,W,D)
            gt   = vl.cpu().numpy()[0]                                  # (3,H,W,D)
            t1c  = vi.cpu().numpy()[0, 1]                               # T1c channel

            mid  = t1c.shape[2] // 2   # axial mid slice
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for c, rn in enumerate(REGIONS):
                axes[0, c].imshow(t1c[:, :, mid], cmap='gray')
                axes[0, c].imshow(gt[c, :, :, mid],   alpha=0.4, cmap='Reds')
                axes[0, c].set_title(f'GT {rn}',   fontsize=12); axes[0, c].axis('off')
                axes[1, c].imshow(t1c[:, :, mid], cmap='gray')
                axes[1, c].imshow(pred[c, :, :, mid], alpha=0.4, cmap='Blues')
                axes[1, c].set_title(f'Pred {rn}', fontsize=12); axes[1, c].axis('off')

            pid = batch.get('patient_id', ['?'])[0]
            fig.suptitle(f'nnUNet | {pid} | Dice={best_dice:.3f}', fontsize=13)
            plt.tight_layout()
            out_png = fig_dir / f'nnunet_val_sample{i}.png'
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {out_png.name}')
        except Exception as e:
            print(f'  Visualization error sample {i}: {e}')

print('Generating 3D inference visualizations...')
visualize_predictions(model, n_samples=3)
print(f'Figures saved to: {fig_dir}')
"""

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
    return torch.cat([(feat * F.interpolate(lbl[:,c:c+1],(H,W,D),mode='nearest')).sum([0,2,3,4]) /
                      (F.interpolate(lbl[:,c:c+1],(H,W,D),mode='nearest').sum() + 1e-6)
                      for c in range(3)])  # 3 channels: WT, TC, ET

def extract_embeddings(model):
    model.eval()
    _f = {}
    hook = None
    # Hook the deepest encoder block for rich features
    for name, mod in model.named_modules():
        if any(k in name for k in ['encoder', 'down_blocks', 'encoders', 'down_layers']):
            if hasattr(mod, 'forward') and len(list(mod.children())) > 0:
                hook = mod.register_forward_hook(lambda m,i,o: _f.update({'enc': o.detach() if not isinstance(o, (list,tuple)) else o[-1].detach()}))

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
                feat = _f['enc']
                ic, lc = roi_crop_resize(img, lbl)
                fr = F.interpolate(feat, ROI_SIZE, mode='trilinear', align_corners=False)
                embs.append(torch.cat([octant_pool(fr), mask_pool(fr, lc)]).cpu().numpy())
                ids.append(batch['patient_id'][0])
                tps.append(batch['timepoint'][0])
            except Exception:
                continue
            if (i+1) % 100 == 0: print(f'  {i+1}/{len(ds)} extracted')

    if hook: hook.remove()
    arr = np.array(embs)
    out = emb_dir / 'cnn_nnunet_embeddings_v2.npz'
    np.savez_compressed(out, embeddings=arr, patient_ids=ids, timepoints=tps)
    print(f'Embeddings: {arr.shape} -> {out}')
    return arr

import gc
print('Cleaning VRAM before embedding extraction...')
# Free up optimizer states and cache
if 'optimizer' in globals(): del optimizer
if 'scaler' in globals(): del scaler
gc.collect()
torch.cuda.empty_cache()

print('Extracting nnUNet embeddings...')
embeddings = extract_embeddings(model)
"""

# ═══════════════════════════════════════════════════════════════
# CELL 11 — Embedding diversity check (from Cyprus C1 notebook)
# ═══════════════════════════════════════════════════════════════
C11_DIV = """
import random

npz_path = OUTPUT_ROOT / 'embeddings' / 'cnn_nnunet_embeddings_v2.npz'
if npz_path.exists():
    data   = np.load(npz_path)
    ids_arr = data['patient_ids']
    emb_arr = data['embeddings']     # (N, D)
    N, D    = emb_arr.shape
    norms   = np.linalg.norm(emb_arr, axis=1)

    # Cosine similarity on random pairs
    emb_norm = emb_arr / (norms[:, None] + 1e-8)
    n_pairs  = min(200, N*(N-1)//2)
    pairs    = random.sample([(i,j) for i in range(N) for j in range(i+1,N)], n_pairs)
    sims     = [float(np.dot(emb_norm[i], emb_norm[j])) for i,j in pairs]
    cos_mean = float(np.mean(sims))
    diverse  = 100.0 * float(np.mean(np.array(sims) < 0.95))
    status   = 'GOOD diversity' if diverse > 20 else 'LOW diversity - check embedding hook'

    print('Embedding Diversity Check')
    print(f'  Scans:    {N}')
    print(f'  Dim:      {D}')
    print(f'  Norm:     [{norms.min():.2f}, {norms.max():.2f}]  mean={norms.mean():.2f}')
    print(f'  Cos sim:  mean={cos_mean:.3f} over {n_pairs} random pairs')
    print(f'  Diverse:  {diverse:.1f}% of pairs have cos < 0.95  [{status}]')

    # Quick per-timepoint check
    tps = data['timepoints']
    for tp in sorted(set(tps)):
        idx = [i for i,t in enumerate(tps) if t == tp]
        print(f'  Timepoint {tp}: {len(idx)} scans')
else:
    print(f'Embeddings not found at {npz_path}')
    print('  Run Cell 10 (embedding extraction) first')
"""

# ═══════════════════════════════════════════════════════════════
# CELL 12 — Summary
# ═══════════════════════════════════════════════════════════════
C10 = """
import json as _j
summary = {
    'model': MODEL_NAME, 'best_dice': float(best_dice),
    'regions': REGIONS, 'label': '{0=BG,1=NCR,2=ED,3=RC,4=ET}->3ch',
    'train_scans': len(train_dicts), 'val_scans': len(val_dicts),
    'target_brats2021': {'WT': 0.9005, 'TC': 0.8673, 'ET': 0.8509},
}
(OUTPUT_ROOT / 'summary.json').write_text(_j.dumps(summary, indent=2))

print('='*55)
print(f'  nnUNet Fine-Tuning Complete')
print(f'  Best Mean Dice:  {best_dice:.4f}')
print(f'  Regions:         {REGIONS}')
print(f'  Target (BraTS2021): WT=0.900 TC=0.867 ET=0.851')
print(f'  Outputs: {OUTPUT_ROOT}')
print('='*55)
"""

C12 = C10  # reuse summary cell content

# ═══════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════════════
cells = [
    md("# Phase 2A — nnUNet v2 Fine-Tuning (BraTS 2024 Post-Treatment)\n"
       "## 3 output channels: WT / TC / ET (RC merged into TC+WT)\n\n"
       "**Pretrained**: nnUNet BraTS 2021 fold_3 "
       "(WT=0.9005, TC=0.8673, ET=0.8509)\n\n"
       "Fine-tuning on BraTS 2024 Post-Treatment with identical label protocol.\n"
       "RC (label 3) is treated as part of TC and WT to maintain checkpoint compatibility.", "c00"),
    code(C01, "c01"),
    code(C02, "c02"),
    code(C03, "c03"),
    code(C04, "c04"),
    code(C05, "c05"),
    code(C06, "c06"),
    code(C07, "c07"),
    code(C08, "c08"),
    code(C09_VIZ, "c09"),   # 3D visualization
    code(C09,     "c10"),   # embeddings
    code(C11_DIV, "c11"),   # diversity check
    code(C12,     "c12"),   # summary
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
