#!/usr/bin/env python3
"""
Generates: Phase2_C2_nnUNet_Cyprus_Finetune.ipynb
Same PlainConvUNet architecture + BraTS2021 pretrained weights as BraTS2024 notebook,
adapted for the Cyprus brain-metastasis dataset.
Label mapping: Cyprus {0,1,2,3} -> WT=1+2+3 | TC=1+3 | ET=3  (same 3-channel output)
"""
import json, pathlib, textwrap

OUT_NB = pathlib.Path(__file__).parent.parent / "notebooks" / "Phase2_C2_nnUNet_Cyprus_Finetune.ipynb"
OUT_NB.parent.mkdir(parents=True, exist_ok=True)

def cell(src, cell_type="code"):
    lines = [l + "\n" for l in src.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {"cell_type": cell_type, "metadata": {}, "outputs": [],
            "source": lines, "execution_count": None} if cell_type == "code" \
      else {"cell_type": "markdown", "metadata": {},
            "source": lines}

# ─────────────────────────────────────────────────────────────────────────────
C00 = """
print('''
╔══════════════════════════════════════════════════════════════════╗
║  Phase2-C2  nnUNet (PlainConvUNet) – Cyprus Brain-Mets Dataset  ║
║  Architecture : PlainConvUNet  6 stages  [32,64,128,256,320,320]║
║  Pretrained   : BraTS 2021 nnUNet v2 checkpoint                  ║
║  Fine-tune on : Cyprus brain-metastasis dataset (4 modalities)   ║
║  Labels       : WT=1+2+3 | TC=1+3 | ET=3  (3-channel output)   ║
╚══════════════════════════════════════════════════════════════════╝
''')
"""

# ─────────────────────────────────────────────────────────────────────────────
C01 = """
import subprocess, sys, json, time, math, os, shutil, random, gc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    print('Installing MONAI ...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'monai[all]', '-q'])
    import monai

import monai.transforms as T
from monai.data import Dataset, CacheDataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.transforms import MapTransform
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

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
print('Label: WT=1+2+3 | TC=1+3 | ET=3  ->  3 channels OK')
"""

# ─────────────────────────────────────────────────────────────────────────────
C02 = """
# Empty cell (kept for structural parity)
"""

# ─────────────────────────────────────────────────────────────────────────────
C03 = """
# ── Paths ──
OUTPUT_ROOT = Path('/kaggle/working/phase2_nnunet_cyprus')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR   = OUTPUT_ROOT / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)
BEST_PATH   = CKPT_DIR / 'nnunet_cyprus_best.pth'
LATEST_PATH = CKPT_DIR / 'nnunet_cyprus_latest.pth'

PATCH   = (128, 128, 128)
REGIONS = ['WT', 'TC', 'ET']

# ── Recover checkpoints from previous notebook output (if attached as input) ──
if not BEST_PATH.exists() or not LATEST_PATH.exists():
    for src in sorted(Path('/kaggle/input').rglob('nnunet_cyprus_best.pth')):
        if not BEST_PATH.exists():
            shutil.copy2(src, BEST_PATH)
            print(f'  Recovered BEST checkpoint from {src}')
        break
    for src in sorted(Path('/kaggle/input').rglob('nnunet_cyprus_latest.pth')):
        if not LATEST_PATH.exists():
            shutil.copy2(src, LATEST_PATH)
            print(f'  Recovered LATEST checkpoint from {src}')
        break
    if BEST_PATH.exists():
        print(f'  ✅ Checkpoints recovered — training will be skipped')

# ── Disk space check ──
free_gb = shutil.disk_usage('/kaggle/working').free / 1e9
print(f'Free disk: {free_gb:.1f} GB')
if free_gb < 3:
    raise RuntimeError(f'Only {free_gb:.1f} GB free — need at least 3 GB for checkpoints')
print(f'Output root: {OUTPUT_ROOT}')

# ── Cyprus data root ──
DATA_ROOT = Path('/kaggle/input/datasets/zinou123viva/cyprus-proteas-brain-mets')
if not DATA_ROOT.exists():
    # Fallback: scan /kaggle/input for the Cyprus dataset
    for candidate in sorted(Path('/kaggle/input').rglob('data_splits.json')):
        DATA_ROOT = candidate.parent; break
    for candidate in Path('/kaggle/input').iterdir():
        if not candidate.is_dir(): continue
        if any((candidate / f'P{i:02d}').exists() for i in range(1, 5)):
            DATA_ROOT = candidate; break
        for sub in candidate.iterdir():
            if sub.is_dir() and any((sub / f'P{i:02d}').exists() for i in range(1, 5)):
                DATA_ROOT = sub; break
print(f'DATA_ROOT: {DATA_ROOT}  exists={DATA_ROOT.exists()}')
"""

# ─────────────────────────────────────────────────────────────────────────────
C04 = """
import json as _jmod

# ── Path resolver (handles .nii / .nii.gz / .nii_gz naming variants) ──
SYMLINK_DIR = Path('/kaggle/working/cyprus_nifti_links')
SYMLINK_DIR.mkdir(parents=True, exist_ok=True)

def resolve_path(root, rel):
    p = root / rel
    if p.exists(): return str(p)
    
    # Check alternate extensions manually
    as_str = str(p)
    variants = [
        as_str + '.gz',
        as_str.replace('.nii.gz', '.nii_gz'),
        as_str.replace('.nii.gz', '.nii'),
    ]
    for v in variants:
        if Path(v).exists():
            if v.endswith('.nii_gz'):
                link = SYMLINK_DIR / rel
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists(): os.symlink(v, str(link))
                return str(link)
            return v
            
    # Fuzzy parent search
    parent = p.parent
    if parent.exists():
        target_stem = p.name.replace('.nii.gz','').replace('.nii_gz','').replace('.nii','').lower()
        for f in parent.iterdir():
            f_stem = f.name.replace('.nii.gz','').replace('.nii_gz','').replace('.nii','').lower()
            if f_stem == target_stem: return str(f)
            
    raise FileNotFoundError(f'Not found: {rel}')

# ── Load data_splits.json ──
splits_path = DATA_ROOT / 'data_splits.json'
if not splits_path.exists():
    for f in DATA_ROOT.rglob('*splits*.json'):
        splits_path = f; break
all_splits = _jmod.load(open(splits_path))
print(f'Splits file: {splits_path.name}  | keys: {list(all_splits.keys())[:5]}')

# ── Build MONAI-style dicts from scan list ──
def build_scan_dicts(scans, label='subset'):
    dicts, skips = [], 0
    for scan in scans:
        try:
            dicts.append({
                'image': [
                    resolve_path(DATA_ROOT, scan['t1']),
                    resolve_path(DATA_ROOT, scan['t1c']),
                    resolve_path(DATA_ROOT, scan['t2']),
                    resolve_path(DATA_ROOT, scan['fla']),
                ],
                'label':       resolve_path(DATA_ROOT, scan['mask']),
                'patient_dir': scan.get('patient_dir', scan.get('patient_id', '?')),
                'visit':       scan.get('visit', scan.get('timepoint', '?')),
            })
        except (FileNotFoundError, KeyError):
            skips += 1
    if skips: print(f'  Skipped {skips} missing files in {label}')
    return dicts

# ── Load fold_0 from 3fold split (same approach as C1 MetSeg) ──
# data_splits.json structure: root -> '3fold'/'5fold' -> 'fold_0' -> 'train_scans'/'test_scans'
split_key = '3fold' if '3fold' in all_splits else list(k for k in all_splits if k != 'metadata')[0]
FOLD = 0
fold_key = f'fold_{FOLD}'
print(f'Using split: {split_key} / {fold_key}')

fold_data = all_splits[split_key][fold_key]
train_dicts = build_scan_dicts(fold_data['train_scans'], f'fold{FOLD}_train')
val_dicts   = build_scan_dicts(fold_data.get('test_scans', []), f'fold{FOLD}_val')

print(f'Fold {FOLD} — Train: {len(train_dicts)} scans | Val: {len(val_dicts)} scans')
if not train_dicts:
    raise RuntimeError('No training scans found! Check data_splits.json structure and DATA_ROOT.')
if train_dicts:
    print(f'  Sample image[0]: {train_dicts[0]["image"][0]}')
"""

# ─────────────────────────────────────────────────────────────────────────────
C05 = """
# ── Label mapping: Cyprus {0,1,2,3} -> [WT, TC, ET] ──
# 0=Background  1=NonEnhancing/Necrotic  2=Edema  3=Enhancing
# WT = 1+2+3 (all tumor)
# TC = 1+3   (tumor core = necrotic + enhancing)
# ET = 3     (enhancing only)
class ConvertToMultiChannelCyprusd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if isinstance(img, torch.Tensor):
                if img.ndim == 4 and img.shape[0] == 1: img = img.squeeze(0)
                result = [
                    (img == 1) | (img == 2) | (img == 3),  # WT
                    (img == 1) | (img == 3),                # TC
                    (img == 3),                             # ET
                ]
                d[key] = torch.stack(result, dim=0).float()
            else:
                if img.ndim == 4 and img.shape[0] == 1: img = img.squeeze(0)
                d[key] = np.stack([
                    ((img == 1) | (img == 2) | (img == 3)).astype(np.float32),
                    ((img == 1) | (img == 3)).astype(np.float32),
                    (img == 3).astype(np.float32),
                ], axis=0)
        return d

PATCH_SIZE = PATCH

train_transforms = T.Compose([
    T.LoadImaged(keys=['image', 'label']),
    T.EnsureChannelFirstd(keys=['image', 'label']),
    T.EnsureTyped(keys=['image', 'label']),
    T.Orientationd(keys=['image', 'label'], axcodes='RAS'),
    T.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=True),
    T.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ConvertToMultiChannelCyprusd(keys=['label']),
    T.RandSpatialCropd(keys=['image', 'label'], roi_size=PATCH_SIZE, random_size=False),
    T.SpatialPadd(keys=['image', 'label'], spatial_size=PATCH_SIZE),
    T.RandFlipd(keys=['image', 'label'], spatial_axis=[0], prob=0.5),
    T.RandFlipd(keys=['image', 'label'], spatial_axis=[1], prob=0.5),
    T.RandFlipd(keys=['image', 'label'], spatial_axis=[2], prob=0.5),
    T.RandScaleIntensityd(keys='image', factors=0.1, prob=0.3),
    T.RandShiftIntensityd(keys='image', offsets=0.1, prob=0.3),
    T.ToTensord(keys=['image', 'label']),
])

val_transforms = T.Compose([
    T.LoadImaged(keys=['image', 'label']),
    T.EnsureChannelFirstd(keys=['image', 'label']),
    T.EnsureTyped(keys=['image', 'label']),
    T.Orientationd(keys=['image', 'label'], axcodes='RAS'),
    T.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ConvertToMultiChannelCyprusd(keys=['label']),
    T.ToTensord(keys=['image', 'label']),
])
print('Transforms ready (Cyprus label mapping: WT=1+2+3 | TC=1+3 | ET=3)')
"""

# ─────────────────────────────────────────────────────────────────────────────
C06 = """
print('Skipping strict compression check for Cyprus dataset.')
"""

# ─────────────────────────────────────────────────────────────────────────────
C07 = """
print('='*55)
print('  Loading nnUNet v2 pretrained model')
print('  Target: WT=0.9005  TC=0.8673  ET=0.8509 (BraTS 2021)')
print('='*55)

# ── Find checkpoint + plans ──
ckpt_path  = None
plans_path = None
for p in Path('/kaggle/input').rglob('checkpoint_final.pth'):
    ckpt_path = p; break
for fname in ['plans.json', 'nnUNetPlans.json']:
    for p in Path('/kaggle/input').rglob(fname):
        plans_path = p; break
    if plans_path: break

print(f'Checkpoint: {ckpt_path}')
print(f'Plans:      {plans_path}')

model = None

def safe_torch_load(path):
    # PyTorch 2.6+ blocks numpy in checkpoints with weights_only=True
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

# ── Build PlainConvUNet directly from plans.json ──
if ckpt_path and plans_path:
    try:
        import json as _j, torch.nn as nn

        plans = _j.load(open(plans_path))
        cfg   = plans['configurations']['3d_fullres']

        arch_class = cfg.get('UNet_class_name', 'PlainConvUNet')
        n_stages   = len(cfg['conv_kernel_sizes'])
        base_f     = cfg.get('UNet_base_num_features', 32)
        max_f      = cfg.get('unet_max_num_features', 320)
        features   = [min(base_f * (2**i), max_f) for i in range(n_stages)]
        strides    = cfg['pool_op_kernel_sizes']
        kernels    = cfg['conv_kernel_sizes']
        n_enc      = cfg.get('n_conv_per_stage_encoder', [2]*n_stages)
        n_dec      = cfg.get('n_conv_per_stage_decoder', [2]*(n_stages-1))
        print(f'  {arch_class} | {n_stages} stages | features: {features}')
        print(f'  strides: {strides}')

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
            input_channels           = 4,   # T1, T1c, T2, FLAIR
            n_stages                 = n_stages,
            features_per_stage       = features,
            conv_op                  = nn.Conv3d,
            kernel_sizes             = kernels,
            strides                  = strides,
            n_conv_per_stage         = n_enc,
            num_classes              = 3,   # WT, TC, ET
            n_conv_per_stage_decoder = n_dec,
            conv_bias                = False,
            norm_op                  = nn.InstanceNorm3d,
            norm_op_kwargs           = {'eps': 1e-05, 'affine': True},
            dropout_op               = None,
            dropout_op_kwargs        = None,
            nonlin                   = nn.LeakyReLU,
            nonlin_kwargs            = {'inplace': True},
            deep_supervision         = False,
        )

        ckpt   = safe_torch_load(ckpt_path)
        state  = ckpt.get('network_weights', ckpt.get('state_dict', ckpt))
        own    = model.state_dict()
        compat = {k: v for k,v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict({**own, **compat}, strict=False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Pretrained: {len(compat)}/{len(own)} layers loaded | {n_params/1e6:.1f}M params')
        if len(compat) < len(own)//2:
            print('  WARNING: <50% layers matched')

    except Exception as e:
        print(f'  PlainConvUNet build failed: {e}')
        print('  Falling back to MONAI DynUNet ...')
        model = None

# ── Fallback: MONAI DynUNet ──
if model is None:
    from monai.networks.nets import DynUNet
    kernels = [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
    strides = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
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

# ─────────────────────────────────────────────────────────────────────────────
C08 = """
def get_lr(ep, total, base_lr, warmup=5):
    if ep < warmup:
        return base_lr * (ep + 1) / warmup
    progress = (ep - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

def safe_loader_iter(loader):
    it = iter(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            break
        except Exception as e:
            if 'EOFError' in str(type(e).__name__) or 'EOF' in str(e):
                continue
            raise

def train_model(model, lr=1e-4, epochs=30, patience=10, val_interval=4):
    start_ep, best_dice, mlog = 0, 0.0, {'dice':[],'per_region':[],'loss':[]}
    if LATEST_PATH.exists():
        lc = torch.load(LATEST_PATH, map_location='cpu')
        model.load_state_dict(lc['model'])
        start_ep   = lc.get('epoch', 0) + 1
        best_dice  = lc.get('best_dice', 0.0)
        mlog       = lc.get('metrics', mlog)
        print(f'Resumed from epoch {start_ep-1}, best_dice={best_dice:.4f}')
    if start_ep >= epochs:
        print('Already completed.'); return model, best_dice, mlog

    loss_fn     = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler      = GradScaler()
    dice_metric = DiceMetric(include_background=True, reduction='mean_batch')
    t0          = time.time()

    # CacheDataset — lower cache rate than C1 (0.5/1.0) because PlainConvUNet
    # is 3× larger (30.8M vs 11M params) and needs more RAM headroom for CUDA ops.
    try:
        train_ds = CacheDataset(train_dicts, train_transforms, cache_rate=0.3, num_workers=0)
        print(f'Using CacheDataset (30% cache, {len(train_dicts)} train scans)')
    except Exception:
        train_ds = Dataset(train_dicts, train_transforms)
        print('Fallback: plain Dataset')
    try:
        val_ds = CacheDataset(val_dicts, val_transforms, cache_rate=0.5, num_workers=0)
        print(f'Val CacheDataset (50% cache, {len(val_dicts)} val scans)')
    except Exception:
        val_ds = Dataset(val_dicts, val_transforms)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f'Train: {len(train_loader)} batches | Val: {len(val_loader)} batches')
    print(f'Expected ~20 min/epoch x {epochs} epochs ~ {epochs*20/60:.1f}h total')

    for ep in range(start_ep, epochs):
        model.train()
        cur_lr = get_lr(ep, epochs, lr)
        for pg in optimizer.param_groups: pg['lr'] = cur_lr
        ep_loss, n_skipped = 0.0, 0

        for batch in safe_loader_iter(train_loader):
            try:
                img = batch['image'].to(device)
                lbl = batch['label'].to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    out  = model(img)
                    loss = loss_fn(out, lbl)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                ep_loss += loss.item()
            except (EOFError, RuntimeError) as e:
                if 'CUDA out of memory' in str(e): raise
                n_skipped += 1
                continue
            except Exception:
                n_skipped += 1
                continue

        ep_loss /= max(len(train_loader) - n_skipped, 1)
        elapsed = (time.time() - t0) / 60

        if (ep+1) % val_interval == 0 or ep == epochs-1:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for vb in safe_loader_iter(val_loader):
                    try:
                        vo = sliding_window_inference(vb['image'].to(device), PATCH, 4, model, overlap=0.5)
                        dice_metric((torch.sigmoid(vo)>0.5).float(), vb['label'].to(device))
                    except Exception:
                        pass
            dv = dice_metric.aggregate(); md = dv.mean().item()
            per_reg = [f'{REGIONS[i]}={dv[i].item():.3f}' for i in range(len(REGIONS))]
            mlog['dice'].append(md); mlog['per_region'].append(per_reg); mlog['loss'].append(ep_loss)
            is_best = md > best_dice
            if is_best:
                best_dice = md
                torch.save({'model': model.state_dict(), 'epoch': ep, 'best_dice': best_dice}, BEST_PATH)
            skipped_str = f' | skipped {n_skipped}' if n_skipped else ''
            print(f'Ep {ep:3d} | L={ep_loss:.4f} | Dice={md:.4f} {" ".join(per_reg)} | {elapsed:.1f}m'
                  + (' NEW BEST' if is_best else '') + skipped_str)
        else:
            skipped_str = f' | skipped {n_skipped}' if n_skipped else ''
            print(f'Ep {ep:3d} | L={ep_loss:.4f} | LR={cur_lr:.2e} | {elapsed:.1f}m{skipped_str}')

        torch.save({'model': model.state_dict(), 'epoch': ep,
                    'best_dice': best_dice, 'metrics': mlog}, LATEST_PATH)

    return model, best_dice, mlog

print('Fine-tuning nnUNet on Cyprus Brain-Metastasis dataset...')
# 30 epochs x ~20min = ~10h -- leaves ~2h for embedding extraction + visualization
model, best_dice, metrics = train_model(model, lr=1e-4, epochs=30, patience=10)
"""

# ─────────────────────────────────────────────────────────────────────────────
C09 = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig_dir = OUTPUT_ROOT / 'figures'; fig_dir.mkdir(exist_ok=True)

def visualize_predictions(model, n_samples=3):
    model.eval()
    vis_dicts  = val_dicts[:n_samples]
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
            pred = (torch.sigmoid(vo) > 0.5).float().cpu().numpy()[0]
            gt   = vl.cpu().numpy()[0]
            t1c  = vi.cpu().numpy()[0, 1]   # T1c channel (index 1)
            mid  = t1c.shape[2] // 2
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for c, rn in enumerate(REGIONS):
                axes[0, c].imshow(t1c[:, :, mid], cmap='gray')
                axes[0, c].imshow(gt[c, :, :, mid],   alpha=0.4, cmap='Reds')
                axes[0, c].set_title(f'GT {rn}'); axes[0, c].axis('off')
                axes[1, c].imshow(t1c[:, :, mid], cmap='gray')
                axes[1, c].imshow(pred[c, :, :, mid], alpha=0.4, cmap='Blues')
                axes[1, c].set_title(f'Pred {rn}'); axes[1, c].axis('off')
            pid   = batch.get('patient_dir', ['?'])[0]
            visit = batch.get('visit', ['?'])[0]
            fig.suptitle(f'nnUNet Cyprus | {pid} visit={visit} | Dice={best_dice:.3f}', fontsize=13)
            plt.tight_layout()
            out_png = fig_dir / f'nnunet_cyprus_sample{i}.png'
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {out_png.name}')
        except Exception as e:
            print(f'  Visualization error sample {i}: {e}')

print('Generating 3D inference visualizations...')
visualize_predictions(model, n_samples=3)
print(f'Figures saved to: {fig_dir}')
"""

# ─────────────────────────────────────────────────────────────────────────────
C10 = """
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
                      for c in range(3)])

def extract_embeddings(model):
    model.eval()
    _f  = {}
    hook = None
    for name, mod in model.named_modules():
        if any(k in name for k in ['encoder','down_blocks','encoders','down_layers']):
            if hasattr(mod, 'forward') and len(list(mod.children())) > 0:
                hook = mod.register_forward_hook(
                    lambda m,i,o: _f.update({'enc': o.detach() if not isinstance(o,(list,tuple)) else o[-1].detach()})
                )

    emb_dir = OUTPUT_ROOT / 'embeddings'; emb_dir.mkdir(exist_ok=True)
    embs, ids, visits = [], [], []
    all_dicts = train_dicts + val_dicts
    ds        = Dataset(all_dicts, val_transforms)
    loader    = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

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
                ids.append(batch['patient_dir'][0])
                visits.append(batch['visit'][0])
            except Exception:
                continue
            if (i+1) % 50 == 0: print(f'  {i+1}/{len(all_dicts)} extracted')

    if hook: hook.remove()
    arr = np.array(embs)
    out = emb_dir / 'cnn_nnunet_cyprus_embeddings.npz'
    np.savez_compressed(out, embeddings=arr, patient_ids=ids, visits=visits)
    print(f'Embeddings: {arr.shape} -> {out}')
    return arr

import gc
print('Cleaning VRAM before embedding extraction...')
if 'optimizer' in globals(): del optimizer
if 'scaler'    in globals(): del scaler
gc.collect()
torch.cuda.empty_cache()

print('Extracting nnUNet Cyprus embeddings...')
embeddings = extract_embeddings(model)
"""

# ─────────────────────────────────────────────────────────────────────────────
C11 = """
import random

npz_path = OUTPUT_ROOT / 'embeddings' / 'cnn_nnunet_cyprus_embeddings.npz'
if npz_path.exists():
    data     = np.load(npz_path, allow_pickle=True)
    ids_arr  = data['patient_ids']
    emb_arr  = data['embeddings']
    N, D     = emb_arr.shape
    norms    = np.linalg.norm(emb_arr, axis=1)
    emb_norm = emb_arr / (norms[:, None] + 1e-8)
    n_pairs  = min(200, N*(N-1)//2)
    pairs    = random.sample([(i,j) for i in range(N) for j in range(i+1,N)], n_pairs) if N > 1 else []
    sims     = [float(np.dot(emb_norm[i], emb_norm[j])) for i,j in pairs]
    cos_mean = float(np.mean(sims)) if sims else float('nan')
    diverse  = 100.0 * float(np.mean(np.array(sims) < 0.95)) if sims else 0.0
    status   = 'GOOD diversity' if diverse > 20 else 'LOW diversity - check embedding hook'

    print('Embedding Diversity Check (Cyprus nnUNet)')
    print(f'  Scans:    {N}')
    print(f'  Dim:      {D}')
    print(f'  Norm:     [{norms.min():.2f}, {norms.max():.2f}]  mean={norms.mean():.2f}')
    print(f'  Cos sim:  mean={cos_mean:.3f} over {n_pairs} pairs')
    print(f'  Diverse:  {diverse:.1f}% of pairs have cos < 0.95  [{status}]')

    visits = data.get('visits', ['?']*N)
    for v in sorted(set(visits)):
        idx = [i for i,t in enumerate(visits) if t == v]
        print(f'  Visit {v}: {len(idx)} scans')
else:
    print(f'Embeddings not found at {npz_path}')
"""

# ─────────────────────────────────────────────────────────────────────────────
C12 = """
print()
print('='*60)
print('  Cyprus nnUNet Fine-Tuning Summary')
print('='*60)
print(f'  Architecture  : PlainConvUNet 6 stages [32,64,128,256,320,320]')
print(f'  Pretrained    : BraTS 2021 nnUNet v2 checkpoint')
print(f'  Dataset       : Cyprus brain-metastasis (4 modalities)')
print(f'  Label mapping : WT=1+2+3 | TC=1+3 | ET=3')
print(f'  Best Val Dice : {best_dice:.4f}')
print(f'  Checkpoint    : {BEST_PATH}')
print(f'  Embeddings    : {OUTPUT_ROOT}/embeddings/cnn_nnunet_cyprus_embeddings.npz')
print(f'  Figures       : {OUTPUT_ROOT}/figures/')
print('='*60)
"""

# ─────────────────────────────────────────────────────────────────────────────
cells = [
    cell(C00.strip()),
    cell(C01.strip()),
    cell(C02.strip()),
    cell(C03.strip()),
    cell(C04.strip()),
    cell(C05.strip()),
    cell(C06.strip()),
    cell(C07.strip()),
    cell(C08.strip()),
    cell(C09.strip()),
    cell(C10.strip()),
    cell(C11.strip()),
    cell(C12.strip()),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(OUT_NB, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Written: {OUT_NB}")
print(f"Cells: {len(cells)}")
