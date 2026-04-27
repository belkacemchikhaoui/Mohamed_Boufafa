"""Build Phase 2 notebooks: SegResNet + DynUNet (nnU-Net arch) fine-tuning."""
import json, textwrap

def md(src):  return {"cell_type":"markdown","metadata":{},"source":src.strip().split("\n"),"id":"a"}
def code(src):return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip().split("\n"),"id":"a"}

def save_nb(cells, path, kaggle=True):
    meta = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python","version":"3.12.12"}}
    if kaggle:
        meta["kaggle"] = {"accelerator":"nvidiaTeslaT4","isGpuEnabled":True,
                          "isInternetEnabled":True,"language":"python","sourceType":"notebook"}
    nb = {"nbformat":4,"nbformat_minor":5,"metadata":meta,"cells":cells}
    for i,c in enumerate(nb["cells"]): c["id"]=f"c{i:02d}"
    with open(path,"w") as f: json.dump(nb,f,indent=1)
    print(f"  ✅ {path} ({len(cells)} cells)")

# ════════════════════════════════════════════════════════
#  SHARED CODE BLOCKS (used by both notebooks)
# ════════════════════════════════════════════════════════

IMPORTS = '''
import subprocess, sys, os, warnings, json, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path; from collections import OrderedDict; from tqdm import tqdm
warnings.filterwarnings('ignore')
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

for pkg in ['monai[all]','nibabel']:
    try: __import__(pkg.split('[')[0])
    except ImportError: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

from monai.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss
from monai.data import DataLoader, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import monai.transforms as T
from monai.transforms.compose import MapTransform
from monai.utils import ensure_tuple_rep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB)')
'''

LABEL_CONVERT = '''
# ── BraTS 2024 Post-Treatment Label Convention ──
# {0=Background, 1=NCR, 2=ED, 3=RC (Resection Cavity), 4=ET}
# ⚠️ DIFFERENT from BraTS 2021 where label 3 was unused!
# Label 3 is present in 75% of post-treatment scans (confirmed by EDA)
#
# Sub-region mapping (4 channels):
#   WT = 1+2+3+4  (Whole Tumor: all non-background, including RC)
#   TC = 1+3+4    (Tumor Core: NCR + RC + ET)
#   ET = 4        (Enhancing Tumor only)
#   RC = 3        (Resection Cavity — post-treatment specific metric)

class ConvertToMultiChannelBratsPostTreatmentd(MapTransform):
    """Convert BraTS 2024 Post-Treatment labels {0,1,2,3,4} → 4-channel [WT, TC, ET, RC].
    Label 3 = Resection Cavity (RC) — critical for post-treatment glioma.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1: img = img.squeeze(0)
            result = [
                (img==1)|(img==2)|(img==3)|(img==4),  # WT: all tumor + RC
                (img==1)|(img==3)|(img==4),             # TC: core + RC + ET
                img==4,                                  # ET: enhancing only
                img==3,                                  # RC: resection cavity (new!)
            ]
            d[key] = (torch.stack(result, 0).float() if isinstance(img, torch.Tensor)
                      else np.stack(result, 0).astype(np.float32))
        return d

# Keep backwards-compatible alias
ConvertToMultiChannelBratsGliomad = ConvertToMultiChannelBratsPostTreatmentd
REGIONS = ['WT', 'TC', 'ET', 'RC']
print('Label mapping: {0=BG, 1=NCR, 2=ED, 3=RC, 4=ET}')
print('  WT=1+2+3+4 | TC=1+3+4 | ET=4 | RC=3 (4 channels) ✅')
'''

DATA_LOADING = '''
# ── .nii_gz → .nii.gz symlink resolver ──
# Files uploaded as .nii_gz to prevent Kaggle auto-extraction of .gz
SYMLINK_DIR = Path('/kaggle/working/nifti_links')

def setup_nii_gz_symlinks(data_dir):
    """Batch-create .nii.gz symlinks for all .nii_gz files — instant, zero disk copy."""
    count = 0
    for nii_gz in Path(data_dir).rglob('*.nii_gz'):
        real_name = nii_gz.name.replace('.nii_gz', '.nii.gz')
        link = SYMLINK_DIR / nii_gz.parent.name / real_name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(str(nii_gz), str(link))
            count += 1
    return count

# ── GPU info ──
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)} | Total memory: {total_mem:.1f} GB')

# ── Data paths ──
DATA_ROOT = Path('/kaggle/input')
NIFTI_ROOT = None

# Step 1: Create symlinks for any .nii_gz uploads
for kaggle_ds in DATA_ROOT.iterdir():
    if not kaggle_ds.is_dir(): continue
    nii_gz_files = list(kaggle_ds.rglob('*.nii_gz'))
    if nii_gz_files:
        n = setup_nii_gz_symlinks(kaggle_ds)
        print(f'  Created {n} symlinks .nii_gz→.nii.gz in {kaggle_ds.name}')

# Step 2: Find patient folder root (symlinks first, then direct)
for search_root in [SYMLINK_DIR, DATA_ROOT]:
    if not search_root.exists(): continue
    for c in search_root.rglob('BraTS-GLI-*'):
        if c.is_dir():
            NIFTI_ROOT = c.parent; break
    if NIFTI_ROOT: break

if NIFTI_ROOT is None:
    raise RuntimeError('No BraTS-GLI-* folders found — check dataset attachments')
print(f'NIFTI_ROOT: {NIFTI_ROOT}')

# Step 3: Load scan_index.json ONLY for split metadata (patient train/val assignment)
# ⚠️  NEVER use stored file paths from scan_index — they are local HDD paths!
split_map = {}   # patient_id → 'train' or 'val'
scan_meta = {}   # scan_id   → {patient_id, timepoint, split}
for f in DATA_ROOT.rglob('scan_index.json'):
    with open(f) as fh:
        si = json.load(fh)
    for s in si.get('training_scans', []):
        pid   = s['patient_id']
        sid   = s['scan_id']
        split = s.get('split')
        if split: split_map[pid] = split
        scan_meta[sid] = {'patient_id': pid, 'timepoint': s.get('timepoint','100'), 'split': split}
    print(f'  Loaded split metadata: {len(split_map)} patients from scan_index.json')
    break

# Step 4: Discover ALL actual file paths fresh from NIFTI_ROOT (machine-independent)
all_dirs = sorted([d for d in NIFTI_ROOT.iterdir() if d.is_dir() and 'BraTS-GLI' in d.name])
training_scans = []
for d in all_dirs:
    files = {m: list(d.glob(f'*-{m}*')) for m in ['t1n','t1c','t2w','t2f']}
    seg   = list(d.glob('*-seg*'))
    if not (all(files[m] for m in files) and seg): continue
    name  = d.name
    parts = name.rsplit('-', 1)
    pid   = parts[0]
    tp    = parts[1] if len(parts) > 1 else '100'
    split = scan_meta.get(name, {}).get('split') or split_map.get(pid)
    training_scans.append({
        'scan_id': name, 'patient_id': pid, 'timepoint': tp,
        't1n': str(files['t1n'][0]), 't1c': str(files['t1c'][0]),
        't2w': str(files['t2w'][0]), 't2f': str(files['t2f'][0]),
        'seg': str(seg[0]),
        'split': split,
    })
scans = training_scans
print(f'Total training scans discovered: {len(scans)}')

# Step 5: Apply splits
if any(s['split'] for s in scans):
    train_scans = [s for s in scans if s['split'] == 'train']
    val_scans   = [s for s in scans if s['split'] == 'val']
    # Unsplit → assign by known patient
    from collections import defaultdict
    known_train = {s['patient_id'] for s in train_scans}
    known_val   = {s['patient_id'] for s in val_scans}
    for s in scans:
        if s['split']: continue
        if   s['patient_id'] in known_train: train_scans.append(s)
        elif s['patient_id'] in known_val:   val_scans.append(s)
        else: train_scans.append(s)
else:
    from collections import defaultdict
    patients  = defaultdict(list)
    for s in scans: patients[s['patient_id']].append(s)
    pids      = sorted(patients.keys())
    split_idx = int(0.8 * len(pids))
    train_pids = set(pids[:split_idx]); val_pids = set(pids[split_idx:])
    train_scans = [s for s in scans if s['patient_id'] in train_pids]
    val_scans   = [s for s in scans if s['patient_id'] in val_pids]

train_pats = len({s['patient_id'] for s in train_scans})
val_pats   = len({s['patient_id'] for s in val_scans})
print(f'Train: {len(train_scans)} scans ({train_pats} patients)')
print(f'Val:   {len(val_scans)} scans ({val_pats} patients)')
print(f'✅ File paths resolved live from {NIFTI_ROOT} (not from scan_index stored paths)')
'''


def make_transforms(patch_str):
    return f'''
patch = {patch_str}
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
print('Transforms ready ✅')
'''

DICT_BUILDER = '''
def build_dicts(scan_list):
    dicts = []
    for s in scan_list:
        try:
            dicts.append({
                'image': [s['t1n'], s['t1c'], s['t2w'], s['t2f']],
                'label': s['seg'],
                'patient_id': s['patient_id'], 'timepoint': s['timepoint'],
            })
        except: pass
    return dicts

train_dicts = build_dicts(train_scans)
val_dicts = build_dicts(val_scans)
print(f'Train dicts: {len(train_dicts)} | Val dicts: {len(val_dicts)}')
'''

EXTRACTION = '''
# ── v2 Embedding Extraction ──
ROI_PAD = 8; ROI_SIZE = (64,64,64)
REGIONS = ['WT','TC','ET','RC']

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
    return F.interpolate(ic,ROI_SIZE,mode='trilinear',align_corners=False), \\
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
    for ch in range(4):
        prob = F.interpolate(lbl[:,ch:ch+1],size=(H,W,D),mode='nearest')
        regions.append((feat*prob).sum(dim=[0,2,3,4]) / (prob.sum()+1e-6))
    return torch.cat(regions)
print('v2 extraction functions ready ✅')
'''

def make_extract_fn(model_name, hook_target_code):
    return f'''
def extract_embeddings(model):
    model.eval(); model.to(device)
    _feats = {{}}
    def hook_fn(m,inp,out): _feats['f'] = (out[0] if isinstance(out,(list,tuple)) else out).detach()
    {hook_target_code}
    h = target.register_forward_hook(hook_fn) if target else None

    all_dicts = build_dicts(scans)
    ds = CacheDataset(all_dicts, val_transforms, cache_rate=0.1, num_workers=0)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    embeddings, skipped = {{}}, 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Extracting')):
            img, lbl = batch['image'].to(device), batch['label'].to(device)
            key = f"{{batch['patient_id'][0]}}__{{batch['timepoint'][0]}}"
            try:
                ir, lr = roi_crop_resize(img, lbl)
                _feats.clear(); _ = model(ir)
                feat = _feats.get('f')
                if feat is None: skipped+=1; continue
                embeddings[key] = torch.cat([octant_pool(feat), mask_weighted_pool(feat,lr)]).cpu().numpy()
                if i<3: print(f'  {{key}}: {{embeddings[key].shape[0]}}-dim')
            except Exception as e: print(f'  ⚠️ {{key}}: {{e}}'); skipped+=1
    if h: h.remove()
    print(f'  Extracted: {{len(embeddings)}} | Skipped: {{skipped}}')
    emb_dir = OUTPUT_ROOT / 'embeddings'; emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez(emb_dir / 'cnn_{model_name}_embeddings_v2.npz', **embeddings)
    dim = list(embeddings.values())[0].shape[0] if embeddings else 0
    print(f'  ✅ Saved: ({{len(embeddings)}} × {{dim}}-dim)')
    return embeddings
print('Extraction function ready ✅')
'''

TRAIN_FN = '''
def get_lr(epoch, total, base_lr):
    warmup = 5
    if epoch < warmup: return 1e-6 + (base_lr-1e-6)*(epoch/warmup)
    progress = (epoch-warmup) / max(total-warmup, 1)
    return 1e-6 + (base_lr-1e-6) * 0.5 * (1 + math.cos(math.pi * progress))

def train_model(model, lr=1e-4, epochs=50, patience=15, val_interval=4):
    ckpt_dir = OUTPUT_ROOT / 'checkpoints'; ckpt_dir.mkdir(parents=True,exist_ok=True)
    best_path = ckpt_dir / f'{MODEL_NAME}_best.pth'
    latest_path = ckpt_dir / f'{MODEL_NAME}_latest.pth'

    # Check if already done
    if best_path.exists() and latest_path.exists():
        lc = torch.load(latest_path, map_location='cpu', weights_only=False)
        if lc.get('epoch',-1) >= epochs-1:
            print(f'  ✅ Already complete (Dice={lc.get("best_dice",0):.4f})')
            model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['model_state_dict'])
            return model, lc.get('best_dice',0), lc.get('metrics',{})

    train_ds = CacheDataset(train_dicts, train_transforms, cache_rate=0.3, num_workers=2)
    val_ds = CacheDataset(val_dicts, val_transforms, cache_rate=0.5, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    dice_loss = DiceLoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5)
    bce_loss = BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    dice_metric = DiceMetric(include_background=True, reduction='mean_batch')
    best_dice, pat_ctr, start_ep = 0.0, 0, 0
    metrics = {'loss':[], 'dice':[], 'per_region':[], 'lr':[]}

    # Resume
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_ep = ckpt['epoch']+1; best_dice = ckpt.get('best_dice',0)
        metrics = ckpt.get('metrics', metrics); pat_ctr = ckpt.get('pat_ctr',0)
        print(f'  🔄 Resuming epoch {start_ep} (best={best_dice:.4f})')

    t0 = time.time()
    for ep in range(start_ep, epochs):
        cur_lr = get_lr(ep, epochs, lr)
        for pg in optimizer.param_groups: pg['lr'] = cur_lr
        model.train(); ep_loss, n = 0.0, 0
        for batch in train_loader:
            imgs, lbls = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = dice_loss(out, lbls) + bce_loss(out, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer); scaler.update()
            ep_loss += loss.item(); n += 1
        avg_loss = ep_loss / max(n,1)
        metrics['loss'].append(avg_loss); metrics['lr'].append(cur_lr)

        if (ep+1) % val_interval == 0 or ep == epochs-1:
            model.eval(); dice_metric.reset()
            with torch.no_grad():
                for vb in val_loader:
                    vi, vl = vb['image'].to(device), vb['label'].to(device)
                    vo = sliding_window_inference(vi, PATCH, 4, model, overlap=0.5, mode='gaussian')
                    dice_metric((torch.sigmoid(vo)>0.5).float(), vl)
            dv = dice_metric.aggregate(); md = dv.mean().item()
            pr = [dv[i].item() for i in range(4)]
            metrics['dice'].append(md); metrics['per_region'].append(pr)
            rs = ' '.join(f'{r}={v:.3f}' for r,v in zip(REGIONS,pr))
            print(f'Ep {ep:3d}/{epochs-1} | L={avg_loss:.4f} | Dice={md:.4f} ({rs}) | LR={cur_lr:.1e} | {(time.time()-t0)/60:.1f}m')
            if md > best_dice:
                best_dice = md; pat_ctr = 0
                torch.save({'epoch':ep,'best_dice':best_dice,'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),'metrics':metrics}, best_path)
                print(f'  ✅ New best!')
            else:
                pat_ctr += 1
                if pat_ctr >= patience: print(f'  Early stop'); break
        else:
            print(f'Ep {ep:3d}/{epochs-1} | L={avg_loss:.4f} | LR={cur_lr:.1e} | {(time.time()-t0)/60:.1f}m')
        torch.save({'epoch':ep,'best_dice':best_dice,'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),'metrics':metrics,'pat_ctr':pat_ctr}, latest_path)

    # Training curves
    fig_dir = OUTPUT_ROOT / 'figures'; fig_dir.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(1,3,figsize=(18,5))
    axes[0].plot(metrics['loss']); axes[0].set_title('Training Loss')
    if metrics['dice']: axes[1].plot(metrics['dice'],marker='o'); axes[1].set_title(f'Val Dice (best={best_dice:.4f})')
    if metrics['lr']: axes[2].plot(metrics['lr']); axes[2].set_title('LR Schedule')
    plt.tight_layout(); plt.savefig(fig_dir/f'{MODEL_NAME}_curves.png',dpi=150); plt.close()

    print(f'\\n  Done: Best Dice = {best_dice:.4f} | {(time.time()-t0)/60:.1f} min')
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['model_state_dict'])
    return model, best_dice, metrics
'''

VIS = '''
def visualize(model, n=3):
    model.eval(); vis_ds = CacheDataset(val_dicts[:n], val_transforms, cache_rate=1.0, num_workers=0)
    fig_dir = OUTPUT_ROOT / 'figures'; fig_dir.mkdir(parents=True,exist_ok=True)
    for i, batch in enumerate(DataLoader(vis_ds,1,False,num_workers=0)):
        vi = batch['image'].to(device); vl = batch['label'].to(device)
        with torch.no_grad():
            vo = sliding_window_inference(vi,PATCH,4,model,overlap=0.5,mode='gaussian')
        pred = (torch.sigmoid(vo)>0.5).float().cpu().numpy()[0]
        gt = vl.cpu().numpy()[0]; img = vi.cpu().numpy()[0,1]
        mid = img.shape[2]//2
        fig,axes = plt.subplots(2,3,figsize=(15,10))
        for c,rn in enumerate(REGIONS):
            axes[0,c].imshow(img[:,:,mid],cmap='gray'); axes[0,c].imshow(gt[c,:,:,mid],alpha=0.3,cmap='Reds')
            axes[0,c].set_title(f'GT {rn}'); axes[0,c].axis('off')
            axes[1,c].imshow(img[:,:,mid],cmap='gray'); axes[1,c].imshow(pred[c,:,:,mid],alpha=0.3,cmap='Blues')
            axes[1,c].set_title(f'Pred {rn}'); axes[1,c].axis('off')
        plt.suptitle(f"{batch['patient_id'][0]}/{batch['timepoint'][0]}",fontsize=14)
        plt.tight_layout(); plt.savefig(fig_dir/f'{MODEL_NAME}_vis_{i}.png',dpi=150); plt.close()
    print(f'  ✅ {n} visualizations saved')
'''

# ══════════════════════════════════════════════════════
#  NOTEBOOK 1: SegResNet
# ══════════════════════════════════════════════════════
c1 = []
c1.append(md("""# Phase 2A — SegResNet Fine-Tuning (BraTS 2024 Post-Treatment)
## CNN Baseline #1: SegResNet from MONAI Model Zoo

**Architecture:** SegResNet (~15M params)
**Pretrained:** BraTS 2018/2021 via MONAI bundle `brats_mri_segmentation`
**Fine-tuning:** Adapt to post-treatment glioma (resection cavities, radiation effects)
**Input:** 4 modalities (T1, T1ce, T2, FLAIR) → 3 sub-regions (WT, TC, ET)"""))

c1.append(code(f"""
from pathlib import Path  # needed before main imports cell
MODEL_NAME = 'segresnet'
PATCH = [128, 128, 128]
REGIONS = ['WT', 'TC', 'ET', 'RC']
OUTPUT_ROOT = Path('/kaggle/working/phase2_segresnet')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f'Model: {{MODEL_NAME}} | Patch: {{PATCH}}')
"""))
c1.append(code(IMPORTS))
c1.append(code(LABEL_CONVERT))
c1.append(code(DATA_LOADING))
c1.append(code(make_transforms("[128,128,128]")))
c1.append(code(DICT_BUILDER))
c1.append(code("""
# ── SegResNet Model + Pretrained Weights ──
from monai.networks.nets import SegResNet

model = SegResNet(spatial_dims=3, in_channels=4, out_channels=4,
                  init_filters=32, blocks_down=(1,2,2,4), blocks_up=(1,1,1), dropout_prob=0.2)
n_params = sum(p.numel() for p in model.parameters())
print(f'SegResNet: {n_params:,} params ({n_params/1e6:.1f}M)')

# Load pretrained weights
WEIGHTS = None
try:
    from monai.bundle import download
    bdir = OUTPUT_ROOT / 'bundles'
    download(name='brats_mri_segmentation', bundle_dir=str(bdir))
    for f in bdir.rglob('*.pt'):
        WEIGHTS = f; break
except Exception as e: print(f'Bundle download failed: {e}')

if WEIGHTS is None:
    for f in Path('/kaggle/input').rglob('*.pt'):
        if 'segresnet' in f.name.lower() or 'brats' in f.name.lower():
            WEIGHTS = f; break

if WEIGHTS:
    sd = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    if 'state_dict' in sd: sd = sd['state_dict']
    msd = model.state_dict()
    matched = {k:v for k,v in sd.items() if k in msd and v.shape == msd[k].shape}
    msd.update(matched); model.load_state_dict(msd)
    print(f'  ✅ Loaded {len(matched)}/{len(msd)} pretrained layers')
else:
    print('  ⚠️ No pretrained weights — training from scratch')
"""))
c1.append(code(TRAIN_FN))
c1.append(code(EXTRACTION))
c1.append(code(make_extract_fn("segresnet",
    "target = model.down_layers[-1] if hasattr(model,'down_layers') else None")))
c1.append(code(VIS))
c1.append(code("""
# ╔════════════════════════════════════════╗
# ║  MAIN: Fine-tune → Extract → Visualize║
# ╚════════════════════════════════════════╝
print('▶ Fine-tuning SegResNet on BraTS 2024 Post-Treatment...')
model, best_dice, metrics = train_model(model, lr=1e-4, epochs=50, patience=15)
print(f'\\n▶ Extracting v2 embeddings...')
extract_embeddings(model)
print(f'\\n▶ Generating visualizations...')
visualize(model)

print(f'\\n{"="*60}')
print(f'  PHASE 2A COMPLETE — SegResNet')
print(f'  Best Dice: {best_dice:.4f}')
print(f'  Outputs: {OUTPUT_ROOT}')
print(f'{"="*60}')

# List all outputs
total = 0
for p in sorted(OUTPUT_ROOT.rglob('*')):
    if p.is_file(): sz=p.stat().st_size; total+=sz; print(f'  {p.relative_to(OUTPUT_ROOT)} ({sz/1e6:.1f}MB)')
print(f'  Total: {total/1e6:.1f} MB')
"""))

save_nb(c1, "/home/moamed/canada_me/explainable_diseas/implementation_brats2024/Phase2/notebooks/Phase2_A1_SegResNet_Finetune.ipynb")

# ══════════════════════════════════════════════════════
#  NOTEBOOK 2: DynUNet (nnU-Net architecture)
# ══════════════════════════════════════════════════════
c2 = []
c2.append(md("""# Phase 2B — DynUNet Fine-Tuning (BraTS 2024 Post-Treatment)
## CNN Baseline #2: DynUNet (nnU-Net architecture via MONAI)

**Architecture:** DynUNet (~31M params) — same architecture as nnU-Net
**Pretrained:** nnU-Net BraTS 2021 checkpoint from Zenodo (256 MB)
**Fine-tuning:** Adapt to post-treatment glioma
**Input:** 4 modalities (T1, T1ce, T2, FLAIR) → 3 sub-regions (WT, TC, ET)

> nnU-Net is the gold standard CNN for medical image segmentation.
> We use MONAI's DynUNet which is architecturally equivalent."""))

c2.append(code(f"""
from pathlib import Path  # needed before main imports cell
MODEL_NAME = 'dynunet'
PATCH = [128, 128, 128]
REGIONS = ['WT', 'TC', 'ET', 'RC']
OUTPUT_ROOT = Path('/kaggle/working/phase2_dynunet')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f'Model: {{MODEL_NAME}} | Patch: {{PATCH}}')
"""))
c2.append(code(IMPORTS))
c2.append(code(LABEL_CONVERT))
c2.append(code(DATA_LOADING))
c2.append(code(make_transforms("[128,128,128]")))
c2.append(code(DICT_BUILDER))
c2.append(code("""
# ── DynUNet Model (nnU-Net architecture) ──
from monai.networks.nets import DynUNet

# nnU-Net-style kernel/strides for 128³ input
kernels = [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
strides = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]

model = DynUNet(
    spatial_dims=3, in_channels=4, out_channels=4,
    kernel_size=kernels, strides=strides, upsample_kernel_size=strides[1:],
    deep_supervision=True, deep_supr_num=3,
)
n_params = sum(p.numel() for p in model.parameters())
print(f'DynUNet: {n_params:,} params ({n_params/1e6:.1f}M)')

# Load pretrained nnU-Net weights (partial — architecture differences expected)
WEIGHTS = None
for f in Path('/kaggle/input').rglob('checkpoint_final.pth'):
    WEIGHTS = f; break
if WEIGHTS is None:
    for f in Path('/kaggle/input').rglob('*.pth'):
        if 'nnunet' in f.name.lower() or 'dynunet' in f.name.lower():
            WEIGHTS = f; break

if WEIGHTS:
    sd = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    if 'network_weights' in sd: sd = sd['network_weights']
    if 'state_dict' in sd: sd = sd['state_dict']
    msd = model.state_dict()
    matched = {k:v for k,v in sd.items() if k in msd and v.shape == msd[k].shape}
    msd.update(matched); model.load_state_dict(msd)
    print(f'  ✅ Loaded {len(matched)}/{len(msd)} nnU-Net pretrained layers')
else:
    print('  ⚠️ No nnU-Net weights found — training from scratch')
    print('  Download from: https://zenodo.org/records/11582627/files/checkpoint_final.pth')
"""))
c2.append(code("""
# DynUNet with deep supervision: modify forward for training vs inference
_orig_forward = model.forward
def dynunet_forward_wrapper(x):
    out = _orig_forward(x)
    if isinstance(out, (list, tuple)):
        return out[0]  # return only main output during inference
    return out

# Override for sliding window inference (expects single tensor output)
model._forward_for_inference = dynunet_forward_wrapper
"""))
c2.append(code(TRAIN_FN.replace(
    "out = model(imgs)",
    "out_all = model(imgs)\n            out = out_all[0] if isinstance(out_all, (list,tuple)) else out_all"
).replace(
    "vo = sliding_window_inference(vi, PATCH, 4, model, overlap=0.5, mode='gaussian')",
    "vo = sliding_window_inference(vi, PATCH, 4, model._forward_for_inference, overlap=0.5, mode='gaussian')"
)))
c2.append(code(EXTRACTION))
c2.append(code(make_extract_fn("dynunet",
    "# Hook the deepest encoder block\\n    target = model.downsamples[-1] if hasattr(model,'downsamples') else None")))
c2.append(code(VIS.replace(
    "vo = sliding_window_inference(vi,PATCH,4,model,overlap=0.5,mode='gaussian')",
    "vo = sliding_window_inference(vi,PATCH,4,model._forward_for_inference,overlap=0.5,mode='gaussian')"
)))
c2.append(code("""
# ╔════════════════════════════════════════╗
# ║  MAIN: Fine-tune → Extract → Visualize║
# ╚════════════════════════════════════════╝
print('▶ Fine-tuning DynUNet on BraTS 2024 Post-Treatment...')
model, best_dice, metrics = train_model(model, lr=1e-4, epochs=50, patience=15)
print(f'\\n▶ Extracting v2 embeddings...')
extract_embeddings(model)
print(f'\\n▶ Generating visualizations...')
visualize(model)

print(f'\\n{"="*60}')
print(f'  PHASE 2B COMPLETE — DynUNet (nnU-Net architecture)')
print(f'  Best Dice: {best_dice:.4f}')
print(f'  Outputs: {OUTPUT_ROOT}')
print(f'{"="*60}')

total = 0
for p in sorted(OUTPUT_ROOT.rglob('*')):
    if p.is_file(): sz=p.stat().st_size; total+=sz; print(f'  {p.relative_to(OUTPUT_ROOT)} ({sz/1e6:.1f}MB)')
print(f'  Total: {total/1e6:.1f} MB')
"""))

save_nb(c2, "/home/moamed/canada_me/explainable_diseas/implementation_brats2024/Phase2/notebooks/Phase2_A2_DynUNet_Finetune.ipynb")

# ══════════════════════════════════════════════════════
#  NOTEBOOK 3: CNN Embedding Comparison
# ══════════════════════════════════════════════════════
c3 = []
c3.append(md("""# Phase 2C — CNN Embedding Comparison
## Comparing SegResNet vs DynUNet Embeddings

**Goal:** Quantitative evaluation of baseline CNN performance + identify limitations of static modeling

This notebook:
1. Loads embeddings from both CNN models
2. Computes embedding quality metrics (diversity, cosine similarity)
3. t-SNE visualization (colored by glioma type, institution)
4. Linear probe: glioma type classification from embeddings
5. **Key analysis:** Limitations of single-timepoint (static) modeling"""))

c3.append(code("""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pandas as pd

OUTPUT_ROOT = Path('/kaggle/working/phase2_comparison')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Load embeddings from both models
models = {}
for name in ['segresnet', 'dynunet']:
    for f in Path('/kaggle/input').rglob(f'cnn_{name}_embeddings_v2.npz'):
        data = np.load(f)
        models[name] = {k: data[k] for k in data.keys()}
        print(f'  {name}: {len(models[name])} embeddings × {list(models[name].values())[0].shape[0]}-dim')
        break
    if name not in models:
        for f in Path('/kaggle/working').rglob(f'cnn_{name}_embeddings_v2.npz'):
            data = np.load(f)
            models[name] = {k: data[k] for k in data.keys()}
            print(f'  {name}: {len(models[name])} embeddings × {list(models[name].values())[0].shape[0]}-dim')
            break

if len(models) < 2:
    print('⚠️ Need both SegResNet and DynUNet embeddings. Run Phase2_A1 and Phase2_A2 first.')
"""))

c3.append(code("""
# ── Embedding Quality Metrics ──
import random

for name, embs in models.items():
    keys = list(embs.keys())
    vals = np.stack([embs[k] for k in keys])
    norms = np.linalg.norm(vals, axis=1)
    vn = vals / (norms[:, None] + 1e-8)
    
    n_pairs = min(200, len(keys)*(len(keys)-1)//2)
    pairs = random.sample([(i,j) for i in range(len(keys)) for j in range(i+1,len(keys))], n_pairs)
    sims = [float(np.dot(vn[i], vn[j])) for i,j in pairs]
    
    print(f'{name}:')
    print(f'  Dimension: {vals.shape[1]}')
    print(f'  Norm: mean={norms.mean():.2f}, std={norms.std():.2f}')
    print(f'  Cosine sim: mean={np.mean(sims):.3f}, std={np.std(sims):.3f}')
    print(f'  Diversity (cos<0.95): {100*np.mean(np.array(sims)<0.95):.0f}%')
    print()
"""))

c3.append(code("""
# ── t-SNE Visualization ──
# Load metadata
meta = None
for f in Path('/kaggle/input').rglob('*.xlsx'):
    meta = pd.read_excel(f); break

for name, embs in models.items():
    keys = list(embs.keys())
    vals = np.stack([embs[k] for k in keys])
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(vals)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if meta is not None:
        # Color by glioma type
        colors_map = {}
        for i, key in enumerate(keys):
            scan_id = key.replace('__', '-')  # patient__tp → scan lookup
            match = meta[meta['BraTS Subject ID'].str.contains(scan_id.split('__')[0], na=False)]
            gtype = match.iloc[0]['Glioma type '] if len(match) > 0 else 'Unknown'
            colors_map[i] = gtype
        
        gtypes = list(set(colors_map.values()))
        cmap = plt.cm.tab10
        for gi, gt in enumerate(gtypes):
            idx = [i for i, g in colors_map.items() if g == gt]
            ax.scatter(coords[idx, 0], coords[idx, 1], c=[cmap(gi)], label=gt, s=10, alpha=0.6)
        ax.legend(fontsize=8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.5)
    
    ax.set_title(f'{name} t-SNE (colored by glioma type)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / f'{name}_tsne.png', dpi=150)
    plt.close()
    print(f'  Saved: {name}_tsne.png')
"""))

c3.append(code("""
# ── Linear Probe: Glioma Type Classification ──
if meta is not None:
    print('=== Glioma Type Classification (Linear Probe) ===')
    for name, embs in models.items():
        keys = list(embs.keys())
        X, y = [], []
        for key in keys:
            pid = key.split('__')[0]
            match = meta[meta['BraTS Subject ID'].str.contains(pid, na=False)]
            if len(match) > 0:
                X.append(embs[key])
                y.append(match.iloc[0]['Glioma type '])
        X = np.stack(X); y = np.array(y)
        le = LabelEncoder(); y_enc = le.fit_transform(y)
        scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X_sc, y_enc, cv=5, scoring='accuracy')
        print(f'  {name}: accuracy={scores.mean():.3f}±{scores.std():.3f} (5-fold CV)')
"""))

c3.append(code("""
# ── Limitations of Static Modeling ──
print('=== LIMITATIONS OF STATIC (SINGLE-TIMEPOINT) MODELING ===')
print()
print('Key finding: Single-timepoint CNN embeddings capture tumor APPEARANCE')
print('but CANNOT model temporal evolution. Specifically:')
print()
print('1. Same patient at different timepoints → different embeddings')
print('   (tumor changes after treatment, but CNN treats each scan independently)')
print()
print('2. No mechanism to track HOW embeddings change over time')
print('   (no temporal attention, no sequence modeling)')
print()
print('3. Treatment response classification requires COMPARING timepoints')
print('   (growing vs stable vs shrinking → needs Δembedding)')
print()
print('→ MOTIVATION FOR PHASE 3: Vision Transformers + Temporal Modeling')
print('   - Extract richer ViT embeddings (self-attention captures global context)')
print('   - Model temporal evolution via embedding sequences')
print('   - Correlate Δembedding with Δvolume (M7 test)')

# Show example of same patient, different timepoints
for name, embs in models.items():
    keys = list(embs.keys())
    patients = {}
    for k in keys:
        pid = k.split('__')[0]
        if pid not in patients: patients[pid] = []
        patients[pid].append(k)
    
    longitudinal = {p: ks for p, ks in patients.items() if len(ks) >= 2}
    print(f'\\n  {name}: {len(longitudinal)} patients with ≥2 timepoints')
    
    # Show embedding distances for first 5 longitudinal patients
    print(f'  Cosine distances between timepoints:')
    for pid, ks in list(longitudinal.items())[:5]:
        e0 = embs[ks[0]] / (np.linalg.norm(embs[ks[0]]) + 1e-8)
        e1 = embs[ks[1]] / (np.linalg.norm(embs[ks[1]]) + 1e-8)
        cos = np.dot(e0, e1)
        print(f'    {pid}: tp0↔tp1 cos={cos:.4f} (distance={1-cos:.4f})')

print(f'\\n✅ Phase 2C Complete — See Phase 3 for temporal modeling')
"""))

save_nb(c3, "/home/moamed/canada_me/explainable_diseas/implementation_brats2024/Phase2/notebooks/Phase2_B1_CNN_Embedding_Comparison.ipynb")

print("\n✅ All Phase 2 notebooks built!")
