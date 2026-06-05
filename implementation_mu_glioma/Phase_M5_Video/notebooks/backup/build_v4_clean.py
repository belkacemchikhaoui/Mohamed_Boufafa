import json

nb_path = '/home/moamed/canada_me/explainable_diseas/implementation_mu_glioma/Phase_M5_Video/notebooks/Video_B_Train_v2_128.ipynb'
nb = json.load(open(nb_path))
import copy
nb4 = copy.deepcopy(nb)

# ?? Cell 1: bump ET weight note in title ??????????????????????????????
nb4['cells'][0]['source'] = ["# Phase M5 - Video B v4: ET/TC Push (ETx10 focal + containment + oversample)\n**Target:** TC>=0.80, ET>=0.80\n---"]

# ?? Cell 3: Dataset + Phase 3 (ET-rich oversampling) ?????????????????
nb4['cells'][3]['source'] = ["""# Dataset + ET-rich oversampling (Phase 3)
from torch.utils.data import WeightedRandomSampler

def seg_to_soft(seg, C=5):
    oh = np.zeros((C, seg.shape[0], seg.shape[1]), dtype=np.float32)
    for c in range(C): oh[c] = (seg == c).astype(np.float32)
    return oh * 1.9 - 0.95

class DS(torch.utils.data.Dataset):
    def __init__(s, ss, se, sg, ti, t1c, aug=True):
        s.ss, s.se, s.sg, s.ti, s.t1c, s.aug = ss, se, sg, ti, t1c, aug
    def __len__(s): return len(s.ss)
    def __getitem__(s, i):
        ss = seg_to_soft(s.ss[i]); se = seg_to_soft(s.se[i])
        sg = seg_to_soft(s.sg[i]); gl = s.sg[i].astype(np.int64)
        t = np.float32(s.ti[i]); t1c = s.t1c[i].copy()
        if s.aug:
            if np.random.rand() > 0.5:
                ss=ss[:,:,::-1].copy(); se=se[:,:,::-1].copy()
                sg=sg[:,:,::-1].copy(); gl=gl[:,::-1].copy(); t1c=t1c[:,::-1].copy()
            if np.random.rand() > 0.5:
                ss=ss[:,::-1].copy(); se=se[:,::-1].copy()
                sg=sg[:,::-1].copy(); gl=gl[::-1].copy(); t1c=t1c[::-1].copy()
        return {'cond': torch.from_numpy(np.concatenate([ss, se], 0)),
                'x0': torch.from_numpy(sg), 'gt_labels': torch.from_numpy(gl),
                't_interp': torch.tensor([t]), 't1c': torch.from_numpy(t1c)}

np.random.seed(42); idx = np.random.permutation(N); nv = int(N * CFG['val_split'])
tr = DS(seg_s[idx[nv:]], seg_e[idx[nv:]], seg_g[idx[nv:]], t_interp[idx[nv:]], t1c_s[idx[nv:]], True)
va = DS(seg_s[idx[:nv]], seg_e[idx[:nv]], seg_g[idx[:nv]], t_interp[idx[:nv]], t1c_s[idx[:nv]], False)

# Phase 3: compute per-sample ET pixel count -> oversample ET-rich slices 3x
et_counts = np.array([(seg_g[i] == 4).sum() for i in idx[nv:]], dtype=np.float32)
# Weight: ET-rich (>50 px) gets 3x, rest gets 1x
sample_weights = np.where(et_counts > 50, 3.0, 1.0)
sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), num_samples=len(tr), replacement=True)

tl = torch.utils.data.DataLoader(tr, batch_size=CFG['bs'], sampler=sampler,
                                  num_workers=2, pin_memory=True, drop_last=True)
vl = torch.utils.data.DataLoader(va, batch_size=CFG['bs'], shuffle=False,
                                  num_workers=2, pin_memory=True)
et_rich = (et_counts > 50).sum()
print(f'Train:{len(tr)} Val:{len(va)} | ET-rich slices:{et_rich}/{len(tr)} (3x oversampled)')"""]

# ?? Cell 6: Loss with Phase 1 (ETx10 focal + TC focal) ???????????????
nb4['cells'][6]['source'] = ["""# Phase 1: Hybrid Loss - ETx10 focal + TC focal + wMSE + Dice
# ET weight: 10.0 (was 3.0), TC weight: 3.0 (was 2.0)
CW = torch.tensor([0.1, 2.0, 2.0, 0.1, 10.0], device=device)

def predict_x0(xt, t, noise_pred):
    x0 = (xt - s1mac[t][:,None,None,None]*noise_pred) / sac[t][:,None,None,None].clamp(min=1e-4)
    return x0.clamp(-2, 2)

def soft_dice(logits, target, smooth=1.0):
    p = torch.softmax(logits, 1)
    toh = torch.nn.functional.one_hot(target, 5).permute(0,3,1,2).float()
    inter = (p*toh).sum((2,3)); union = p.sum((2,3)) + toh.sum((2,3))
    return 1 - ((2*inter+smooth)/(union+smooth))[:,1:].mean()

def binary_focal(logits_ch, gt_bin, gamma, alpha):
    """Binary focal loss on a single channel."""
    p = torch.sigmoid(logits_ch)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits_ch, gt_bin.float(), reduction='none')
    pt = torch.where(gt_bin==1, p, 1-p)
    focal_w = alpha * (1-pt)**gamma
    return (focal_w * bce).mean()

def hybrid_loss(noise_pred, noise, xt, t, gl, x0):
    # 1. Spatially weighted MSE (tumor 50x)
    tmask = (gl > 0).float().unsqueeze(1).expand_as(noise)
    w = 1.0 + 49.0 * tmask
    mse = (w * (noise_pred - noise)**2).mean()
    # 2. Dice on predicted x0 (skip background)
    x0p = predict_x0(xt, t, noise_pred)
    dl = soft_dice(x0p, gl)
    # 3. Focal CE (ETx10)
    ce = torch.nn.functional.cross_entropy(x0p, gl, weight=CW)
    # 4. Binary focal for ET (gamma=3, alpha=0.97) - Phase 1
    gt_et = (gl == 4).long()
    fl_et = binary_focal(x0p[:,4], gt_et, gamma=3, alpha=0.97)
    # 5. Binary focal for TC (gamma=2, alpha=0.90) - Phase 1
    gt_tc = ((gl==1)|(gl==4)).long()
    fl_tc = binary_focal(x0p[:,1]+x0p[:,4], gt_tc, gamma=2, alpha=0.90)
    return mse + 0.5*dl + 0.3*ce + 0.4*fl_et + 0.2*fl_tc

print('Loss: wMSE + Dice + CE(ETx10) + focal_ET(?=3) + focal_TC(?=2)')"""]

# ?? Cell 7: Training + Phase 2 (containment at val Dice) ?????????????
nb4['cells'][7]['source'] = ["""# Training + containment post-processing at validation (Phase 2)
import shutil
CKPT = OUT / 'ddpm_v2_128_v4_best.pth'
se = 0; best_dice = 0.0; tl_ = []; vl_ = []; vd_ = []

ckpt_src = None
# Try v4 checkpoint first, then fall back to v2_128 checkpoint
for fname in ['ddpm_v2_128_v4_best.pth', 'ddpm_v2_128_best.pth']:
    for p in sorted(Path('/kaggle/input').rglob(fname)):
        ckpt_src = p; break
    if ckpt_src: break

if ckpt_src:
    ck = torch.load(ckpt_src, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    se = ck.get('epoch', 0) + 1
    best_dice = ck.get('best_dice', 0.0)
    shutil.copy(str(ckpt_src), str(CKPT))
    print(f'? Checkpoint loaded: ep{se} | best_dice={best_dice:.4f}')
else:
    print('No checkpoint - starting from scratch')

def enforce_containment(seg):
    """Phase 2: ET subset TC subset WT - fix hierarchy violations."""
    out = seg.clone()
    wt = out > 0
    tc = (out == 1) | (out == 4)
    et = out == 4
    # ET pixels outside TC -> become TC (label 1)
    et_outside_tc = et & ~tc
    out[et_outside_tc] = 1
    # TC pixels outside WT -> force WT (shouldn't happen but safety)
    tc = (out == 1) | (out == 4)
    tc_outside_wt = tc & ~wt
    out[tc_outside_wt] = 2  # edema
    return out

def dice_sc(p, g):
    r = {}
    for n, pm, gm in [('WT',p>0,g>0), ('TC',(p==1)|(p==4),(g==1)|(g==4)), ('ET',p==4,g==4)]:
        r[n] = (2*(pm&gm).sum().float()/(pm.sum().float()+gm.sum().float()+1e-8)).item()
    return r

if SKIP_TRAIN:
    print('?  SKIP_TRAIN=True - skipping training, proceeding to sampling')
else:
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'], eta_min=1e-6)
    pat = 0
    print(f'Training ep{se}->{CFG["epochs"]} | patience={CFG["patience"]} | FP32')
    t0 = time.time()

    for ep in range(se, CFG['epochs']):
        model.train(); el = []
        for b in tl:
            cond=b['cond'].to(device); x0=b['x0'].to(device)
            gl=b['gt_labels'].to(device); ti=b['t_interp'].to(device)
            B_=cond.shape[0]; t=torch.randint(0,T,(B_,),device=device)
            xt,noise=q_sample(x0,t); np_=model(xt,t,cond,ti)
            loss=hybrid_loss(np_,noise,xt,t,gl,x0)
            if torch.isnan(loss) or torch.isinf(loss): continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); el.append(loss.item())

        sch.step()
        if not el: print(f'  Ep{ep+1}: all NaN'); continue
        tl_.append(np.mean(el))

        model.eval(); vll=[]; vd={'WT':[],'TC':[],'ET':[]}
        with torch.no_grad():
            for b in vl:
                cond=b['cond'].to(device); x0=b['x0'].to(device)
                gl=b['gt_labels'].to(device); ti=b['t_interp'].to(device)
                B_=cond.shape[0]; t=torch.randint(0,T,(B_,),device=device)
                xt,noise=q_sample(x0,t); np_=model(xt,t,cond,ti)
                if torch.isnan(np_).any(): continue
                vll.append(hybrid_loss(np_,noise,xt,t,gl,x0).item())
                pl_raw = predict_x0(xt,t,np_).argmax(1)
                for i in range(B_):
                    # Phase 2: apply containment before computing Dice
                    pl_fixed = enforce_containment(pl_raw[i])
                    ds = dice_sc(pl_fixed, gl[i])
                    for k in ds: vd[k].append(ds[k])

        if not vll: continue
        vl_.append(np.mean(vll))
        md_ = np.mean([np.mean(vd[k]) for k in vd]); vd_.append(md_)
        imp = md_ > best_dice
        if imp:
            best_dice=md_; pat=0
            torch.save({'model':model.state_dict(),'epoch':ep,'best_dice':best_dice,'cfg':CFG}, str(CKPT))
        else:
            pat += 1

        if (ep+1)%5==0 or imp:
            print(f'  Ep{ep+1:3d}/{CFG["epochs"]} l={tl_[-1]:.4f}/{vl_[-1]:.4f} '
                  f'WT={np.mean(vd["WT"]):.3f} TC={np.mean(vd["TC"]):.3f} ET={np.mean(vd["ET"]):.3f} '
                  f'avg={md_:.3f} best={best_dice:.3f} pat={pat} {time.time()-t0:.0f}s')
        if pat >= CFG['patience']: print(f'  Early stop ep{ep+1}'); break

    print(f'Done. Best Dice: {best_dice:.4f}')"""]

# ?? Cell 9 (sampling): apply containment at inference too ?????????????
src = ''.join(nb4['cells'][9]['source'])
src = src.replace(
    "pred=sample_sdedit(model,cond,ti,start_step=25).argmax(1)[0].cpu().numpy()",
    "pred_raw=sample_sdedit(model,cond,ti,start_step=25).argmax(1)[0]\npred=enforce_containment(pred_raw).cpu().numpy()"
)
nb4['cells'][9]['source'] = [src]

out = '/home/moamed/canada_me/explainable_diseas/implementation_mu_glioma/Phase_M5_Video/notebooks/Video_B_Train_v2_128_v4.ipynb'
with open(out, 'w') as f:
    json.dump(nb4, f, indent=1)
print(f'Created {out}')
