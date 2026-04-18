import json, ast
cells = []
def md(src):
    cells.append({'cell_type':'markdown','metadata':{},'source':[s+'\n' for s in src.split('\n')]})
def code(src):
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':[s+'\n' for s in src.split('\n')]})

md("""# Validation Step 2.5 — OpenBTAI Embedding Evaluation
**Objective:** Verify that BSF embeddings extracted on the external openBTAI dataset encode the exact same geometry/intensity semantics as they did on the Cyprus training set (the original "16-Test Battery").

If the embeddings are "good", they should achieve comparable $R^2$ scores when linearly probed on OpenBTAI data.""")

code("""import os, numpy as np, pandas as pd, json, warnings
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path("/home/moamed/canada_me/explainable_diseas/implementation_cyprus")
EMB_PATH = ROOT / "Validation" / "openbtai_hybrid_embeddings_v2.npz" # Where you downloaded Kaggle output
PREPROCESS_ROOT = Path("/home/moamed/HDD/validation_data/preprocessed_openbtai")
CLINICAL_CSV = PREPROCESS_ROOT / "openbtai_patient_timelines.csv"

print(f"Embedding file exists: {EMB_PATH.exists()}")
print(f"Preprocessed root exists: {PREPROCESS_ROOT.exists()}")""")

code("""# ── Load Embeddings ───────────────────────────────────────────────────
if EMB_PATH.exists():
    d = np.load(str(EMB_PATH), allow_pickle=True)
    embeddings = {k: d[k] for k in d.files}
    print(f"Loaded {len(embeddings)} BSF embeddings.")
    if len(embeddings) > 0:
        emb_dim = list(embeddings.values())[0].shape[0]
        print(f"Embedding dimension: {emb_dim} (Expected 8466 for hybrid v2)")
        
        # Split BSF (8448) from explicitly appended Shape features (18)
        # We want to probe purely the BSF part to see what it organically learned!
        bsf_only = {k: v[:8448] for k, v in embeddings.items()}
""")

code("""# ── Compute Target Variables to Predict ────────────────────────────────
# To evaluate the embeddings, we extract the ground-truth volume and shape
# directly from the validation dataset masks.
import nibabel as nib
from scipy.stats import skew, kurtosis

shape_dict = {}
for k in bsf_only.keys():
    pid, visit = k.split('__')
    msk_path = PREPROCESS_ROOT / pid / visit / 'mask_subregions.nii.gz'
    if not msk_path.exists(): msk_path = PREPROCESS_ROOT / pid / visit / 'mask_subregions.nii'
    
    if msk_path.exists():
        img = nib.load(str(msk_path))
        mask = img.get_fdata()
        spacing = img.header.get_zooms()[:3]
        
        binary = (mask > 0).astype(np.uint8)
        vol_mm3 = float(binary.sum() * np.prod(spacing))
        
        shape_dict[k] = np.array([
            np.log1p(vol_mm3),                 # M1: log volume
            float(len(np.unique(mask[binary>0]))) # M8: heterogeneity (# compartments)
        ])

print(f"Extracted ground truth targets for {len(shape_dict)} scans.")
""")

code("""# ── Evaluation Probe ───────────────────────────────────────────────────
# We use a constrained PCA -> Ridge regression to prevent overfitting.
class _SafePCA(PCA):
    def fit_transform(self, X, y=None):
        self.n_components = min(self.n_components, X.shape[0]-1, X.shape[1])
        return super().fit_transform(X, y)
    def fit(self, X, y=None):
        self.n_components = min(self.n_components, X.shape[0]-1, X.shape[1])
        return super().fit(X, y)

def run_probe(X, y, name):
    n_comp = min(85, max(2, len(X) // 2))
    pipe = make_pipeline(StandardScaler(), _SafePCA(n_components=n_comp), Ridge(alpha=10.0))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
    mean_r2 = scores.mean()
    status = "✅ GOOD" if mean_r2 > 0.2 else "❌ WEAK"
    print(f"{name:20s}: R² = {mean_r2:7.3f}  {status}")

# Run tests
if len(shape_dict) > 0:
    common = sorted([k for k in bsf_only if k in shape_dict])
    X = np.array([bsf_only[k] for k in common])
    Y = np.array([shape_dict[k] for k in common])
    
    print("="*50)
    print("  OPENBTAI ZERO-SHOT EMBEDDING PROBE TESTS")
    print("="*50)
    run_probe(X, Y[:, 0], "M1: Log Volume")
    run_probe(X, Y[:, 1], "M8: Sub-compartments")
""")

with open('/home/moamed/canada_me/explainable_diseas/implementation_cyprus/Validation/notebooks/Validation_Step2.5_Embedding_Eval.ipynb', 'w') as f:
    json.dump({'cells':cells, 'metadata':{}, 'nbformat':4, 'nbformat_minor':5}, f, indent=2)
print("Notebook created")
