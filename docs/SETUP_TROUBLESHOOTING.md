# Setup & Inference Troubleshooting

Issues and fixes discovered while setting up the environment and running inference for the first time.

---

## 1. `psbody-mesh` not on PyPI

`conda_env.yml` originally contained `mesh==1.0.0a1` as a stand-in for
`psbody-mesh`. That package is no longer maintained and its API differs slightly.
**Fix:** remove the pip entry and build from the MPI-IS GitHub repo directly.

```bash
# After creating the conda env (without mesh==1.0.0a1):
git clone https://github.com/MPI-IS/mesh.git /tmp/mesh
cd /tmp/mesh
# Bypass Makefile — its --install-option flag was removed in pip 23
/root/miniconda3/envs/diffposetalk/bin/python setup.py install \
    --boost-location=/usr/include
```

System packages `libboost-dev` and `freeglut3-dev` must be present
(`apt-get install -y libboost-dev freeglut3-dev`); they were pre-installed in
this environment.

---

## 2. `chumpy` incompatible with NumPy ≥ 1.20

`chumpy==0.70` does `from numpy import bool, int, float, ...` — aliases removed
in NumPy 1.20 and deleted in 1.24. This manifests as an `ImportError` when
the FLAME pkl file is unpickled (which imports chumpy internally).

**Fix:** patch the installed `chumpy/__init__.py`:

```python
# Replace:
from numpy import bool, int, float, complex, object, unicode, str, nan, inf

# With:
from numpy import nan, inf
import builtins as _builtins
bool    = _builtins.bool
int     = _builtins.int
float   = _builtins.float
complex = _builtins.complex
object  = _builtins.object
str     = _builtins.str
unicode = _builtins.str
```

File location:
`$CONDA_PREFIX/lib/python3.8/site-packages/chumpy/__init__.py`

---

## 3. Pose dimension mismatch when running `demo.py`

The pretrained checkpoint `head-SA-hubert-WM` (iter 110000) was trained on the
`HDTF_THHQ` dataset, which stores FLAME pose as **6-dimensional**
(3 global rotation + 3 jaw rotation, axis-angle).

The default `--coef_stats` path points to
`datasets/HDTF_TFHP/lmdb/stats_train.npz`, which has **9-dimensional** pose
stats (includes neck joints). This causes:

```
RuntimeError: The size of tensor a (6) must match the size of tensor b (9)
at non-singleton dimension 2
```

The TFHP pose stats are all-zero mean / all-ones std (identity normalization),
so truncating to 6 dims is functionally equivalent.

**Fix:** create a 6-dim pose variant of the stats file:

```python
import numpy as np
stats = np.load('datasets/HDTF_TFHP/lmdb/stats_train.npz')
new_stats = {}
for k, v in stats.items():
    new_stats[k] = v[:6] if k.startswith('pose_') else v
np.savez_compressed('datasets/HDTF_TFHP/lmdb/stats_train_6dpose.npz', **new_stats)
```

Then pass `--coef_stats datasets/HDTF_TFHP/lmdb/stats_train_6dpose.npz` to
`demo.py`.

---

## Working demo command

```bash
conda activate diffposetalk
python demo.py \
    --exp_name head-SA-hubert-WM --iter 110000 \
    -a demo/input/audio/FAST.flac \
    -c demo/input/coef/TH217.npy \
    -s demo/input/style/TH217.npy \
    -o TH217-FAST-TH217.mp4 \
    -n 3 -ss 3 -sa 1.15 -dtr 0.99 \
    --coef_stats datasets/HDTF_TFHP/lmdb/stats_train_6dpose.npz
```

Output (with `-n 3`) is written to three files:
`TH217-FAST-TH217_000.mp4`, `_001.mp4`, `_002.mp4` under
`demo/output/head-SA-hubert-WM/iter_0110000/`.

Note: the style file path `demo/input/style/TH217.npy` does not exist literally;
`demo.py` auto-resolves it to
`demo/input/style/<style_enc_subdir>/TH217.npy` based on the style encoder
checkpoint embedded in the DPT model args.
