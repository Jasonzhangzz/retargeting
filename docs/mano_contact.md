# Pose-only MANO contact prediction

The predictor accepts only `pose51[B,T,51]`, `1 <= T <= 8`. Each frame is the
physical wrist position in metres, its axis-angle orientation in radians, and
15 absolute local joint axis-angle rotations. It estimates the current frame;
the input window ends at that frame. A changing camera frame is recorded in
the corpus metadata; use a common world frame when extrinsics are available.

No RGB, image backbone, object mesh, object pose, subject ID, or shape parameter
is passed to the network. Shape and object geometry are used by offline data
preparation. Identical pose histories can correspond to different contacts;
the outputs are conditional probabilities, not a physical contact detector.

## Network and training

Wrist transforms are expressed relative to the first wrist in the window.
Translation is divided by 0.1 m; all 16 rotations become continuous 6D
representations (99 features per frame). The causal Transformer encoder has
256-wide tokens, four layers, eight heads, a 1024-wide FFN and a 64D bottleneck.

Stage 1 uses a `64 -> 256 -> 99` reconstruction MLP. Its objective is L1 pose
reconstruction plus 0.1 supervised contrastive loss (temperature 0.1) and 0.1
frame-contact BCE. Small input perturbations are reconstructed against clean
poses. Unknown labels contribute only to reconstruction. A frame projection
head separates contact from noncontact while the reconstruction retains pose
information. Stage-1 sampling balances labelled contact states across the batch
while retaining the 75/25 temporal/static allocation. Static state proportions
remain natural; real noncontact examples supply the balance. Stage 2 keeps
natural label proportions within each domain and uses HACO's VCB weighting.

Stage 2 removes the reconstruction and contrastive projection heads. Latent
tokens are projected to 1280 channels and passed directly to the original HACO
cross-attention decoder: one query, six layers, width 1024, eight heads, head
width 64. The token adapter changes its input source, not the decoder's
computation. Its four outputs are logits at 778/336/84 vertices and 21 joints;
`frame_contact_logit[B,1]` is an additional head on the last encoder latent.
Lengths are bucketed, so the original HACO attention never sees padded keys.

HACO's four VCB losses, 0.1 prior loss, spatial smooth loss, and an additional
frame BCE train the model. Class counts and contact priors come only from the
train split. Effective-number beta is 0.9999; vertex/global frequency mixing
ramps from 0 to 0.3. The released false-positive penalty is retained. The sparse
mesh adjacency preserves the original repeated triangle-edge weights.

Encoder parameters receive gradients in stage 2. After each optimizer step,
the full inference model is EMA-updated with decay 0.999, including its decoder
and frame head. Checkpoints contain the online and EMA states, optimizer state,
step, encoder configuration and random states. Validation and inference load
the complete EMA model by default. No frame-based hard gate is imposed on the
HACO vertex outputs.

## Data contract

`mano_contact_v1.h5` contains `records/<source-key>/` groups. Each group stores
`pose51[N,51]`, binary `contact[N,778]`, row-level `contact_valid[N]`, and sorted
`frame_id[N]`, plus source-specific provenance and quality fields. Attributes
identify dataset, source, temporal/static kind, leakage group, split, coordinate
frame, hand side and label definition. All 778 labels in a known row must be
known; partially labelled rows must be masked until a full-vertex annotation
is available.

Unknown contact arrays contain zeros for storage efficiency but are excluded
from contact objectives, class statistics and evaluation. Real temporal windows
must contain consecutive frame IDs. Static grasps use one frame; optimizer
iterations and rollout horizons are never promoted into motion sequences.

Completed groups are immutable. A single writer lock protects each corpus.
Data is first written into `__pending`, flushed, then committed to `records`.
On restart the incomplete group is discarded. Generated-data workers write
independent resumable shards, which the parent merges into the unified file.

Generated sources are frozen in `generated_snapshot.json` using successful
verification receipts. Their checksums and model checksums are verified before
reading. Duplicate file hashes are excluded. Aligned MANO/WUJI HDF5 exports and
seed files are derivatives of these source files and are not counted again.
Completed generated rows must be finite, valid and rollout-stable. Stability
failures are never used as noncontact labels. Source wrist positions are
checked against the saved MANO joints.

Dense generated labels use the corresponding calibrated MANO LBS and the exact
closest point on the object triangle mesh, not nearest vertices or sampled
points. CPU Warp BVHs implement distance queries. The contact threshold is
10 mm. Unsigned surface distance defines contact; original penetration quality
is retained separately. Generated grouping uses object identity across scale
variants and hand variants. Real grouping keeps both hands and synchronized
camera views of a physical sequence together. EPIC groups by original video.
Group-hashed splits are 80/10/10 with seed 42; these are custom pose splits,
not scores on the original datasets' image benchmarks.

HOI4D poses use `use_pca=False, flat_hand_mean=True`. DexYCB PCA values are
expanded with its MANO components and mean. H2O uses official cam4 pose packs,
not duplicate copies in `manolabel_v1.1`; camera-to-world extrinsics are composed
when present. Left-hand rotations are reflected as proper rotations and keep
MANO vertex IDs. Source beta parameters are retained outside network inputs.

HACO label IDs differ between datasets; the adapters follow the original
dataset loaders. H2O includes dense geometric labels for missing released
labels when object pose and mesh exist. EPIC central manual annotations are
stored as a separate label definition. Propagated poses must not be treated
as contact-transition ground truth. Verified propagated poses are retained as
independent, unknown-label reconstruction observations at 0.25 times their
confidence weight. The central-frame FK is checked against its released mesh;
the source's constant translation offset is corrected and recorded. FK failures
keep pose reconstruction data but mask contact supervision.

The primitive grasp export is a derivative of the personalized fitting run.
Its saved MANO vertices and joints provide the authoritative posed geometry;
labels use the exact primitive triangle mesh. The paired HDF5 derivative is
not added again. Primitive sizes and grasp types share an object-family split.
Mirroring reflects the source geometry and preserves vertex IDs; it does not
replace source-specific shape blend shapes with a different MANO template.

## Commands

Install `requirements-mano-contact.txt`. The existing remote `pi0` environment
already has the required runtime packages. Use the new module inside the
existing retarget checkout; no simulation job needs to be interrupted.

```bash
python -m retarget.learning.mano_contact.cli snapshot \
  --source-root /mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget \
  --output /mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang/MANO_contact/generated_snapshot.json

python -m retarget.learning.mano_contact.cli prepare \
  --dataset /mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang \
  --source-root /mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget \
  --assets /mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang/MANO_contact/assets \
  --output /mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang/MANO_contact/mano_contact_v1.h5 \
  --workers 8 --threads 4

python -m retarget.learning.mano_contact.cli train --stage 1 \
  --corpus CORPUS.h5 --assets ASSETS --output RUN/stage1 \
  --config configs/mano_contact.json --steps 100000 --device cpu

python -m retarget.learning.mano_contact.cli train --stage 2 \
  --stage1 RUN/stage1/last.pt --corpus CORPUS.h5 --assets ASSETS \
  --output RUN/stage2 --config configs/mano_contact.json --steps 50000 --device cpu

python -m retarget.learning.mano_contact.cli evaluate \
  --checkpoint RUN/stage2/last.pt --corpus CORPUS.h5 --assets ASSETS \
  --split val --output RUN/validation.json

python -m retarget.learning.mano_contact.cli infer \
  --checkpoint RUN/stage2/last.pt --assets ASSETS \
  --pose-npy pose51.npy --output contacts.npz
```

`--device cuda:N --bf16` explicitly selects GPU training. The default is CPU,
including `--device auto`; availability alone does not mean a GPU is unused.
Full training is intentionally not launched for the small-verification task.
`--resume` continues the same stage; stage 2 also needs the original `--stage1`.
Keep the corpus and training configuration fixed when resuming. Resume skips
sampled indices without rereading past HDF5 samples and preserves the model RNG
independently of DataLoader setup.

For an authorized full-data run, `scripts/train_mano_contact_full.py` consumes
a frozen `MANIFEST.json` and chains CPU/GPU numerical prechecks (zero optimizer
updates), stage 1, stage 2, full natural validation/test and EMA export. Its
`CONTROL.json` chooses idle-only admission or explicitly authorized resource
sharing. A controller lock prevents duplicate launch. Source hashes, corpus
SHA256, device process inventories, worker receipts and complete stage
checkpoints are retained. A `STOP` file requests a checkpoint at the next
optimizer boundary and prevents the following stage from starting.

The 2026-11-10 full run uses one GPU with an actual batch of 8192, FP32, and
10 effective epochs per stage. An effective epoch counts weighted replacement
sample draws divided by the size of that stage's complete train pool; the
75/25 temporal/static sampler remains active. It does not mean visiting every
row once. The exact update budgets are in the run's frozen config. `--preload`
caches only pose, labels and reconstruction weights in host RAM to avoid many
small HDF5 reads; data, losses and gradients match the ordinary reader exactly.
Prechecks and startup receipts determine whether the requested batch fits;
do not infer a successful GPU launch from configuration alone.

Small verification uses 16 distinct real contact and 16 real noncontact
windows from train only, both full objectives, and the complete HACO head:

```bash
python -m retarget.learning.mano_contact.cli smoke \
  --corpus CORPUS.h5 --assets ASSETS --output RUN/smoke --threads 4
python -m pytest tests/learning/test_mano_contact.py -q
```

The gate checks >=70% reconstruction reduction, vertex F1 >=0.90 and frame F1
>=0.95 for both the online and EMA models, verifies encoder gradients, the exact
EMA update, and checkpoint restoration. The default small run uses 600 stage-1
and 5000 stage-2 steps to allow the constant-decay EMA to converge. This run is
not sufficient to assess full-data generalization. Evaluate the natural val
distribution separately. The fixed vertex threshold is 0.52; only the val
command computes an additional calibrated threshold. The test command uses
the explicitly supplied fixed threshold.

For large CPFS datasets, prepare on a sufficiently large local scratch volume,
then checksum and atomically copy the completed HDF5 to the dataset directory.
The file remains self-contained; temporary shards are merged before delivery.
Static batches retain per-row `leakage_id` values when multiple objects share a
container. The auditor checks those IDs against legacy per-object groups.
Bounded, unsigned BVH queries preserve exact 10 mm labels while rejecting distant
vertices quickly. A missing minimum distance outside that radius is censored
(`NaN`), not a measured zero. Original generated penetration diagnostics remain
available separately. A shared scratch mesh cache is capped at 8 GiB and
checks source mesh size and modification time before reuse. It can be removed
between independent dataset versions; it is not part of the delivered corpus.

Independent source preparations can use separate scratch files. Once all are
complete, merge them transactionally (the output may already hold completed
generated data):

```bash
python -m retarget.learning.mano_contact.cli merge \
  --inputs hoi4d.h5 h2o.h5 dexycb.h5 epic.h5 primitives.h5 \
  --output mano_contact_v1.h5
```

Export contains the complete EMA encoder, adapter, decoder, frame head and
three regressor assets. It does not require licensed MANO geometry at inference:

```bash
python -m retarget.learning.mano_contact.cli export \
  --checkpoint RUN/stage2/last.pt --output RUN/export
python -m retarget.learning.mano_contact.cli infer \
  --checkpoint RUN/export/model.pt --assets RUN/export/assets \
  --pose-npy pose51.npy --output contacts.npz
```

The delivered `output/mano_contact_20261109/export/model.pt` is a **32-example
verification model**, not a model trained on the consolidated corpus. Its
training-set overfit metrics must be read separately from natural validation
and test metrics in the accompanying report.

## Assets and attribution

Supply licensed standard `MANO_RIGHT.pkl` / `MANO_LEFT.pkl`. `assets` builds the
HACO 21-joint regressor and face table. `fetch-regressors --assets ASSETS --cache
CACHE` retrieves the exact `V_regressor_336.npy` and `V_regressor_84.npy` from
HACO's archive, retaining download provenance and hashes. Its range cache can
be put on a large temporary filesystem; it stops once the required assets are
available in the contiguous archive prefix. MANO weights and datasets are
not included in source control.

HACO decoder code is derived from upstream commit
`c774211b968bdf836f9511dbd5ff23bb19fe1c44`; source hashes are recorded in
`retarget/learning/mano_contact/UPSTREAM.json`. Its CC BY-NC 4.0 license is
preserved in `third_party/HACO_CONTACT_LICENSE.txt`. See
[HACO](https://github.com/dqj5182/HACO_RELEASE) and
[CARD](https://arxiv.org/html/2610.36539) for the contact formulation.
