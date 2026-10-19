# Source and asset notices

This extraction preserves the current local versions; it does not fetch substitute implementations at runtime.

| Location | Origin and retained purpose |
| --- | --- |
| `retarget/fa/` | Local retarget FA L-BFGS, contact-surface AdamW, and direct trigrasp-style ComFree symbolic acados SQP-RTI |
| `retarget/trigrasp.py` | Local contact-surface AdamW replacement and trigrasp state/rollout utilities |
| `retarget/contact_exploration.py`, `retarget/contact_solvers.py`, `retarget/contact_projection.py`, `retarget/contact_lm_warp.py`, `retarget/contact_memory.py`, AdamW alternating loop | Independently implemented bounded surface projection, sampled contact-set / batched force LM design informed by local scsp-robot `planning/lambda_contact_lm.py` and `lambda_batched_lm.py`; pressure QP and object/hand/scaled-dual splitting informed by [Dexonomy GraspADMM](https://github.com/JYChen18/Dexonomy/blob/main/dexonomy/op/gen_grasp_admm.py). No runtime dependency on either external project; point-contact QP omits torsional moments |
| `retarget/fa/trigrasp_comfree_sqp.py` | Direct CasADi/acados trigrasp MPC with the ComFree soft contact response |
| `retarget/support/` | The used acados discovery, cache and OCP adapter functions from the former planning package; repository discovery and machine paths removed |
| `forceaware/` | Local DexForge ForceAware runtime sources: configuration, time grids, FA loss, targets, interpolation, servo prior, native rollout/adjoint allocation and executor; upstream application/render pipeline and Adam solve loop removed |
| `comfree_warp/` | Complete local custom ComFree runtime and its native adjoints; unused test data and bytecode excluded; runtime source files retain upstream notices |
| `assets/wuji/` | Only referenced right-hand STL and query-point files from local Wuji description and hand assets; XML paths rewritten relative to this repository |
| `data/example/` | Prepared right-hand sugar-box experiment from the successful local FA baseline, with MANO overlay, contact guidance and reference arrays; absolute paths rewritten; unused preparation outputs excluded |
| `data/example/sequence/mano_raw/*.npy`, `semantic_contacts.npz` | Corresponding original MANO trajectory parameters, camera/object transforms and semantic vertex correspondences from the same local FA baseline; used by optional noise preparation. No MANO model weights are included |
| `data/dexycb/` | Five different grasped-object MANO sequences from the official DexYCB 20210415 dataset; see its README for provenance, modifications and attribution |

MuJoCo-Warp and identified derived source files retain Apache-2.0 notices. See `licenses/comfree/Apache-2.0.txt`. ComFree Core retains its **ComFree Core Academic Research License v1.0** (noncommercial research and specified other permitted purposes); see `comfree_warp/comfree_core/LICENSE` and `licenses/comfree_core-LICENSE`. The repository licensing notice is retained at `licenses/ComFree-LICENSE`.

Wuji description assets retain the MIT license and copyright of Wuji Technology at `licenses/Wuji-LICENSE`. Query-point derived assets preserve the local supplied form. DexYCB data retain [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and author attribution; see the [official dataset page](https://dex-ycb.github.io/). The example's object mesh/texture originate from the DexYCB/YCB object assets. MANO model weights are not distributed here.

The ForceAware and local retarget sources were copied from the user's working tree and adapted as described above. No new blanket software license has been assigned to these sources or to third-party assets. Their original rights remain in force. External Python packages, acados, HPIPM and BLASFEO are installed separately and retain their own licenses.
