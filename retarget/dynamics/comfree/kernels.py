# Copyright (c) 2026 ASU IRIS
# Licensed for noncommercial academic research use only.
# See licenses/comfree_core-LICENSE.
"""Unmodified upstream contact-response kernels; provenance in source.json."""
import warp as wp

wp.set_module_options({"enable_backward": False})

@wp.kernel
def _advance_vel(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_smooth_in: wp.array2d(dtype=float),
  # Data out:
  qvel_out: wp.array2d(dtype=float),
  qfrc_constraint: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_smooth_in[worldid, dofid] * timestep
  qfrc_constraint[worldid, dofid] = 0.0


@wp.kernel
def _compute_qfrc_constraint(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # comfree model parameter
  comfree_stiffness: wp.array(dtype=float),
  comfree_damping: wp.array(dtype=float),
  # Data in: 
  J: wp.array3d(dtype=float),
  efc_dist: wp.array2d(dtype=float),
  efc_mass: wp.array2d(dtype=float),
  qvel_smooth_pred: wp.array2d(dtype=float),
  nv: int,
  nefc: wp.array(dtype=int),
  # Out: 
  efc_force: wp.array2d(dtype=float),
  qfrc_constraint: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  if efcid >= nefc[worldid]:
    return
  
  efc_vel = float(0.0)
  for i in range(nv):
    efc_vel += J[worldid, efcid, i] * qvel_smooth_pred[worldid, i]

  # key steps in the comfree paper
  stiffness = comfree_stiffness[worldid % comfree_stiffness.shape[0]] / timestep
  damping = comfree_damping[worldid % comfree_damping.shape[0]] / timestep
  # predictive penetration with smoothing velocity
  efc_penetration = efc_vel * timestep + efc_dist[worldid, efcid]
  # efc_acc = -damping * efc_b[worldid, efcid] * efc_vel - stiffness * efc_k[worldid, efcid] * efc_penetration
  efc_acc = -damping *  efc_vel - stiffness *  efc_penetration
  efc_frc=  efc_mass[worldid, efcid] * efc_acc
  efc_frc= wp.max(efc_frc, 0.0)

  # output
  efc_force[worldid, efcid] =efc_frc
  for i in range(nv):
    # qfrc_constraint[worldid][i] +=  J[worldid, efcid, i] * efc_frc
    wp.atomic_add(qfrc_constraint, worldid, i, J[worldid, efcid, i] * efc_frc)


@wp.kernel
def _compute_qfrc_total(
  qfrc_smooth: wp.array2d(dtype=float),
  qfrc_constraint: wp.array2d(dtype=float),
  qfrc_total: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qfrc_total[worldid, dofid] = qfrc_smooth[worldid, dofid] + qfrc_constraint[worldid, dofid]
