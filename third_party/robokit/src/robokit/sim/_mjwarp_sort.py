# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportOperatorIssue=false
# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MuJoCo Warp's contact and constraint sorts, which mujoco-warp 3.17.0 leaves as empty stubs."""

# Vendored from google-deepmind/mujoco_warp#1422 (`mujoco_warp/_src/sort.py` at 831a476); delete once a release has it.

import importlib.metadata
import inspect
from typing import Any, Callable, Optional

import warp as wp
from mujoco_warp._src import types  # pyright: ignore[reportMissingImports]
from mujoco_warp._src.warp_util import cache_kernel  # pyright: ignore[reportMissingImports]


wp.set_module_options({"default_grid_stride": False})


@wp.struct
class _ConstraintKeys:
    primary: wp.array2d[wp.vec4i]
    J: wp.array3d[wp.uint32]
    J_rownnz: wp.array2d[int]
    J_rowadr: wp.array2d[int]
    J_colind: wp.array3d[int]
    pos: wp.array2d[wp.uint32]
    margin: wp.array2d[wp.uint32]
    D: wp.array2d[wp.uint32]
    vel: wp.array2d[wp.uint32]
    aref: wp.array2d[wp.uint32]
    frictionloss: wp.array2d[wp.uint32]
    contact: wp.array3d[wp.uint32]  # contact_keys: contact rows compare their contacts, as a contact sort would
    nv: int


@wp.func
def _float_key(bits: wp.uint32) -> wp.uint32:
    """Total ordering of float bits, including signed zero and NaNs."""
    return wp.where((bits & wp.uint32(0x80000000)) != wp.uint32(0), ~bits, bits ^ wp.uint32(0x80000000))


@wp.func
def _contact_less(keys: wp.array3d[wp.uint32], worldid: int, a: int, b: int) -> bool:
    for j in range(keys.shape[2]):
        ka = keys[worldid, a, j]
        kb = keys[worldid, b, j]
        if ka != kb:
            return ka < kb
    return False


@wp.func
def _constraint_less(keys: _ConstraintKeys, worldid: int, a: int, b: int) -> bool:
    ka = keys.primary[worldid, a]
    kb = keys.primary[worldid, b]
    for i in range(4):  # (type, subtype, id, component): connect/weld and contact components differ here
        if ka[i] != kb[i]:
            if i != 2 or ka[0] < int(types.ConstraintType.CONTACT_FRICTIONLESS):
                return ka[i] < kb[i]
            for j in range(keys.contact.shape[2]):  # two contacts: by their keys, as `_contact_less`
                ua = keys.contact[0, ka[2], j]
                ub = keys.contact[0, kb[2], j]
                if ua != ub:
                    return ua < ub

    # Equal type/id is not sufficient for flex edges: compare Jacobian rows, not their atomic allocation addresses.
    if keys.J_rownnz.shape[1] > 0:
        na = keys.J_rownnz[worldid, a]
        nb = keys.J_rownnz[worldid, b]
        aa = keys.J_rowadr[worldid, a]
        ab = keys.J_rowadr[worldid, b]
        for i in range(wp.min(na, nb)):
            ca = keys.J_colind[worldid, 0, aa + i]
            cb = keys.J_colind[worldid, 0, ab + i]
            if ca != cb:
                return ca < cb
            va = _float_key(keys.J[worldid, 0, aa + i])
            vb = _float_key(keys.J[worldid, 0, ab + i])
            if va != vb:
                return va < vb
        if na != nb:
            return na < nb
    else:
        for i in range(keys.nv):
            va = _float_key(keys.J[worldid, a, i])
            vb = _float_key(keys.J[worldid, b, i])
            if va != vb:
                return va < vb

    va = _float_key(keys.pos[worldid, a])
    vb = _float_key(keys.pos[worldid, b])
    if va != vb:
        return va < vb
    va = _float_key(keys.margin[worldid, a])
    vb = _float_key(keys.margin[worldid, b])
    if va != vb:
        return va < vb
    va = _float_key(keys.D[worldid, a])
    vb = _float_key(keys.D[worldid, b])
    if va != vb:
        return va < vb
    va = _float_key(keys.vel[worldid, a])
    vb = _float_key(keys.vel[worldid, b])
    if va != vb:
        return va < vb
    va = _float_key(keys.aref[worldid, a])
    vb = _float_key(keys.aref[worldid, b])
    if va != vb:
        return va < vb
    va = _float_key(keys.frictionloss[worldid, a])
    vb = _float_key(keys.frictionloss[worldid, b])
    if va != vb:
        return va < vb
    return False


@wp.func
def _block_less(keys: wp.array2d[int], worldid: int, a: int, b: int) -> bool:
    return keys[worldid, a] < keys[worldid, b]


@wp.kernel
def _identity(order_out: wp.array2d[int]):
    worldid, i = wp.tid()
    order_out[worldid, i] = i


@cache_kernel
def _merge(compare: Callable):
    @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
    def kernel(
        # In:
        count_in: wp.array[int],
        keys_in: Any,
        order_in: wp.array2d[int],
        width: int,
        # Out:
        order_out: wp.array2d[int],
    ):
        worldid, i = wp.tid()
        count = wp.min(count_in[worldid], order_in.shape[1])
        if i >= count:
            order_out[worldid, i] = i
            return
        start = (i // (2 * width)) * (2 * width)
        middle = wp.min(start + width, count)
        end = wp.min(start + 2 * width, count)
        left = i < middle
        lo = wp.where(left, middle, start)
        hi = wp.where(left, end, middle)
        other_start = lo
        source = order_in[worldid, i]
        while lo < hi:
            mid = (lo + hi) // 2
            other = order_in[worldid, mid]
            before = compare(keys_in, worldid, other, source)
            # Stable merge of *identical* keys, not a key based on emission order.
            if not left:
                before = not compare(keys_in, worldid, source, other)
            if before:
                lo = mid + 1
            else:
                hi = mid
        dest = start + (i - wp.where(left, start, middle)) + lo - other_start
        order_out[worldid, dest] = source

    return kernel


@wp.kernel
def _gather(
    # In:
    count_in: wp.array[int],
    order_in: wp.array2d[int],
    value_in: wp.array3d[Any],
    # Out:
    value_out: wp.array3d[Any],
):
    worldid, row = wp.tid()
    if row < count_in[worldid]:
        source = row  # no order: copy
        if order_in.shape[1] > 0:
            source = order_in[worldid, row]
        for col in range(value_out.shape[2]):
            value_out[worldid, row, col] = value_in[worldid, source, col]


@cache_kernel
def _contact_key_field(is_float: bool, feature: bool):
    @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
    def kernel(
        # Data in:
        nacon_in: wp.array[int],
        # In:
        geom_in: wp.array[wp.vec2i],
        value_in: wp.array3d[wp.uint32],
        offset: int,
        # Out:
        keys_out: wp.array3d[wp.uint32],
    ):
        i, j = wp.tid()
        if i >= wp.min(nacon_in[0], keys_out.shape[1]):
            return
        bits = value_in[0, i, j]
        if wp.static(feature):
            # Rigid writers do not initialize flex feature fields; ignore those slots.
            if geom_in[i][j] >= 0:
                bits = wp.uint32(0xFFFFFFFF)
        if wp.static(is_float):
            key = _float_key(bits)
        else:
            key = bits ^ wp.uint32(0x80000000)
        keys_out[0, i, offset + j] = key

    return kernel


@wp.kernel
def _constraint_counts(
    # Data in:
    nefc_in: wp.array[int],
    njmax_in: int,
    njmax_nnz_in: int,
    # In:
    nnz_in: wp.array[int],
    # Out:
    count_out: wp.array[int],
):
    worldid = wp.tid()
    count = nefc_in[worldid]
    # Overflow may leave partial groups/rows: leave them and the counters alone; the overflow checks report it.
    if count > njmax_in:  # canonical ordering requires sufficient buffers
        count = 0
    if nnz_in.shape[0] > 0:
        if nnz_in[worldid] > njmax_nnz_in:
            count = 0
    count_out[worldid] = count


@wp.kernel
def _constraint_primary(
    # Model:
    eq_type: wp.array[int],
    # In:
    count_in: wp.array[int],
    type_in: wp.array2d[int],
    id_in: wp.array2d[int],
    # Out:
    keys_out: wp.array2d[wp.vec4i],
):
    worldid, row = wp.tid()
    if row < count_in[worldid]:
        row_type = type_in[worldid, row]
        id = id_in[worldid, row]
        subtype = int(0)
        component = int(0)
        if row_type == int(types.ConstraintType.EQUALITY):
            subtype = eq_type[id]
            if subtype == int(types.EqType.CONNECT) or subtype == int(types.EqType.WELD):
                # Assembly emits each group contiguously, XYZ force then XYZ torque.
                nrow = 3 if subtype == int(types.EqType.CONNECT) else 6
                while component + 1 < nrow and row > component:
                    previous = row - component - 1
                    if type_in[worldid, previous] != row_type or id_in[worldid, previous] != id:
                        break
                    component += 1  # the offset within that group, not its nondeterministic allocation address
        # Force reconstruction requires all connects before welds, regardless of XML equality IDs.
        keys_out[worldid, row] = wp.vec4i(row_type, subtype, id, component)


@wp.kernel
def _contact_constraint_primary(
    # Data in:
    nacon_in: wp.array[int],
    # In:
    count_in: wp.array[int],
    worldid_in: wp.array[int],
    efc_address_in: wp.array2d[int],
    type_in: wp.array2d[int],
    id_in: wp.array2d[int],
    dim_in: wp.array[int],
    is_pyramidal: bool,
    # Out:
    keys_out: wp.array2d[wp.vec4i],
):
    conid, component = wp.tid()
    if conid >= wp.min(nacon_in[0], efc_address_in.shape[0]):
        return
    dim = dim_in[conid]
    nrow = wp.max(1, 2 * (dim - 1)) if is_pyramidal else dim
    if component >= nrow:
        return
    worldid = worldid_in[conid]
    row = efc_address_in[conid, component]
    if row >= 0 and row < count_in[worldid]:
        if type_in[worldid, row] >= int(types.ConstraintType.CONTACT_FRICTIONLESS) and id_in[worldid, row] == conid:
            keys_out[worldid, row] = wp.vec4i(type_in[worldid, row], 0, conid, component)


@wp.kernel
def _inverse(order_in: wp.array2d[int], inverse_out: wp.array2d[int]):
    worldid, row = wp.tid()
    inverse_out[worldid, order_in[worldid, row]] = row


@wp.kernel
def _remap_contacts(
    # Data in:
    nacon_in: wp.array[int],
    # In:
    count_in: wp.array[int],
    worldid_in: wp.array[int],
    inverse_in: wp.array2d[int],
    address_in: wp.array2d[int],
    # Out:
    address_out: wp.array2d[int],
):
    conid, component = wp.tid()
    if conid < wp.min(nacon_in[0], address_in.shape[0]):
        row = address_in[conid, component]
        worldid = worldid_in[conid]
        if row >= 0 and row < count_in[worldid]:
            address_out[conid, component] = inverse_in[worldid, row]


@wp.kernel
def _sparse_offsets(
    # In:
    count_in: wp.array[int],
    order_in: wp.array2d[int],
    rownnz_in: wp.array2d[int],
    # Out:
    offsets_out: wp.array2d[int],
):
    worldid = wp.tid()
    offset = int(0)
    # Scan assembled rows within each world, not the (possibly interleaved) contact buffer.
    for row in range(count_in[worldid]):
        offsets_out[worldid, row] = offset
        offset += rownnz_in[worldid, order_in[worldid, row]]


@wp.kernel
def _pack_sparse(
    # In:
    count_in: wp.array[int],
    order_in: wp.array2d[int],
    offsets_in: wp.array2d[int],
    rownnz_in: wp.array2d[int],
    rowadr_in: wp.array2d[int],
    colind_in: wp.array3d[int],
    J_in: wp.array3d[float],
    # Out:
    rownnz_out: wp.array2d[int],
    rowadr_out: wp.array2d[int],
    colind_out: wp.array3d[int],
    J_out: wp.array3d[float],
):
    worldid, row = wp.tid()
    if row >= count_in[worldid]:
        return
    old = order_in[worldid, row]
    nnz = rownnz_in[worldid, old]
    start = rowadr_in[worldid, old]
    adr = offsets_in[worldid, row]
    rownnz_out[worldid, row] = nnz
    rowadr_out[worldid, row] = adr
    for i in range(nnz):
        colind_out[worldid, 0, adr + i] = colind_in[worldid, 0, start + i]
        J_out[worldid, 0, adr + i] = J_in[worldid, 0, start + i]


@wp.kernel
def _remap_blocks(
    # In:
    count_in: wp.array[int],
    nblock_in: wp.array[int],
    inverse_in: wp.array2d[int],
    adr_in: wp.array2d[int],
    nrow_in: wp.array2d[int],
    # Out:
    adr_out: wp.array2d[int],
    nblock_out: wp.array[int],
):
    worldid, block = wp.tid()
    if block == 0:
        nblock_out[worldid] = wp.where(count_in[worldid] > 0, nblock_in[worldid], 0)
    if count_in[worldid] == 0 or block >= nblock_in[worldid]:
        return
    start = adr_in[worldid, block]
    first = inverse_in[worldid, start]
    for i in range(1, nrow_in[worldid, block]):
        first = wp.min(first, inverse_in[worldid, start + i])
    adr_out[worldid, block] = first


def permutation(count: wp.array, capacity: int, keys: wp.array | _ConstraintKeys.cls, compare: Callable) -> wp.array:
    """Stable parallel mergesort, bounded by capacity and compatible with CUDA capture."""
    order = wp.empty((count.size, capacity), dtype=int)
    scratch = wp.empty_like(order)
    wp.launch(_identity, dim=order.shape, outputs=[order])
    width = 1
    while width < capacity:
        wp.launch(_merge(compare), dim=order.shape, inputs=[count, keys, order, width], outputs=[scratch])
        order, scratch = scratch, order
        width *= 2
    return order


def row_view(value: wp.array, nworld: int, capacity: int, dtype=None) -> wp.array:
    """View a contiguous array as scalar rows without moving its storage."""
    scalar = (
        value.dtype._wp_scalar_type_
        if wp.types.type_is_vector(value.dtype) or wp.types.type_is_matrix(value.dtype)
        else value.dtype
    )
    view = value.view(dtype or scalar)
    width = view.size // max(nworld * capacity, 1)
    # Keep the backing allocation alive when a captured gather retains the view.
    return view.reshape((nworld, capacity, width))


def gather(count: wp.array, order: wp.array, value: wp.array, capacity: int):
    """Permute active rows in place (copying only those, for less GPU time), keeping array references and padding."""
    dest_view = row_view(value, count.size, capacity)
    source_view, copy = wp.empty_like(dest_view), wp.empty((0, 0), dtype=int)  # copy: the active rows, not padding
    wp.launch(_gather, dim=dest_view.shape[:2], inputs=[count, copy, dest_view], outputs=[source_view])
    wp.launch(_gather, dim=dest_view.shape[:2], inputs=[count, order, source_view], outputs=[dest_view])


def contact_keys(d: types.Data) -> wp.array:
    """Keys that order contacts by world, features, geometry, then material; never by an allocated contact index.

    Returns:
        `(1, naconmax, fields)` uint32 keys; equal keys describe interchangeable contacts.
    """
    fields = []
    offset = 0
    for value, is_float, feature in (
        (d.contact.worldid, False, False),
        (d.contact.geom, False, False),
        (d.contact.flex, False, True),
        (d.contact.elem, False, True),
        (d.contact.vert, False, True),
        (d.contact.pos, True, False),
        (d.contact.frame, True, False),
        (d.contact.dist, True, False),
        (d.contact.dim, False, False),
        (d.contact.type, False, False),
        (d.contact.includemargin, True, False),
        (d.contact.friction, True, False),
        (d.contact.solref, True, False),
        (d.contact.solreffriction, True, False),
        (d.contact.solimp, True, False),
        (d.contact.adhesion, True, False),
        (d.contact.geomcollisionid, False, False),
    ):
        if not value.size:
            continue
        view = row_view(value, 1, d.naconmax, wp.uint32)
        fields.append((view, offset, is_float, feature))
        offset += view.shape[2]
    keys = wp.empty((1, d.naconmax, offset), dtype=wp.uint32)
    for view, offset, is_float, feature in fields:
        wp.launch(
            _contact_key_field(is_float, feature),
            dim=(d.naconmax, view.shape[2]),
            inputs=[d.nacon, d.contact.geom, view, offset],
            outputs=[keys],
        )
    return keys


def constraints(m: types.Model, d: types.Data, nnz: wp.array | None = None):
    """Sort assembled rows and repair their contact, sparse-J, and JTDAJ references."""
    count = wp.empty_like(d.nefc)
    wp.launch(
        _constraint_counts,
        dim=d.nworld,
        inputs=[d.nefc, d.njmax, d.njmax_nnz, nnz if m.is_sparse and nnz is not None else wp.empty(0, dtype=int)],
        outputs=[count],
    )
    keys = _ConstraintKeys()
    keys.primary = wp.empty((d.nworld, d.njmax), dtype=wp.vec4i)
    keys.J = d.efc.J.view(wp.uint32)
    keys.J_rownnz = d.efc.J_rownnz
    keys.J_rowadr = d.efc.J_rowadr
    keys.J_colind = d.efc.J_colind
    keys.pos = d.efc.pos.view(wp.uint32)
    keys.margin = d.efc.margin.view(wp.uint32)
    keys.D = d.efc.D.view(wp.uint32)
    keys.vel = d.efc.vel.view(wp.uint32)
    keys.aref = d.efc.aref.view(wp.uint32)
    keys.frictionloss = d.efc.frictionloss.view(wp.uint32)
    keys.contact = contact_keys(d)
    keys.nv = m.nv
    wp.launch(
        _constraint_primary,
        dim=keys.primary.shape,
        inputs=[m.eq_type, count, d.efc.type, d.efc.id],
        outputs=[keys.primary],
    )
    # Contact rows compare their contacts' keys, not #1422's contact ranks: its order with or without CONTACTS, faster.
    wp.launch(
        _contact_constraint_primary,
        dim=d.contact.efc_address.shape,
        inputs=[
            d.nacon,
            count,
            d.contact.worldid,
            d.contact.efc_address,
            d.efc.type,
            d.efc.id,
            d.contact.dim,
            m.opt.cone == types.ConeType.PYRAMIDAL,
        ],
        outputs=[keys.primary],
    )
    order = permutation(count, d.njmax, keys, _constraint_less)
    inverse = wp.empty_like(order)
    wp.launch(_inverse, dim=order.shape, inputs=[order], outputs=[inverse])
    wp.launch(
        _remap_contacts,
        dim=d.contact.efc_address.shape,
        inputs=[d.nacon, count, d.contact.worldid, inverse, d.contact.efc_address],
        outputs=[d.contact.efc_address],
    )
    if m.is_sparse:
        offsets = wp.empty_like(order)
        wp.launch(_sparse_offsets, dim=d.nworld, inputs=[count, order, d.efc.J_rownnz], outputs=[offsets])
        wp.launch(
            _pack_sparse,
            dim=order.shape,
            inputs=[
                count,
                order,
                offsets,
                wp.clone(d.efc.J_rownnz),
                wp.clone(d.efc.J_rowadr),
                wp.clone(d.efc.J_colind),
                wp.clone(d.efc.J),
            ],
            outputs=[d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J],
        )
        if m.opt.solver == types.SolverType.NEWTON:
            nblock = wp.zeros_like(d.efc.jtdaj_nblock)
            wp.launch(
                _remap_blocks,
                dim=order.shape,
                inputs=[count, d.efc.jtdaj_nblock, inverse, d.efc.jtdaj_adr, d.efc.jtdaj_nrow],
                outputs=[d.efc.jtdaj_adr, nblock],
            )
            blocks = permutation(nblock, d.njmax, d.efc.jtdaj_adr, _block_less)
            gather(nblock, blocks, d.efc.jtdaj_adr, d.njmax)
            gather(nblock, blocks, d.efc.jtdaj_nrow, d.njmax)
    else:
        gather(count, order, d.efc.J, d.efc.J.shape[1])
    for value in (
        d.efc.type,
        d.efc.id,
        d.efc.pos,
        d.efc.margin,
        d.efc.D,
        d.efc.vel,
        d.efc.aref,
        d.efc.frictionloss,
        d.efc.force,
        d.efc.state,
        d.efc.island,
        d.efc.Jqvel,
    ):
        if value.size:
            gather(count, order, value, value.shape[1])


def patch() -> None:
    """Fill 3.17.0's stub sorts, so `DeterminismType.CONSTRAINT` gives each world one canonical constraint row order.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> patch()
        >>> from mujoco_warp._src import constraint
        >>> constraint._sort_constraints is _sort_constraints or importlib.metadata.version("mujoco-warp") != "3.17.0"
        True
    """
    from mujoco_warp._src import collision_driver, constraint  # pyright: ignore[reportMissingImports]

    stubs = collision_driver._sort_contacts, constraint._sort_constraints
    if importlib.metadata.version("mujoco-warp") != "3.17.0" or not all("TODO" in inspect.getsource(f) for f in stubs):
        return  # a no-op on any other version
    collision_driver._sort_contacts, constraint._sort_constraints = _sort_contacts, _sort_constraints


def _sort_contacts(m: types.Model, d: types.Data):
    """Sort contacts canonically under `DeterminismType.CONTACTS` (#1422's `collision_driver._sort_contacts`)."""
    d.contact.efc_address.fill_(-1)  # #1422 resets reused slots in make_constraint; it runs right after this
    order = permutation(d.nacon, d.naconmax, contact_keys(d), _contact_less)  # on overflow, the retained contacts only
    for value in (
    d.contact.dist, d.contact.pos, d.contact.frame, d.contact.includemargin, d.contact.friction, d.contact.solref,
    d.contact.solreffriction, d.contact.solimp, d.contact.dim, d.contact.geom, d.contact.flex, d.contact.elem,
    d.contact.vert, d.contact.efc_address, d.contact.worldid, d.contact.type, d.contact.geomcollisionid,
    d.contact.adhesion,
  ):  # fmt: skip
        if value.size:
            gather(d.nacon, order, value, d.naconmax)


def _sort_constraints(m: types.Model, d: types.Data, nnz: Optional[wp.array] = None):
    """Sort constraint rows canonically (#1422's `constraint._sort_constraints`)."""
    constraints(m, d, nnz)
