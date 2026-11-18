"""Transport projection only: inactive render geometry stays in its worker."""
WUJI_GEOMETRY=frozenset(('source_vertices_m','source_taxel_positions_m','taxel_positions_m',
    'source_taxel_normals','taxel_normals','link_rotations','link_positions_m','source_joints_m',
    'contact_mask','contact_weights','geometry_f32','retarget_graph'))


def inactive_wuji(state):
    return {k:v for k,v in state.items() if k not in WUJI_GEOMETRY}


def needs_wuji(views,now,recording=False):
    # Legacy clients request the whole state. Explicit views lease only the
    # worker they display; raw recording still requires all worker outputs.
    return recording or not views or any(0<=now-views.get(k,-float('inf'))<2. for k in ('wuji2','all'))
