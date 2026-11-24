"""CART recording identities and current measured-EMF policy architecture."""

LEGACY_PROTOCOL = 'cart_contact_v3'
TUBE_PROTOCOL = 'cart_contact_tube_v4'
PROTOCOL = 'cart_diff_ik_v6'
SUPPORTED_RECORDINGS = (PROTOCOL, 'cart_diff_v5', TUBE_PROTOCOL, LEGACY_PROTOCOL)
ARCHITECTURES = {'attention': 'cart_emf_cross_attention128_finger_delta_v5'}
POLICY_OUTPUT = 'normalized_delta_q_v1'
POLICY_INPUT = 'future_emf10_pose9_v1'
POLICY_TRANSITION = 'current_q_plus_normalized_delta_v2'
ARCHITECTURE = ARCHITECTURES['attention']
LOOKAHEAD = 10
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')

V9_OBJECTIVE = 'card_raw_neural_self_contact_v4_actuated'
WRIST_OBJECTIVE = 'card_raw_neural_self_contact_v5_wrist'
LINK_SDF_OBJECTIVE = 'card_style_link_sdf_contacts_v6_wrist'
CONDITIONAL_SDF_OBJECTIVE = 'card_conditional_sdf_v7_wrist'
OBJECTIVES = (V9_OBJECTIVE, WRIST_OBJECTIVE, LINK_SDF_OBJECTIVE, CONDITIONAL_SDF_OBJECTIVE)
