"""Read load-bearing hand/object contacts from an executor, without prediction."""
import re
import numpy as np


def body_fingers(model):
    labels = np.full(model.nbody, -1, dtype=int)
    for body in range(1, model.nbody):
        parent = body
        while parent:
            match = re.match(r'right_finger(\d+)_', model.body(parent).name or '')
            if match:
                labels[body] = int(match.group(1))
                break
            parent = int(model.body_parentid[parent])
    return labels


def loaded_fingers(labels, distances, active, normal_forces, threshold=.05):
    labels = np.asarray(labels)
    valid = ((labels >= 0) & (np.asarray(distances) <= 0.)
             & (np.asarray(active) != 0) & np.isfinite(normal_forces))
    fingers = []
    for finger in np.unique(labels[valid]):
        force = np.maximum(np.asarray(normal_forces)[valid & (labels == finger)], 0.).sum()
        if force >= threshold:
            fingers.append(int(finger))
    return tuple(fingers)


class WarpContactObserver:
    def __init__(self, model, index, compiled):
        body_labels = body_fingers(model)
        geoms = compiled.base.collision.contact_layout.contact_geom.numpy()
        bodies = model.geom_bodyid[geoms]
        self.labels = np.full(len(geoms), -1, dtype=int)
        for side in (0, 1):
            is_object = bodies[:, side] == index.object_body
            self.labels[is_object] = body_labels[bodies[is_object, 1-side]]
        self.row_contacts = compiled.base.contact_rows.parameters.row_contact.numpy()

    def read(self, workspace):
        contacts = workspace.base.collision.contacts
        row_force = workspace.contact.output.force.numpy()[0]
        valid = self.row_contacts >= 0
        forces = np.bincount(self.row_contacts[valid],
                             weights=np.maximum(row_force[:len(valid)][valid], 0.),
                             minlength=len(self.labels))
        return loaded_fingers(self.labels, contacts.distance.numpy()[0],
                              contacts.active.numpy()[0], forces)


class MujocoContactObserver:
    def __init__(self, model, index):
        self.model, self.object_body = model, index.object_body
        self.body_labels = body_fingers(model)

    def read(self, data):
        import mujoco
        labels, distances, forces = [], [], []
        for i in range(data.ncon):
            contact = data.contact[i]
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.object_body not in bodies:
                continue
            other = int(bodies[1] if bodies[0] == self.object_body else bodies[0])
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, wrench)
            labels.append(self.body_labels[other]); distances.append(contact.dist)
            forces.append(max(0., wrench[0]))
        return loaded_fingers(np.asarray(labels), np.asarray(distances),
                              np.ones(len(labels)), np.asarray(forces))
