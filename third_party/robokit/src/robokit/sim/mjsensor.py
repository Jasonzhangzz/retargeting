"""`ContactSensor`: MuJoCo contact sensors in a `MjWarpSim` read together, as torch tensors."""

import re
from typing import Any, Optional

import torch


class ContactSensorData:
    """Contact readings as torch tensors read from `sim.data.torch`, so `torch.compile` works.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import mujoco
        >>> from robokit.sim import MjWarpSim
        >>> xml = '<mujoco><worldbody><geom type="plane" size="1 1 1"/><body pos="0 0 .1"><freejoint/>'
        >>> xml += '<geom name="ball" size=".1"/></body></worldbody><sensor>'
        >>> xml += '<contact name="touch_ball_found" geom1="ball" data="found"/></sensor></mujoco>'
        >>> sim = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2, substeps=2)
        >>> sim.step()
        >>> data = sim.build_contact_sensor("touch").data
        >>> data.found.tolist(), tuple(data.found_history.shape)  # (substeps, nworld, primaries)
        ([[1.0], [1.0]], (2, 2, 1))
    """

    def __init__(self, sim: Any, found: list, force: list, track_air_time: bool):
        self._views = sim.data.torch
        device = self._views.sensordata.device
        self._found = torch.tensor(found, dtype=torch.long, device=device)  # (primaries,)
        self._force = torch.tensor(force, dtype=torch.long, device=device).reshape(-1, 3)  # (primaries, 3)
        primaries = max(len(found), len(force))
        self.current_air_time: Optional[torch.Tensor] = None  # (nworld, primaries), s, with `track_air_time`
        self.current_contact_time: Optional[torch.Tensor] = None
        if track_air_time:
            self.current_air_time = torch.zeros(sim.nworld, primaries, device=device)
            self.current_contact_time = torch.zeros(sim.nworld, primaries, device=device)

    @property
    def found(self) -> torch.Tensor:
        """(nworld, primaries): the contacts each found after the last substep."""
        return self._views.sensordata[:, self._found]

    @property
    def force(self) -> torch.Tensor:
        """(nworld, primaries, 3): each one's contact force after the last substep."""
        return self._views.sensordata[:, self._force]

    @property
    def found_history(self) -> torch.Tensor:
        """(substeps, nworld, primaries): `found` after each substep of the last step."""
        return self._views.substep_sensordata[:, :, self._found]

    @property
    def force_history(self) -> torch.Tensor:
        """(substeps, nworld, primaries, 3): `force` after each substep of the last step."""
        return self._views.substep_sensordata[:, :, self._force]


class ContactSensor:
    """The MuJoCo contact sensors `{name}_{primary}_{found or force}`, one per primary.

    Args:
        track_air_time: Let `update()` count each primary's air and contact time per substep.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import mujoco
        >>> from robokit.sim import MjWarpSim, mjscene
        >>> spec = mjscene.build_scene_spec()
        >>> ball = spec.worldbody.add_body(name="ball", pos=[0, 0, 0.1])
        >>> _ = ball.add_freejoint()
        >>> _ = ball.add_geom(name="ball", size=[0.1, 0, 0])
        >>> for field in ("found", "force"):
        ...     _ = mjscene.build_contact_sensor(
        ...         spec, f"touch_ball_{field}", (mujoco.mjtObj.mjOBJ_GEOM, "ball"), data=(field,))
        >>> sim = MjWarpSim(spec.compile(), nworld=2, substeps=2)
        >>> touch = sim.build_contact_sensor("touch", track_air_time=True)
        >>> sim.step()
        >>> touch.update()
        >>> touch.primary_names, tuple(touch.data.force.shape), (touch.data.current_contact_time > 0).tolist()
        (['ball'], (2, 1, 3), [[True], [True]])
    """

    def __init__(self, sim: Any, name: str, track_air_time: bool = False):
        mjm = sim.mjm
        pattern = re.compile(re.escape(name) + r"_(.+)_(found|force)")
        self.primary_names = []
        found = {}
        force = {}
        for i in range(mjm.nsensor):
            match = pattern.fullmatch(mjm.sensor(i).name)
            if match is None:
                continue
            primary, field = match.groups()
            if primary not in self.primary_names:
                self.primary_names.append(primary)
            adr = int(mjm.sensor_adr[i])
            if field == "found":
                found[primary] = adr
            else:
                force[primary] = [adr, adr + 1, adr + 2]
        self.substep_dt = mjm.opt.timestep
        found_adr = [found[p] for p in self.primary_names if p in found]
        force_adr = [force[p] for p in self.primary_names if p in force]
        self.data = ContactSensorData(sim, found_adr, force_adr, track_air_time)

    def update(self):
        """Count each primary's air and contact time over the last step's substeps, from `found`.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> import mujoco
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom name="ball" size=".1"/></body>'
            >>> xml += '</worldbody><sensor><contact name="touch_ball_found" geom1="ball" data="found"/></sensor>'
            >>> xml += '</mujoco>'
            >>> sim = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2, substeps=2)
            >>> touch = sim.build_contact_sensor("touch", track_air_time=True)
            >>> sim.step()
            >>> touch.update()
            >>> (touch.data.current_air_time > 0).tolist(), (touch.data.current_contact_time > 0).tolist()
            ([[True], [True]], [[False], [False]])
        """
        air = self.data.current_air_time
        contact = self.data.current_contact_time
        assert air is not None and contact is not None, "needs track_air_time"
        for found in self.data.found_history:
            touching = found > 0
            air.copy_(torch.where(touching, 0.0, air + self.substep_dt))
            contact.copy_(torch.where(touching, contact + self.substep_dt, 0.0))

    def compute_first_contact(self, dt: float, abs_tol: float = 1e-6) -> torch.Tensor:
        """(nworld, primaries): the primaries that touched down within the last `dt` s.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> import mujoco
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><geom type="plane" size="1 1 1"/><body pos="0 0 .1"><freejoint/>'
            >>> xml += '<geom name="ball" size=".1"/></body></worldbody><sensor>'
            >>> xml += '<contact name="touch_ball_found" geom1="ball" data="found"/></sensor></mujoco>'
            >>> sim = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2, substeps=2)
            >>> touch = sim.build_contact_sensor("touch", track_air_time=True)
            >>> sim.step()
            >>> touch.update()
            >>> touch.compute_first_contact(dt=sim.substeps * sim.mjm.opt.timestep).tolist()
            [[True], [True]]
        """
        contact = self.data.current_contact_time
        assert contact is not None, "needs track_air_time"
        return (contact > 0) & (contact < dt + abs_tol)

    def reset(self, mask: torch.Tensor):
        """Zero the air and contact times in the worlds where `mask` (nworld,) is set.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> import mujoco
            >>> from robokit.sim import MjWarpSim
            >>> xml = '<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom name="ball" size=".1"/></body>'
            >>> xml += '</worldbody><sensor><contact name="touch_ball_found" geom1="ball" data="found"/></sensor>'
            >>> xml += '</mujoco>'
            >>> sim = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2, substeps=2)
            >>> touch = sim.build_contact_sensor("touch", track_air_time=True)
            >>> sim.step()
            >>> touch.update()
            >>> touch.reset(torch.tensor([True, False], device=touch.data.found.device))
            >>> (touch.data.current_air_time > 0).tolist()
            [[False], [True]]
        """
        for time in (self.data.current_air_time, self.data.current_contact_time):
            if time is not None:
                time.copy_(torch.where(mask[:, None], 0.0, time))
