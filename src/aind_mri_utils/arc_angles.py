"""Tools for computing arc angles, stereotaxic angles, and the conversions between them.

Angle naming
------------
Parameter names refer to the *axis of rotation* in a RAS frame, not the
colloquial clinical alias for the angle:

* ``rx`` — rotation about the x-axis (the ML axis in RAS).
  Aliases: "AP angle" / "AP tilt" / mouse-pitch (nose up/down).
  Tilts the probe in the AP plane.
* ``ry`` — rotation about the y-axis (the AP axis in RAS).
  Aliases: "ML angle" / "ML tilt" / mouse-roll (left/right ear up).
  Tilts the probe in the ML plane.
* ``rz`` — rotation about the z-axis (the DV axis in RAS).
  Aliases: spin / mouse-yaw.

The pitch/roll/yaw labels above use the *mouse-physical* sense (mouse's
forward is +y/AP). Note that ``rotations.py`` defines functions named
``roll`` / ``pitch`` / ``yaw`` using the *aerospace* sense (forward is +x),
which makes ``rotations.pitch`` a Y-axis rotation — i.e. a mouse-roll for
RAS data. See the ``rotations`` module docstring.

The arc rx convention is *not* right-handed — ``invert_rx=True`` (the default)
flips the sign so that "AP angle = +14" colloquially corresponds to a
right-handed rotation of −14° about ML. ``invert_rz=True`` does the same for
spin.

Frames
------
Several RAS frames may appear in callers of this module. Common ones:

* **Headframe (plan) frame.** The mouse's lambda–bregma plane is the xy
  plane; +z is normal to it. A probe perpendicular to lambda–bregma has
  rx = ry = 0 here.
* **Arc / ephys-rig frame.** When mounted on the AIND ephys rig, the
  headframe is pitched down by ~14° (a right-handed −14° rotation about
  ML), so a probe vertical in headframe reads as arc (rx=14, ry=0). In the
  AIND non-right-handed arc convention, ``arc_rx = plan_rx + 14``.
* **Off-plane stereotax frame.** Used by the Kopf 1500 off-plane insertion
  tool; angles are polar/azimuthal from stereotax-vertical. An earbar
  gimbal can re-tilt the headframe within this frame.

Which RAS frame a given vector lives in (headframe / arc / stereotax)
depends on context; individual function docstrings call it out. Vector
inputs and outputs are RAS (x=right, y=anterior, z=superior) with
components ordered (ML, AP, DV) unless documented otherwise.

Note that ``arc_angles_to_affine`` returns an **LPS** transform (for
ITK/Slicer interop), unlike the rest of this module which is RAS-centric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

from aind_mri_utils.rotations import ras_to_lps_transform

# Euler sequence (intrinsic) defining the arc-angle rotation convention. Used by
# both the forward (arc_angles_to_rotation) and inverse (vector_to_arc_angles)
# so the order is stated exactly once.
_ARC_EULER_SEQ: Final = "XYZ"

# A residual Kopf polar tilt below this (radians) is treated as "vertical
# reached" when reporting whether the earbar fully absorbed the insertion tilt.
_VERTICAL_TILT_TOL_RAD: Final = math.radians(0.05)


def arc_angles_to_rotation(
    rx: float,
    ry: float,
    rz: float = 0.0,
    degrees: bool = True,
    invert_rx: bool = True,
    invert_rz: bool = True,
) -> Rotation:
    """Build the RAS rotation for arc angles (rx, ry, rz).

    This is the single source of truth for the arc-angle rotation
    convention. :func:`arc_angles_to_vector` and :func:`arc_angles_to_affine`
    are both derived from it, so they cannot drift apart.

    The angles are composed as intrinsic XYZ Euler rotations (``Rx`` outermost,
    ``Ry``, then ``Rz``), after the AIND non-right-handed sign flips. Applied
    to the probe's neutral axis ``[0, 0, 1]`` (straight up in RAS), this yields
    the probe direction returned by :func:`arc_angles_to_vector`.

    Parameters
    ----------
    rx : float
        Rotation about the x/ML axis. Alias: AP angle, pitch.
    ry : float
        Rotation about the y/AP axis. Alias: ML angle, roll.
    rz : float, optional
        Rotation about the z/DV axis. Alias: spin, yaw. Default 0.
    degrees : bool, optional
        If True, input angles are in degrees; otherwise radians (default True).
    invert_rx : bool, optional
        If True, apply the AIND non-right-handed sign convention to rx
        (default True). See module docstring.
    invert_rz : bool, optional
        If True, apply the AIND non-right-handed sign convention to rz
        (default True).

    Returns
    -------
    scipy.spatial.transform.Rotation
        The rotation, in RAS, described by the arc angles.
    """
    if invert_rx:
        rx = -rx
    if invert_rz:
        rz = -rz
    return Rotation.from_euler(_ARC_EULER_SEQ, [rx, ry, rz], degrees=degrees)


def vector_to_arc_angles(
    vec: NDArray[np.floating[Any]],
    degrees: bool = True,
    invert_rx: bool = True,
) -> tuple[float, float] | None:
    """Calculate the arc angles (rx, ry) for a given direction vector.

    Parameters
    ----------
    vec : array_like
        A 3-element direction vector with (ML, AP, DV) components in RAS.
    degrees : bool, optional
        If True, return angles in degrees; otherwise radians (default True).
    invert_rx : bool, optional
        If True, apply the AIND non-right-handed sign convention to rx
        (default True). See module docstring.

    Returns
    -------
    tuple of float or None
        ``(rx, ry)`` — the arc angles, in degrees if ``degrees=True`` else
        radians. ``rx`` is the rotation about the x-axis (alias: AP angle,
        pitch). ``ry`` is the rotation about the y-axis (alias: ML angle,
        roll). Returns None if ``vec`` is the zero vector.
    """
    vec = np.asarray(vec)
    if np.linalg.norm(vec) == 0:
        return None
    if np.dot(vec, [0, 0, 1]) < 0:
        vec = -vec
    nv = vec / np.linalg.norm(vec)
    # Invert arc_angles_to_rotation by decomposing, through scipy, the rotation
    # that carries the probe's neutral axis [0, 0, 1] onto nv. The spin (rz) is
    # unconstrained by a single vector, but rx/ry are fixed by the image of
    # [0, 0, 1] alone, so the same Euler sequence is the single source of truth
    # for both directions and they cannot drift apart.
    R, _ = Rotation.align_vectors([nv], [[0.0, 0.0, 1.0]])
    rx, ry, _ = R.as_euler(_ARC_EULER_SEQ, degrees=degrees)
    if invert_rx:
        rx = -rx
    return float(rx), float(ry)


def arc_angles_to_vector(
    rx: float,
    ry: float,
    degrees: bool = True,
    invert_rx: bool = True,
) -> NDArray[np.floating[Any]]:
    """Calculate a direction vector from arc angles (rx, ry).

    Parameters
    ----------
    rx : float
        Rotation about the x-axis (which is the ML axis in RAS).
        Aliases: "AP angle", "AP tilt", pitch — tilts the probe in the AP
        plane.
    ry : float
        Rotation about the y-axis (which is the AP axis in RAS).
        Aliases: "ML angle", "ML tilt", roll — tilts the probe in the ML
        plane.
    degrees : bool, optional
        If True, ``rx`` and ``ry`` are interpreted as degrees; otherwise
        radians (default True).
    invert_rx : bool, optional
        If True, apply the AIND non-right-handed sign convention to rx
        (default True). See module docstring.

    Returns
    -------
    numpy.ndarray
        A unit 3-vector with (ML, AP, DV) components in RAS.
    """
    R = arc_angles_to_rotation(rx, ry, degrees=degrees, invert_rx=invert_rx)
    # The probe's neutral axis is straight up (+DV); its image under the
    # arc rotation is the probe direction.
    vec = R.apply([0.0, 0.0, 1.0])
    return np.asarray(vec / np.linalg.norm(vec), dtype=float)


def vector_to_stereotax_angles(
    vec: NDArray[np.floating[Any]],
    degrees: bool = True,
    zero_rz_to_left: bool = False,
) -> tuple[float, float] | None:
    """Calculate stereotaxic angles (ry, rz) for a given direction vector.

    Used for the Kopf 1500 off-plane insertion tool, which parameterizes a
    probe direction by a polar angle from vertical (``ry``) and an azimuthal
    spin (``rz``).

    Parameters
    ----------
    vec : array_like
        A 3-element direction vector with (ML, AP, DV) components in RAS.
    degrees : bool, optional
        If True, return angles in degrees; otherwise radians (default True).
    zero_rz_to_left : bool, optional
        If True, the zero of ``rz`` points to the subject's left; otherwise
        it points to the right (default False).

    Returns
    -------
    tuple of float or None
        ``(ry, rz)`` — the stereotaxic angles, in degrees if ``degrees=True``
        else radians.  ``ry`` is the polar tilt from vertical (rotation about
        the y/AP axis), zero = vertical.  ``rz`` is the azimuthal spin
        (rotation about the z/DV axis); zero points right unless
        ``zero_rz_to_left=True``. Returns None if ``vec`` is the zero vector.
    """
    vec = np.asarray(vec)
    if np.linalg.norm(vec) == 0:
        return None
    if np.dot(vec, [0, 0, 1]) < 0:
        vec = -vec
    nv = vec / np.linalg.norm(vec)
    ry = np.arccos(nv[2])
    rz = np.arctan2(nv[1], nv[0])
    if zero_rz_to_left:
        # Flip rz by 180° (wrapped to [-π, π)) so that the zero direction
        # points to the subject's left instead of the right.
        rz = (rz + 2 * np.pi) % (2 * np.pi) - np.pi
    if degrees:
        ry = math.degrees(ry)
        rz = math.degrees(rz)
    return ry, rz


def stereotax_angles_to_vector(
    ry: float, rz: float, degrees: bool = True, zero_rz_to_left: bool = False
) -> NDArray[np.floating[Any]]:
    """Calculate a direction vector from stereotaxic angles (ry, rz).

    Used for the Kopf 1500 off-plane insertion tool.

    Parameters
    ----------
    ry : float
        Polar tilt from vertical (rotation about the y/AP axis).
    rz : float
        Azimuthal spin (rotation about the z/DV axis); zero points right
        unless ``zero_rz_to_left=True``.
    degrees : bool, optional
        If True, ``ry`` and ``rz`` are interpreted as degrees; otherwise
        radians (default True).
    zero_rz_to_left : bool, optional
        If True, the zero of ``rz`` is taken to point to the subject's left;
        otherwise to the right (default False).

    Returns
    -------
    numpy.ndarray
        A unit 3-vector with (ML, AP, DV) components in RAS.
    """
    if degrees:
        ry = math.radians(ry)
        rz = math.radians(rz)
    if zero_rz_to_left:
        # Flip rz by 180° (wrapped to [-π, π)) to convert from the
        # zero-points-left convention back to zero-points-right.
        rz = (rz + 2 * np.pi) % (2 * np.pi) - np.pi

    vec = np.array(
        [
            np.cos(rz) * np.sin(ry),  # ML component
            np.sin(rz) * np.sin(ry),  # AP component using trig
            np.cos(ry),  # DV component
        ]
    )
    return np.asarray(vec / np.linalg.norm(vec), dtype=float)


def earbar_angles_to_rotation_matrix(
    earbar_pitch: float = 0.0,
    earbar_roll: float = 0.0,
    degrees: bool = True,
) -> NDArray[np.floating[Any]]:
    """Build a rotation matrix from earbar pitch and roll angles.

    Applies pitch (rotation about the x/ML axis, "nose up" positive) followed
    by roll (rotation about the y/AP axis, "left side up" positive), in
    intrinsic XYZ Euler order. The returned matrix maps a vector expressed in
    the head/MRI-bregma RAS frame into the stereotax RAS frame, given the
    head's earbar tilt.

    Parameters
    ----------
    earbar_pitch : float, optional
        Pitch about the x/ML axis. Positive = nose up. Default 0.
    earbar_roll : float, optional
        Roll about the y/AP axis. Positive = left side up. Default 0.
    degrees : bool, optional
        If True, input angles are in degrees; otherwise radians (default
        True).

    Returns
    -------
    numpy.ndarray
        A 3x3 rotation matrix in RAS that takes head/bregma-frame vectors
        into the stereotax frame.
    """
    return Rotation.from_euler("XYZ", [earbar_pitch, earbar_roll, 0], degrees=degrees).as_matrix()


def arc_angles_to_stereotax_angles(
    rx: float,
    ry: float,
    degrees: bool = True,
    invert_rx: bool = True,
    zero_rz_to_left: bool = False,
    earbar_pitch: float = 0.0,
    earbar_roll: float = 0.0,
    headframe_rx_in_arc_system: float | None = None,
) -> tuple[float, float]:
    """Convert ephys-rig arc angles to Kopf 1500 off-plane stereotaxic angles for a matched insertion.

    Use case
    --------
    You have an arc reading (rx, ry) for a planned insertion on the ephys
    rig, and you want to make the same physical insertion — same probe
    direction relative to the lambda–bregma plane — on the off-plane
    stereotax. The off-plane setup uses an earbar gimbal to re-tilt the
    head; you choose the earbar pose (often to make the insertion less
    steep), and this function returns the off-plane tool angles
    ``(ry_st, rz_st)`` that complete the match.

    Pipeline
    --------
    1. **Arc → headframe frame.** Subtract ``headframe_rx_in_arc_system``
       (default 14) from ``rx`` to undo the fixed −14° headframe pitch
       imposed by the ephys rig. After this, (rx, ry) describe the probe
       direction in the headframe (lambda–bregma) frame.
    2. **Headframe angles → vector.** Convert via
       :func:`arc_angles_to_vector`. The result is a unit vector in the
       headframe RAS frame.
    3. **Headframe → stereotax frame.** If earbar pitch/roll are nonzero,
       apply :func:`earbar_angles_to_rotation_matrix` to rotate the vector
       into the off-plane stereotax frame. (At earbar (0, 0), the
       headframe frame coincides with the stereotax frame and this is a
       no-op.)
    4. **Stereotax vector → off-plane angles.** Convert via
       :func:`vector_to_stereotax_angles`.

    The invariant: the probe direction expressed in the headframe frame is
    identical for the ephys rig and the off-plane setup. The earbar tilt
    redistributes that fixed headframe-frame vector between the off-plane
    arm's polar tilt (``ry_st``) and any residual the user accepts.

    Parameters
    ----------
    rx : float
        Arc rx in the ephys-rig frame (i.e., what the arc reads).
        Conceptually ``plan_rx + headframe_rx_in_arc_system``.
        Alias: AP angle, pitch.
    ry : float
        Arc ry in the ephys-rig frame. Equal to plan ry (no offset).
        Alias: ML angle, roll.
    degrees : bool, optional
        If True, all input and output angles (including ``earbar_pitch``,
        ``earbar_roll``, and ``headframe_rx_in_arc_system``) are in
        degrees; otherwise radians (default True).
    invert_rx : bool, optional
        If True, apply the AIND non-right-handed sign convention to rx
        (default True). See module docstring.
    zero_rz_to_left : bool, optional
        If True, the zero of the returned ``rz_st`` points left; otherwise
        right (default False).
    earbar_pitch : float, optional
        Earbar pitch about the x/ML axis (rotation that takes headframe-
        frame vectors into stereotax-frame vectors). Positive = nose up.
        Default 0. Units follow ``degrees``.
    earbar_roll : float, optional
        Earbar roll about the y/AP axis. Positive = left side up.
        Default 0. Units follow ``degrees``.
    headframe_rx_in_arc_system : float, optional
        The arc rx reading at which a probe vertical in the headframe
        (perpendicular to lambda–bregma) appears in the ephys rig. The
        headframe is pitched down by this amount when mounted in the arc,
        so plan_rx and arc_rx are related by
        ``arc_rx = plan_rx + headframe_rx_in_arc_system``.
        Default 14. Units follow ``degrees`` — if you pass
        ``degrees=False``, you must convert this value to radians too.

    Returns
    -------
    tuple of float
        ``(ry_st, rz_st)`` — the Kopf 1500 off-plane angles, in degrees if
        ``degrees=True`` else radians. ``ry_st`` is the polar tilt from
        stereotax-vertical; ``rz_st`` is the azimuthal spin.
    """
    if headframe_rx_in_arc_system is None:
        headframe_rx_in_arc_system = 14 if degrees else math.radians(14)
    rx -= headframe_rx_in_arc_system
    vec = arc_angles_to_vector(rx, ry, degrees=degrees, invert_rx=invert_rx)
    if earbar_pitch != 0.0 or earbar_roll != 0.0:
        R = earbar_angles_to_rotation_matrix(earbar_pitch, earbar_roll, degrees=degrees)
        vec = R @ vec
    # arc_angles_to_vector always returns a unit vector, so the conversion
    # below cannot return None.
    result = vector_to_stereotax_angles(vec, degrees=degrees, zero_rz_to_left=zero_rz_to_left)
    assert result is not None
    return result


@dataclass(frozen=True)
class EarbarStereotaxSolution:
    """Earbar pose and Kopf off-plane angles for a matched insertion.

    Angles are in degrees if the producing call used ``degrees=True`` (the
    default), otherwise radians.

    Attributes
    ----------
    earbar_pitch : float
        Chosen earbar pitch about the x/ML axis ("nose up" positive).
    earbar_roll : float
        Chosen earbar roll about the y/AP axis ("left side up" positive).
    kopf_ry : float
        Kopf 1500 off-plane polar tilt from stereotax-vertical. This is the
        residual tilt the earbar could not absorb, and is what the solver
        minimizes.
    kopf_rz : float
        Kopf 1500 off-plane azimuthal spin.
    vertical_achievable : bool
        True if the earbar (within its bounds) could bring the insertion
        essentially vertical (``kopf_ry`` ~ 0). False means a bound was
        binding and the Kopf tool must take up the remaining ``kopf_ry``.
    """

    earbar_pitch: float
    earbar_roll: float
    kopf_ry: float
    kopf_rz: float
    vertical_achievable: bool


def _earbar_pose_minimizing_tilt(
    v_head: NDArray[np.floating[Any]],
    pitch_bounds: tuple[float, float],
    roll_bounds: tuple[float, float],
) -> tuple[float, float, float]:
    """Search the earbar (pitch, roll) box for the pose closest to vertical.

    Maximizes the absolute DV component of ``R_eb @ v_head`` (equivalently,
    minimizes the polar tilt) by successively refined grid search. Bounds are
    in radians; the returned pose is in radians.

    Parameters
    ----------
    v_head : numpy.ndarray
        Unit insertion direction (ML, AP, DV) in the head/bregma RAS frame.
    pitch_bounds, roll_bounds : tuple of float
        ``(low, high)`` earbar pitch / roll limits, in radians.

    Returns
    -------
    tuple of float
        ``(pitch_rad, roll_rad, abs_dv)`` — the best earbar pose and the
        absolute DV component it achieves (1.0 = perfectly vertical).
    """
    p_lo0, p_hi0 = pitch_bounds
    r_lo0, r_hi0 = roll_bounds
    p_lo, p_hi, r_lo, r_hi = p_lo0, p_hi0, r_lo0, r_hi0
    best_p, best_r, best_z = 0.0, 0.0, -1.0
    n = 25
    for _ in range(3):
        for pp in np.linspace(p_lo, p_hi, n):
            for rr in np.linspace(r_lo, r_hi, n):
                R = earbar_angles_to_rotation_matrix(float(pp), float(rr), degrees=False)
                z = abs(float((R @ v_head)[2]))
                if z > best_z:
                    best_z, best_p, best_r = z, float(pp), float(rr)
        dp = (p_hi - p_lo) / (n - 1)
        dr = (r_hi - r_lo) / (n - 1)
        p_lo, p_hi = max(p_lo0, best_p - dp), min(p_hi0, best_p + dp)
        r_lo, r_hi = max(r_lo0, best_r - dr), min(r_hi0, best_r + dr)
    return best_p, best_r, best_z


def solve_earbar_for_vertical(
    v_head: NDArray[np.floating[Any]],
    *,
    pitch_bounds: tuple[float, float] = (-10.0, 10.0),
    roll_bounds: tuple[float, float] = (-15.0, 15.0),
    degrees: bool = True,
    zero_rz_to_left: bool = False,
) -> EarbarStereotaxSolution:
    """Find the earbar pose bringing a head insertion vector closest to vertical.

    Given an insertion direction expressed relative to the head (the
    head/bregma RAS frame), choose the earbar gimbal pose — within its
    mechanical limits — that makes the insertion as close to stereotax-vertical
    as possible, then report the Kopf 1500 off-plane angles for the residual.
    The earbar absorbs as much tilt as its bounds allow; whatever remains
    becomes the Kopf polar tilt ``kopf_ry``.

    The earbar has two degrees of freedom, applied as intrinsic rotations first
    about the x/ML axis (pitch) then the y/AP axis (roll); see
    :func:`earbar_angles_to_rotation_matrix`. Two DOF can null any tilt unless a
    bound binds, so ``vertical_achievable`` is True whenever no bound is active.

    Parameters
    ----------
    v_head : numpy.ndarray
        Insertion direction with (ML, AP, DV) components in the head/bregma RAS
        frame. Need not be unit length; only its direction matters.
    pitch_bounds : tuple of float, optional
        ``(low, high)`` earbar pitch (about x/ML) limits. Default the safe
        ``(-10, 10)``; the mechanical hard limit is ``(-12, 12)``. Units follow
        ``degrees``.
    roll_bounds : tuple of float, optional
        ``(low, high)`` earbar roll (about y/AP) limits. Default the safe
        ``(-15, 15)``; the mechanical hard limit is ``(-30, 30)``. Units follow
        ``degrees``.
    degrees : bool, optional
        If True, all input bounds and output angles are in degrees; otherwise
        radians (default True).
    zero_rz_to_left : bool, optional
        If True, the zero of the returned ``kopf_rz`` points to the subject's
        left; otherwise right (default False).

    Returns
    -------
    EarbarStereotaxSolution
        The chosen earbar pose, the resulting Kopf off-plane angles, and
        whether vertical was reachable.

    Raises
    ------
    ValueError
        If ``v_head`` is the zero vector.
    """
    v = np.asarray(v_head, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("v_head must be a nonzero direction vector")
    v = v / norm

    if degrees:
        p_bounds = (math.radians(pitch_bounds[0]), math.radians(pitch_bounds[1]))
        r_bounds = (math.radians(roll_bounds[0]), math.radians(roll_bounds[1]))
    else:
        p_bounds = (float(pitch_bounds[0]), float(pitch_bounds[1]))
        r_bounds = (float(roll_bounds[0]), float(roll_bounds[1]))
    p_bounds = (min(p_bounds), max(p_bounds))
    r_bounds = (min(r_bounds), max(r_bounds))

    pitch_rad, roll_rad, abs_dv = _earbar_pose_minimizing_tilt(v, p_bounds, r_bounds)

    R = earbar_angles_to_rotation_matrix(pitch_rad, roll_rad, degrees=False)
    kopf = vector_to_stereotax_angles(R @ v, degrees=degrees, zero_rz_to_left=zero_rz_to_left)
    assert kopf is not None  # R @ v is a unit vector
    kopf_ry, kopf_rz = kopf

    vertical_achievable = math.acos(min(1.0, abs_dv)) <= _VERTICAL_TILT_TOL_RAD
    if degrees:
        pitch_out, roll_out = math.degrees(pitch_rad), math.degrees(roll_rad)
    else:
        pitch_out, roll_out = pitch_rad, roll_rad

    return EarbarStereotaxSolution(
        earbar_pitch=pitch_out,
        earbar_roll=roll_out,
        kopf_ry=kopf_ry,
        kopf_rz=kopf_rz,
        vertical_achievable=vertical_achievable,
    )


def arc_angles_to_earbar_stereotax(
    rx: float,
    ry: float,
    *,
    pitch_bounds: tuple[float, float] = (-10.0, 10.0),
    roll_bounds: tuple[float, float] = (-15.0, 15.0),
    invert_rx: bool = True,
    degrees: bool = True,
    zero_rz_to_left: bool = False,
    headframe_rx_in_arc_system: float | None = None,
) -> EarbarStereotaxSolution:
    """Solve for the earbar pose and Kopf angles matching an ephys-rig arc reading.

    Convenience wrapper over :func:`solve_earbar_for_vertical`: takes an arc
    reading ``(rx, ry)`` in the ephys-rig frame, removes the fixed headframe
    pitch to recover the head-relative insertion vector (as in
    :func:`arc_angles_to_stereotax_angles`), then solves for the earbar pose
    that brings the off-plane insertion closest to vertical.

    Parameters
    ----------
    rx : float
        Arc rx in the ephys-rig frame (what the arc reads); conceptually
        ``plan_rx + headframe_rx_in_arc_system``. Alias: AP angle, pitch.
    ry : float
        Arc ry in the ephys-rig frame (equal to plan ry). Alias: ML angle.
    pitch_bounds : tuple of float, optional
        Earbar pitch limits. Default the safe ``(-10, 10)`` (hard ``(-12, 12)``).
        Units follow ``degrees``.
    roll_bounds : tuple of float, optional
        Earbar roll limits. Default the safe ``(-15, 15)`` (hard ``(-30, 30)``).
        Units follow ``degrees``.
    invert_rx : bool, optional
        If True, apply the AIND non-right-handed sign convention to rx
        (default True). See module docstring.
    degrees : bool, optional
        If True, all input and output angles are in degrees; otherwise radians
        (default True).
    zero_rz_to_left : bool, optional
        If True, the zero of the returned ``kopf_rz`` points left; otherwise
        right (default False).
    headframe_rx_in_arc_system : float, optional
        Arc rx at which a probe vertical in the headframe appears on the ephys
        rig (``arc_rx = plan_rx + headframe_rx_in_arc_system``). Default 14.
        Units follow ``degrees``.

    Returns
    -------
    EarbarStereotaxSolution
        The chosen earbar pose, resulting Kopf off-plane angles, and whether
        vertical was reachable.
    """
    if headframe_rx_in_arc_system is None:
        headframe_rx_in_arc_system = 14 if degrees else math.radians(14)
    plan_rx = rx - headframe_rx_in_arc_system
    v_head = arc_angles_to_vector(plan_rx, ry, degrees=degrees, invert_rx=invert_rx)
    return solve_earbar_for_vertical(
        v_head,
        pitch_bounds=pitch_bounds,
        roll_bounds=roll_bounds,
        degrees=degrees,
        zero_rz_to_left=zero_rz_to_left,
    )


def arc_angles_to_affine(
    rx: float,
    ry: float,
    rz: float = 0.0,
    invert_rx: bool = True,
    invert_rz: bool = True,
) -> NDArray[np.floating[Any]]:
    """Build an LPS rotation matrix from arc angles (rx, ry, rz).

    Parameters
    ----------
    rx : float
        Rotation about the x/ML axis, in degrees. Alias: AP angle, pitch.
    ry : float
        Rotation about the y/AP axis, in degrees. Alias: ML angle, roll.
    rz : float, optional
        Rotation about the z/DV axis, in degrees. Alias: spin, yaw.
        Default 0.
    invert_rx : bool, optional
        If True, apply the AIND non-right-handed sign convention to rx
        (default True). See module docstring.
    invert_rz : bool, optional
        If True, apply the AIND non-right-handed sign convention to rz
        (default True).

    Returns
    -------
    numpy.ndarray
        A 3x3 rotation matrix in **LPS** (not RAS — converted via
        ``ras_to_lps_transform`` for compatibility with ITK/Slicer
        transforms).

    Notes
    -----
    Builds the RAS rotation via :func:`arc_angles_to_rotation` (intrinsic XYZ
    Euler rotations after the AIND sign-flips) and converts it RAS → LPS.
    """
    R = arc_angles_to_rotation(rx, ry, rz, degrees=True, invert_rx=invert_rx, invert_rz=invert_rz).as_matrix()
    return ras_to_lps_transform(R)[0]
