"""
Tools for computing arc angles, stereotaxic angles, and the conversions
between them.

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
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

from aind_mri_utils.rotations import ras_to_lps_transform


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
    # using trig identity to get the angle from vertical
    rx = -np.arcsin(nv[1])
    ry = np.arctan2(nv[0], nv[2])
    if degrees:
        rx = math.degrees(rx)
        ry = math.degrees(ry)
    if invert_rx:
        rx = -rx
    return rx, ry


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
    if degrees:
        rx = math.radians(rx)
        ry = math.radians(ry)
    if invert_rx:
        rx = -rx

    vec = np.array(
        [
            np.sin(ry) * np.cos(rx),  # ML component
            -np.sin(rx),  # AP component using trig identity
            np.cos(ry) * np.cos(rx),  # DV component
        ]
    )
    return vec / np.linalg.norm(vec)


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
    return vec / np.linalg.norm(vec)


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
    return Rotation.from_euler(
        "XYZ", [earbar_pitch, earbar_roll, 0], degrees=degrees
    ).as_matrix()


def arc_angles_to_stereotax_angles(
    rx: float,
    ry: float,
    degrees: bool = True,
    invert_rx: bool = True,
    zero_rz_to_left: bool = False,
    earbar_pitch: float = 0.0,
    earbar_roll: float = 0.0,
    headframe_rx_in_arc_system: float = 14,
) -> tuple[float, float]:
    """Convert ephys-rig arc angles to Kopf 1500 off-plane stereotaxic
    angles for a matched insertion.

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
    if headframe_rx_in_arc_system != 0:
        rx -= headframe_rx_in_arc_system
    vec = arc_angles_to_vector(rx, ry, degrees=degrees, invert_rx=invert_rx)
    if earbar_pitch != 0.0 or earbar_roll != 0.0:
        R = earbar_angles_to_rotation_matrix(
            earbar_pitch, earbar_roll, degrees=degrees
        )
        vec = R @ vec
    # arc_angles_to_vector always returns a unit vector, so the conversion
    # below cannot return None.
    result = vector_to_stereotax_angles(
        vec, degrees=degrees, zero_rz_to_left=zero_rz_to_left
    )
    assert result is not None
    return result


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
    Composes the angles as intrinsic XYZ Euler rotations
    ``Rotation.from_euler("XYZ", [rx, ry, rz])`` after applying the AIND
    sign-flips. The result is then converted RAS → LPS.
    """
    if invert_rx:
        rx = -rx
    if invert_rz:
        rz = -rz
    euler_angles = np.array([rx, ry, rz])
    R = (
        Rotation.from_euler("XYZ", euler_angles, degrees=True)
        .as_matrix()
        .squeeze()
    )
    return ras_to_lps_transform(R)[0]
