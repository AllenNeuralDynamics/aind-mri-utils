""" Module to deal with coordinate systems
"""

import numpy as np


def find_coordinate_perm_and_flips(src: str, dst: str):  # noqa: C901
    """Determine how to convert between coordinate systems

    This function takes a source `src` and destination `dst` string specifying
    two coordinate systems, and finds the permutation and sign flip such that a
    source array can be transformed from its own coordinate system to the
    destination coordinate system by first applying the permutation and then
    multiplying the resulting array by the direction. That is, the input can
    be transformed to the desired coordinate system with the following code:
    `dst_array = direction * src_array[:, perm]`, where `direction` and `perm`
    are the returned values of this function.

    Coordinate system are defined by strings specifying how each axis aligns to
    anatomical directions, with each character belonging to the set 'APLRIS',
    corresponding to Anterior, Posterior, Left, Right, Inferior, Superior.

    An example string would be 'RAS' corresponding to Right, Anterior, Superior
    for the first, second, and third axes respectively. The axis increases in
    the direction indicated (i.e. 'R' means values are more positive as you
    move to the patient's right).

    Parameters
    ---------
    src: str
        String specifying a coordinate system, as described above
    dst: str
        String specifying a coordinate system, as described above

    Returns
    -------
    perm : np.ndarray(dtype=int16) (N)
        Permutation array used to convert the `src` coordinate system to the
        `dst` coordinate system
    direction: np.ndarray(dtype=int16) (N)
        Direction array used to multiply the `src` coordinate system after
        permutation into the `dst` coordinate system
    """
    nel = len(src)
    if len(dst) != nel:
        raise ValueError("Inputs should be the same length")
    src_u = src.upper()
    dst_u = dst.upper()
    basic_pairs = dict(R="L", A="P", S="I")
    pairs = dict()
    for k, v in basic_pairs.items():
        pairs[k] = v
        pairs[v] = k
    src_order = dict()
    for i, s in enumerate(src_u):
        if s not in pairs:
            raise ValueError(
                "Source direction '{}' not in R/L, A/P, or I/S".format(s)
            )
        if s in src_order or pairs[s] in src_order:
            raise ValueError("Source axis '{}' not unique".format(s))
        src_order[s] = i
    perm = -1 * np.ones(nel, dtype="int16")
    direction = np.zeros(nel, dtype="int16")
    dst_set = set()
    for i, d in enumerate(dst_u):
        if d not in pairs:
            raise ValueError(
                "Destination direction '{}' not in R/L, A/P, or I/S".format(d)
            )
        if d in dst_set or pairs[d] in dst_set:
            raise ValueError("Destination axis '{}' not unique".format(d))
        if d in src_order:
            perm[i] = src_order[d]
            direction[i] = 1
        elif pairs[d] in src_order:
            perm[i] = src_order[pairs[d]]
            direction[i] = -1
        else:
            raise ValueError(
                (
                    "Destination direction '{}' has "
                    + "no match in source directions '{}'"
                ).format(d, src_u)
            )
        dst_set.add(d)
    return perm, direction


def convert_coordinate_system(
    arr: np.ndarray, src_coord: str, dst_coord: str
) -> np.ndarray:
    """Converts points in one anatomical coordinate system to another

    This will permute and multiply the NxM input array `arr` so that N
    M-dimensional points in the coordinate system specified by `src_coord` will
    be transformed into the destination coordinate system specified by
    `dst_coord`. The current implementation does not allow the dimensions to
    change.

    Coordinate systems are defined by strings specifying how each axis aligns
    to anatomical directions, with each character belonging to the set
    'APLRIS', corresponding to Anterior, Posterior, Left, Right, Inferior,
    Superior, respectively.

    An example string would be 'RAS' corresponding to Right, Anterior, Superior
    for the first, second, and third axes respectively. The axis increases in
    the direction indicated (i.e. 'R' means values are more positive as you
    move to the patient's right).

    Parameters
    ----------
    arr : np.ndarray (N x M)
        N points of M dimensions (at most three)
    src_coord: str
        String specifying a coordinate system, as described above
    dst_coord: str
        String specifying a coordinate system, as described above

    Returns
    -------
    out : np.ndarray (N x M)
        The N input points transformed into the destination coordinate system
    """
    perm, direction = find_coordinate_perm_and_flips(src_coord, dst_coord)
    if arr.ndim == 1:
        out = arr[perm]
    else:
        out = arr[:, perm]
    out *= direction
    return out
