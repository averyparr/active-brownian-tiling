import jax
import jax.numpy as jnp
from typing import List, Tuple, Union
import math


from objects import ConvexPolygon

def sort_vertices_ccw(coords):
    # Find the centroid of the polygon
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    cx = sum(x) / len(coords)
    cy = sum(y) / len(coords)

    # Calculate the angle between each vertex and the centroid
    angles = []
    for i in range(len(coords)):
        x_diff = coords[i][0] - cx
        y_diff = coords[i][1] - cy
        angle = math.atan2(y_diff, x_diff)
        angles.append(angle)

    # Sort the vertices by their angles
    sorted_vertices = [v for _, v in sorted(zip(angles, coords))]

    return sorted_vertices

def project(vertices: jnp.ndarray, axis: jnp.ndarray) -> Tuple[float, float]:
    """Project a set of vertices onto an axis."""
    projections = jnp.dot(vertices, axis)
    return jnp.min(projections), jnp.max(projections)

def is_separating_axis(axis: jnp.ndarray, o1_v: jnp.ndarray, o2_v: jnp.ndarray) -> bool:
    """Check if the given axis is a separating axis for the two polygons."""
    o1_proj_min, o1_proj_max = project(o1_v, axis)
    o2_proj_min, o2_proj_max = project(o2_v, axis)
    return o1_proj_max < o2_proj_min or o2_proj_max < o1_proj_min


@jax.jit
def collide_ow(o1: ConvexPolygon, c1: jnp.ndarray, a1: float, BB_MAX: float = 1.) -> jnp.array:
    '''
    Returns an array of mpvs, one for each vertex of o1. The returned array is of shape (v, 2)
    '''
    
    v = o1.get_vertices(c1, a1) # (v, 2)
    vs = jnp.sign(v)
    vab = jnp.abs(v)
    
    correction = jax.lax.clamp(0., vab - jnp.array([BB_MAX, BB_MAX]), float("inf")) # positive correction (v, 2)
    mpvs = -vs * correction # corrects signs

    relative_positions = (v - c1)
    forces = mpvs * o1.pos_gamma
    torques_dt = relative_positions[:,0] * forces[:,1] - relative_positions[:,1] * forces[:,0]

    return jnp.sum(mpvs), -jnp.sum(torques_dt) / o1.rot_gamma / 1e1