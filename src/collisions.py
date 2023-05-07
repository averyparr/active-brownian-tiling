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
def collide_oo(o1: ConvexPolygon, c1: jnp.ndarray, a1: float, o2: ConvexPolygon, c2: jnp.ndarray, a2: float) -> Tuple[bool, Tuple[jnp.ndarray, jnp.ndarray]]:
    '''
    Returns True and a pair of antiparallel MPVs if object1 and object2 collide.
    Otherwise, return False and [UNDETERMINED]

    o1 and o2 are two ConvexPolygon objects, which we will try to check for collision. 
    c1 and c2 are the centroids of the ConvexPolygons, a1 and a2 are their angles.
    '''
    # Initialize variables to store the minimum overlap and the corresponding separating axis
    min_overlap = float('inf')
    min_axis = None
    
    # returns vertices, normals, projections (of vertices on these normals)
    v1,n1,p1 = o1.get_vertices_normals_proj_jax(c1,a1)
    v2,n2,p2 = o2.get_vertices_normals_proj_jax(c2,a2)

    # produce a sequence of (v_i,v_j) Arrays encoding the 
    # projection of the vertices of polygon i on polygon j's
    # normals. 
    p11 = v1 @ n1.transpose()
    p21 = v2 @ n1.transpose()

    # compute min and max projections over all vertices
    min_p11,max_p11 = jnp.min(p11,axis=0),jnp.max(p11,axis=0)
    min_p21,max_p21 = jnp.min(p21,axis=0),jnp.max(p21,axis=0)

    # We have two intervals [p11min,p11max], [p21min,p21max] and want to 
    # either return False, (_,_) if they don't overlap, or we want to determine
    # the amount of overlap between them. This should be given by 
    # p11max-p21min or p21max-p11min, but can be generally given by taking the min
    # of the two. If either of these quantities is negative, we don't want to
    # correct any of the walls, so we clamp it to 0. 
    iolap_1 = jax.lax.clamp(0.,jnp.minimum(max_p11-min_p21, max_p21-min_p11),float("inf")) # (v1,) Array

    mindex_1 = jnp.argmin(iolap_1)

    # wanted to check something like iolap_1[mindex_1] <= iolap_2[mindex_2]
    # but experimentally it seems that these two are always equal... so we
    # ignore the check and just return

    min_correction_if_moving_only_o1 = iolap_1[mindex_1] * n1[mindex_1]

    return min_correction_if_moving_only_o1/2, -min_correction_if_moving_only_o1/2

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