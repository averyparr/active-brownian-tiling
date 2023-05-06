import jax.numpy as jnp
from typing import List, Tuple, Union

def project(vertices: jnp.ndarray, axis: jnp.ndarray) -> Tuple[float, float]:
    """Project a set of vertices onto an axis."""
    projections = jnp.dot(vertices, axis)
    return jnp.min(projections), jnp.max(projections)

def is_separating_axis(axis: jnp.ndarray, o1_v: jnp.ndarray, o2_v: jnp.ndarray) -> bool:
    """Check if the given axis is a separating axis for the two polygons."""
    o1_proj_min, o1_proj_max = project(o1_v, axis)
    o2_proj_min, o2_proj_max = project(o2_v, axis)
    return o1_proj_max < o2_proj_min or o2_proj_max < o1_proj_min

# def collide_oo(o1_v: List[Tuple[float, float]], o2_v: List[Tuple[float, float]]) -> Tuple[bool, Union[jnp.ndarray, None]]:
#     '''
#     Returns True and the MPV (Minimum Push Vector) if object1 and object2 collide.
#     Otherwise, return False and None.
    
#     o1_v and o2_v are lists of ordered pairs, the vertices of two convex polygons.
#     '''
#     o1_v = jnp.array(o1_v)
#     o2_v = jnp.array(o2_v)

#     for i in range(len(o1_v)):
#         edge = o1_v[i] - o1_v[i - 1]
#         axis = jnp.array([-edge[1], edge[0]])
#         if is_separating_axis(axis, o1_v, o2_v):
#             return False, None

#     for i in range(len(o2_v)):
#         edge = o2_v[i] - o2_v[i - 1]
#         axis = jnp.array([-edge[1], edge[0]])
#         if is_separating_axis(axis, o1_v, o2_v):
#             return False, None

#     return True, jnp.array([0, 0])  # This implementation does not compute the actual MPV.

def collide_oo(o1_v: List[Tuple[float, float]], o2_v: List[Tuple[float, float]]) -> Tuple[bool, Union[jnp.ndarray, None]]:
    '''
    Returns True and the MPV (Minimum Push Vector) if object1 and object2 collide.
    Otherwise, return False and None.

    o1_v and o2_v are lists of ordered pairs, the vertices of two convex polygons.
    '''
    o1_v = jnp.array(o1_v)
    o2_v = jnp.array(o2_v)
    min_overlap = float('inf')
    min_axis = None

    for polygons in [(o1_v, o2_v), (o2_v, o1_v)]:
        for i in range(len(polygons[0])):
            edge = polygons[0][i] - polygons[0][i - 1]
            axis = jnp.array([-edge[1], edge[0]])

            o1_proj_min, o1_proj_max = project(polygons[0], axis)
            o2_proj_min, o2_proj_max = project(polygons[1], axis)

            if o1_proj_max < o2_proj_min or o2_proj_max < o1_proj_min:
                return False, None

            overlap = min(o1_proj_max, o2_proj_max) - max(o1_proj_min, o2_proj_min)
            if overlap < min_overlap:
                min_overlap = overlap
                min_axis = axis

    # Normalize the separating axis and multiply by the overlap to get the MPV
    min_axis_norm = min_axis / jnp.linalg.norm(min_axis)
    mpv = min_axis_norm * min_overlap

    return True, mpv



def collide_ow(o_v: List[Tuple[float, float]], BB_MAX: float = 1) -> Tuple[bool, jnp.ndarray]:
    '''
    Returns True and the (sum of) wall-normal push vector(s) if object is 
    colliding with a wall.
    
    o_v is a list of ordered pairs, the vertices of the polygon. BB_MAX is 
    the max coordinate of the bounding box in the x and y directions. 
    Default assumes a box defined by the corners (-1, -1) and (1, 1).
    '''
    o_v = jnp.array(o_v)
    push_vector = jnp.zeros(2)

    for vertex in o_v:
        if vertex[0] < -BB_MAX:
            push_vector += jnp.array([abs(vertex[0]) - BB_MAX, 0])
        if vertex[0] > BB_MAX:
            push_vector += jnp.array([BB_MAX - vertex[0], 0])
        if vertex[1] < -BB_MAX:
            push_vector += jnp.array([0, abs(vertex[1]) - BB_MAX])
        if vertex[1] > BB_MAX:
            push_vector += jnp.array([0, BB_MAX - vertex[1]])

    is_colliding = jnp.any(push_vector)
    return is_colliding, push_vector