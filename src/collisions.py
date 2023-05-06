import jax.numpy as jnp
from typing import List, Tuple, Union
import math

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

def collide_oo(o1_v: List[Tuple[float, float]], o2_v: List[Tuple[float, float]]) -> Tuple[bool, Union[Tuple[jnp.ndarray, jnp.ndarray], None]]:
    '''
    Returns True and a pair of antiparallel MPVs if object1 and object2 collide.
    Otherwise, return False and None.

    o1_v and o2_v are lists of ordered pairs, the vertices of two convex polygons.
    '''
    # Convert input lists of vertices to JAX arrays
    o1_v = jnp.array(o1_v)
    o2_v = jnp.array(o2_v)

    # Initialize variables to store the minimum overlap and the corresponding separating axis
    min_overlap = float('inf')
    min_axis = None

    # Loop over both polygons (o1_v and o2_v)
    for polygon in (o1_v, o2_v):
        # Loop over the edges of the current polygon
        for i in range(len(polygon)):
            # Calculate the edge vector
            edge = polygon[i] - polygon[i - 1]
            # Calculate the normal vector to the edge (separating axis)
            axis = jnp.array([edge[1], -edge[0]])
            axis = axis / jnp.linalg.norm(axis)

            # Project both polygons onto the separating axis
            o1_proj_min, o1_proj_max = project(o1_v, axis)
            o2_proj_min, o2_proj_max = project(o2_v, axis)

            # Check if the projections are disjoint, which means there is a separating axis
            if o1_proj_max < o2_proj_min or o2_proj_max < o1_proj_min:
                # If there is a separating axis, the polygons do not collide, return False and None
                return False, None

            # Calculate the overlap between the projections on the current axis
            overlap = o1_proj_max - o2_proj_min

            # If the current overlap is smaller than the previously found minimum, update the minimum overlap and axis
            if jnp.abs(overlap) < jnp.abs(min_overlap):
                min_overlap = overlap
                min_axis = axis
                # min_p = p

    # Normalize the separating axis and multiply by the overlap to get the MPV
    # Multiply by p to ensure correct direction of separation
    mpv = min_axis * min_overlap # * p

    # Divide the MPV by 2 and return a pair of antiparallel vectors
    mpv1 = (-mpv / 2)
    mpv2 = (-mpv1)

    # Return True and the pair of antiparallel MPVs if the polygons collide
    return True, (mpv1, mpv2)




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