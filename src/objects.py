from typing import List, Tuple, Union
import math
import numpy as np
import jax.numpy as jnp

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

def is_convex_polygon(coords):
    # Check that there are at least three vertices
    if len(coords) < 3:
        return False
    
    # Check that the polygon is convex
    num_vertices = len(coords)
    for i in range(num_vertices):
        p1 = coords[i]
        p2 = coords[(i + 1) % num_vertices]
        p3 = coords[(i + 2) % num_vertices]
        
        cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        
        if cross_product < 0:
            return False
    
    return True


class ConvexPolygon():
    '''
    A convex polygon, defined by a set of vertices
    '''
    def __init__(self, vertices_init) -> None:
        
        sorted_vertices = np.array(sort_vertices_ccw(vertices_init))
        
        assert is_convex_polygon(sorted_vertices), "vertices_init list does not define a convex polygon"
        
        normals = []
        
        for i in range(sorted_vertices.shape[0]):
            edge = sorted_vertices[i] - sorted_vertices[i-1]
            normals.append([edge[1], -edge[0]] / np.linalg.norm([edge[1], -edge[0]]))
        
        centroid = jnp.array(np.sum(sorted_vertices, axis=0) / sorted_vertices.shape[0])
        
        # self.centroid = jnp.array(np.sum(sorted_vertices, axis=0) / sorted_vertices.shape[0])
        # self.angle = 0
        self.normals_0 = jnp.array(normals)
        self.vertices_rel = jnp.array(sorted_vertices - centroid)
        pass

    def get_vertices_normals_proj_jax(self, centroid: jnp.array, angle: float) -> Tuple[jnp.array, jnp.array, jnp.array]:
        '''
        Returns a tuple of arrays: (vertices, normals, projections) where vertices 
        contains the current positions of all the vertices, normals contains the current 
        set of properly oriented normals, and projections contains the projected location 
        of each edge onto its respective normal.
        '''
        cos_theta = jnp.cos(angle)
        sin_theta = jnp.sin(angle)
        rotation_matrix = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        vertices = jnp.dot(self.vertices_rel, rotation_matrix) + centroid
        normals = jnp.dot(self.normals_0, rotation_matrix)
        projections = jnp.einsum('ij,ij->i', vertices, normals)
        
        return vertices, normals, projections
    
    # DO NOT USE ANYTHING BELOW THIS LINE IN JAX

    # def update_position(self, dr):
    #     '''
    #     Updates the centroid, translating by the vector dr. DONT USE THESE IF YOU WANT TO JAX.
    #     '''
    #     self.centroid += dr
    #     pass
    
    # def update_angle(self, dtheta):
    #     '''
    #     Updates the angle, rotating by the angle dtheta. DONT USE THESE IF YOU WANT TO JAX.
    #     '''
    #     self.angle += dtheta
    #     pass
    
    # def get_vertices_normals_proj(self):
    #     '''
    #     Returns a tuple of arrays: (vertices, normals, projections) where vertices 
    #     contains the current positions of all the vertices, normals contains the current 
    #     set of properly oriented normals, and projections contains the projected location 
    #     of each edge onto its respective normal.
    #     '''
    #     cos_theta = jnp.cos(self.angle)
    #     sin_theta = jnp.sin(self.angle)
    #     rotation_matrix = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
    #     vertices = jnp.dot(self.vertices_rel, rotation_matrix) + self.centroid
    #     normals = jnp.dot(self.normals_0, rotation_matrix)
    #     projections = jnp.einsum('ij,ij->i', vertices, normals)
        
    #     return vertices, normals, projections
        
    # def get_vertices(self):
    #     '''
    #     Returns an array of the current location of vertices, appropriately translated
    #     and rotated with respect to self.centroid and self.angle.
    #     '''
    #     cos_theta = jnp.cos(self.angle)
    #     sin_theta = jnp.sin(self.angle)
    #     rotation_matrix = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
    #     vertices = jnp.dot(self.vertices_rel, rotation_matrix) + self.centroid
        
    #     return vertices
    
    # def get_normals(self):
    #     '''
    #     Returns an array of the current normal vectors to the polygon edges, appropriately
    #     rotated by self.angle.
    #     '''
        
    #     cos_theta = jnp.cos(self.angle)
    #     sin_theta = jnp.sin(self.angle)
    #     rotation_matrix = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
    #     normals = jnp.dot(self.normals_0, rotation_matrix)
        
    #     return normals
        
   
        