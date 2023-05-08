from typing import Tuple, Union
import math
import jax.numpy as jnp
from jax import jit,tree_util,lax
from matplotlib import pyplot as plt
from constants import DEFAULT_WALL_GAMMA, DEFAULT_WALL_ROTATIONAL_GAMMA
from constants import DEFAULT_BOX_SIZE, PROJECT_DIR, BOUNDING_BOX_VERTICES


class ConvexPolygon:
    '''
    A convex polygon, defined by a set of vertices. 
    Note that this is a named tuple to make it JAX
    compatible. Call `vertices_to_polygon` to get 
    proper constructor functionality. 
    '''
    def __init__(self, normals_0: jnp.ndarray, vertices_0: jnp.ndarray, pos_gamma: float, rot_gamma: float) -> None:
        self.normals_0 = normals_0
        self.vertices_0 = vertices_0
        self.pos_gamma = pos_gamma
        self.rot_gamma = rot_gamma
        pass
    
    @jit
    def get_vertices_normals_proj_jax(self, centroid: jnp.ndarray, angle: float) -> Tuple[jnp.array, jnp.array, jnp.array]:
        '''
        Returns a tuple of arrays: (vertices, normals, projections) where vertices 
        contains the current positions of all the vertices, normals contains the current 
        set of properly oriented normals, and projections contains the projected location 
        of each edge onto its respective normal.
        '''
        cos_theta = jnp.cos(angle)
        sin_theta = jnp.sin(angle)
        rotation_matrix = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        vertices = self.vertices_0 @ rotation_matrix + centroid
        normals = self.normals_0 @ rotation_matrix
        projections = jnp.einsum('ij,ij->i', vertices, normals)
        
        return vertices, normals, projections
    
    @jit
    def get_vertices(self, centroid: jnp.ndarray, angle: float) -> jnp.ndarray:
        cos_theta = jnp.cos(angle)
        sin_theta = jnp.sin(angle)
        rotation_matrix = jnp.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        return self.vertices_0 @ rotation_matrix + centroid

    
    @jit
    def get_min_particle_push_vector(self, centroid: jnp.ndarray, angle: float, positions: jnp.ndarray) -> jnp.ndarray:
        _, normals, polygon_projections = self.get_vertices_normals_proj_jax(centroid, angle)
        bacteria_projection = positions @ normals.transpose() #(n,2) x (2, v) -> (n,v) Array
        only_negative_projections = lax.clamp(-float("inf"),bacteria_projection - polygon_projections,0.) # (n,v) Array
        
        mpv_indx = jnp.argmax(only_negative_projections, axis=1) # (n,) Array
        max_mpv = jnp.max(only_negative_projections, axis=1) #(n, Array); I know this is inefficient but can't get the better way to jax

        return -max_mpv[:, None] * normals[mpv_indx]

    @jit
    def get_rotation_from_wall_particle_interaction(self, 
            centroid: jnp.ndarray, 
            r: jnp.ndarray, 
            poly_correction_to_particles: jnp.ndarray, 
            particle_gamma: float
            ) -> float:
        r"""
        We want to compute the angle `\theta` by which the polygon rotates
        as a result of all the forces it experiences. Each of these forces is of
        the form `fi dt = -c_i \gamma_i` where `c_i` is the amount by which the 
        polygon has moved particle i in the last timestep.
        
        Relative to the center of mass of the walls r_com, this yields a torque

        `\tau_i dt = r x fi = r[0] * fi[1] - r[1] * fi[0]`

        (implicitly, in the z direction, but everything is in the xy plane, so
        we can treat this as a scalar). We assume that this translates directly
        to a rotational velocity by a rotational drag `\gamma_R`, so 

        `\Delta \theta = \sum_i \tau_i dt / \gamma_R`

        Parameters
        ----------
        centroid: jnp.ndarray
            (2) Array specifying current COM of the polygon.
        r: jnp.ndarray
            (n,2) Array specifying the current positions of each particle. 
        poly_correction_to_particles: jnp.ndarray
            (n,2) Array specifying our polygon's modificaiton to particles' position 
            changes. polygon_correction_to_particles[i,0] = `c_i`. 
        particle_gamma: float
            Specifies the fluid drag coefficient of the particles. 
        
        Returns
        ----------
        theta: angle by which the polygon rotates due to the effects of forces
        from the particles on the wall. 
        """

        relative_positions = (r - centroid)
        forces = - poly_correction_to_particles * particle_gamma
        torques_dt = relative_positions[:,0] * forces[:,1] - relative_positions[:,1] * forces[:,0]
        return jnp.sum(torques_dt) / self.rot_gamma
        
    def is_inside(self, centroid: jnp.ndarray, angle: float, r: jnp.ndarray) -> jnp.ndarray:
        _, normals, polygon_projections = self.get_vertices_normals_proj_jax(centroid, angle)
        bacteria_projection = r @ normals.transpose() #(n,2) x (2, v) -> (n,v) Array
        relative_projection = (bacteria_projection - polygon_projections)  # (n,v) Array
        are_all_projections_negative = jnp.all(relative_projection < 0,axis=1) # (n,) Array
        return are_all_projections_negative

    def plot(self):
        plt.figure()
        plt.xlim(-DEFAULT_BOX_SIZE*1.2/2,DEFAULT_BOX_SIZE*1.2/2)
        plt.ylim(-DEFAULT_BOX_SIZE*1.2/2,DEFAULT_BOX_SIZE*1.2/2)
        plt.fill(*self.get_vertices(0.0,0.0).transpose(),linewidth=3,c="r", facecolor="none")
        plt.fill(*BOUNDING_BOX_VERTICES.transpose(), linewidth=3,c="k",facecolor = "none")
        plt.savefig(f"{PROJECT_DIR}/plots/wall_shape.png")
        

tree_util.register_pytree_node(ConvexPolygon, lambda s: ((s.normals_0, s.vertices_0, s.pos_gamma, s.rot_gamma), None), lambda _, xs: ConvexPolygon(xs[0], xs[1], xs[2], xs[3]))

def sort_vertices_ccw(coords: jnp.ndarray) -> jnp.ndarray:
    coords = jnp.array(coords)
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

def is_convex_polygon(coords: jnp.ndarray) -> bool:
    coords = jnp.array(coords)
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

def convex_polygon(vertices: jnp.ndarray, return_centroid: bool = False) -> Union[ConvexPolygon,Tuple[ConvexPolygon,jnp.ndarray]]:
    vertices = jnp.array(vertices)

    sorted_vertices = jnp.array(sort_vertices_ccw(vertices))

    assert is_convex_polygon(sorted_vertices), "vertices_init list does not define a convex polygon"

    normals = []
        
    for i in range(sorted_vertices.shape[0]):
        edge = sorted_vertices[i] - sorted_vertices[i-1]
        normals.append(jnp.array([edge[1], -edge[0]]) / jnp.linalg.norm(edge))
    
    centroid = jnp.mean(sorted_vertices, axis=0)
    
    normals_0 = jnp.array(normals)
    vertices_0 = jnp.array(sorted_vertices - centroid)
    if return_centroid:
        return ConvexPolygon(normals_0, vertices_0, DEFAULT_WALL_GAMMA, DEFAULT_WALL_ROTATIONAL_GAMMA), centroid
    else:
        return ConvexPolygon(normals_0, vertices_0, DEFAULT_WALL_GAMMA, DEFAULT_WALL_ROTATIONAL_GAMMA)