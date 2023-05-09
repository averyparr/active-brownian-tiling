from functools import partial
from typing import List, Tuple, Union
import math
import jax
import jax.numpy as jnp
from jax import jit,tree_util,lax, vmap
from matplotlib import pyplot as plt
import numpy as np
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
    
    @jit
    def hell_query(self, centroid: jnp.ndarray, angle: float, r: jnp.ndarray, cutoff: float=0.1) -> jnp.ndarray:
        '''
        Returns a (n,) array of 1s and 0s with a 1 in each position where a particle 
        might be crushed by the polygon (close to a wall).
        '''
        
        _, normals, projections = self.get_vertices_normals_proj_jax(centroid, angle)
        
        bacteria_projection = r @ normals.transpose() #(n,2) x (2, v) -> (n,v) Array
        only_negative_projections = lax.clamp(-float("inf"),bacteria_projection - (projections + cutoff),0.) # (n,v) Array
        
        max_mpv = jnp.max(only_negative_projections, axis=1) #(n, Array); I know this is inefficient but can't get the better way to jax
        
        hell_q = jnp.heaviside(-max_mpv, 0.) # (n,) Array
        
        return hell_q
        
    def is_inside(self, centroid: jnp.ndarray, angle: float, r: jnp.ndarray) -> jnp.ndarray:
        _, normals, polygon_projections = self.get_vertices_normals_proj_jax(centroid, angle)
        bacteria_projection = r @ normals.transpose() #(n,2) x (2, v) -> (n,v) Array
        relative_projection = (bacteria_projection - polygon_projections)  # (n,v) Array
        are_all_projections_negative = jnp.all(relative_projection < 0,axis=1) # (n,) Array
        return are_all_projections_negative

    def plot(self):
        plt.xlim(-DEFAULT_BOX_SIZE*1.2/2,DEFAULT_BOX_SIZE*1.2/2)
        plt.ylim(-DEFAULT_BOX_SIZE*1.2/2,DEFAULT_BOX_SIZE*1.2/2)
        plt.fill(*self.get_vertices(0.0,0.0).transpose(),linewidth=1,c="r", facecolor="none")
        plt.fill(*BOUNDING_BOX_VERTICES.transpose(), linewidth=1,c="k",facecolor = "none")
        plt.savefig(f"{PROJECT_DIR}/plots/wall_shape.png")
        

tree_util.register_pytree_node(ConvexPolygon, lambda s: ((s.normals_0, s.vertices_0, s.pos_gamma, s.rot_gamma), None), lambda _, xs: ConvexPolygon(xs[0], xs[1], xs[2], xs[3]))

def get_angles(coords):
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
    return angles

def sort_vertices_ccw(coords: jnp.ndarray) -> jnp.ndarray:
    coords = jnp.array(coords)
    
    angles = get_angles(coords)

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

    sorted_vertices = vertices

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

@jit
def collide_oo(o1: ConvexPolygon, c1: jnp.ndarray, a1: float, o2: ConvexPolygon, c2: jnp.ndarray, a2: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    '''
    Returns a pair of antiparallel MPVs if object1 and object2 collide.
    Otherwise, return zeros in the same shape.

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

class GluedPolygons:
    def __init__(self, polygon_list, relative_centroids: float, pos_gamma: float, rot_gamma: float) -> None:
        self.polygon_list = polygon_list
        self.relative_centroids = relative_centroids
        self.pos_gamma = pos_gamma
        self.rot_gamma = rot_gamma
    
    @jit
    def get_vertices(self, com: jnp.ndarray, angle: float) -> jnp.ndarray:
        return jnp.concatenate([poly.get_vertices(com+sub_com,angle) for poly,sub_com in zip(self.polygon_list,self.get_relative_centroids(angle))])
    
    @jit
    def get_min_particle_push_vector(self, centroid: jnp.ndarray, angle: float, positions: jnp.ndarray) -> jnp.ndarray:
        return sum(
            poly.get_min_particle_push_vector(centroid+sub_com,angle,positions) 
            for poly,sub_com in zip(self.polygon_list,self.get_relative_centroids(angle)))

    @jit
    def get_rotation_from_wall_particle_interaction(self, 
            centroid: jnp.ndarray, 
            angle: float,
            r: jnp.ndarray, 
            poly_correction_to_particles: jnp.ndarray, 
            particle_gamma: float
            ) -> float:
        return sum(
            poly.get_rotation_from_wall_particle_interaction(centroid+sub_com,r,poly_correction_to_particles,particle_gamma) 
            for poly,sub_com in zip(self.polygon_list, self.get_relative_centroids(angle))
            )/len(self.polygon_list)
    @jit
    def get_relative_centroids(self, angle: float) -> jnp.ndarray:
        rotation_matrix = jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]])
        return self.relative_centroids @ rotation_matrix
    
    @jit
    def collide_oo_wrapper(self, c1: jnp.ndarray, a1: float, other: ConvexPolygon, c2: jnp.ndarray, a2: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        self_mpv = jnp.zeros_like(c1)
        other_mpv = jnp.zeros_like(c2)
        for self_poly,self_com in zip(self.polygon_list,self.get_relative_centroids(a1)):
            for other_poly,other_com in zip(other.polygon_list,other.get_relative_centroids(a2)):
                self_mpv_incremental, other_mpv_incremental = collide_oo(self_poly,c1+self_com,a1,other_poly,c2+other_com,a2)
                self_mpv += self_mpv_incremental/len(other.polygon_list)/len(self.polygon_list)
                other_mpv += other_mpv_incremental/len(other.polygon_list)/len(self.polygon_list)
        return self_mpv, other_mpv
    
    @jit
    def hell_query(self, centroid: jnp.ndarray, angle: float, r: jnp.ndarray, cutoff: float=0.1) -> jnp.ndarray:
        '''
        Returns a (n,) array of 1s and 0s with a 1 in each position where a particle 
        might be crushed by the polygon (close to a wall).
        '''
        return jnp.heaviside(sum(
            poly.hell_query(centroid+sub_com,angle,r) 
            for poly,sub_com in zip(self.polygon_list,self.get_relative_centroids(angle))), 0)


tree_util.register_pytree_node(GluedPolygons, lambda s: ((s.polygon_list,s.relative_centroids,s.pos_gamma,s.rot_gamma), None), lambda _, xs: GluedPolygons(xs[0],xs[1],xs[2],xs[3]))

def glue_polygons_together(list_of_poly_vertices: List[jnp.ndarray]) -> Tuple[GluedPolygons,jnp.ndarray]:
    poly_centroid_list = [convex_polygon(vertices,return_centroid=True) for vertices in list_of_poly_vertices]
    polygons = [poly for poly,_ in poly_centroid_list]
    centroids = [centroid for _,centroid in poly_centroid_list]
    mean_centroid = jnp.mean(jnp.array(centroids),axis=0)
    # for i in range(len(list_of_poly_vertices)):
    #     polygons[i].vertices_0 += centroids[i] - mean_centroid

    glued_polygons = GluedPolygons(polygons, jnp.array(centroids)-mean_centroid, DEFAULT_WALL_GAMMA,DEFAULT_WALL_ROTATIONAL_GAMMA)

    return glued_polygons, mean_centroid

# t1 = jnp.array([[-10.,-10.],[10.,-10.],[5.,-5],[-5.,-5]])
# t2 = jnp.array([[-10.,-10.],[-5.,-5.],[-5.,5.],[-10.,10.]])

# print(jnp.mean((t1+t2)/2,axis=0))

# t3 = jnp.array([[-10.,-10.],[10.,-10.],[5.,-5],[-5.,-5]]) + jnp.array([10.,10.])
# t4 = jnp.array([[-10.,-10.],[-5.,-5.],[-5.,5.],[-10.,10.]]) + jnp.array([10.,10.])

# print(jnp.mean((t3+t4)/2,axis=0))

# glu1,c1 = glue_polygons_together([t1,t2])
# glu2,c1 = glue_polygons_together([t3,t4])

# # p1, c1 = convex_polygon(t1, return_centroid=True)
# # p2, c2 = convex_polygon(t2, return_centroid=True)
# # p3, c3 = convex_polygon(t3, return_centroid=True)
# # p4, c4 = convex_polygon(t4, return_centroid=True)

# centroids = np.array([c1, c2, c3, c4])
# angles = jnp.zeros(centroids.shape[0])

# gluing_list.enforce_gluing(centroids, angles)
