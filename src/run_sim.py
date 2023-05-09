from typing import Tuple, List
import jax
import re
import os

from numpy import array as np_convert_to_array

from visualization import get_parameter_report

np_objarr = lambda x: np_convert_to_array(x,dtype="object")

from collisions import collide_ow

from collections.abc import Iterable
import jax.numpy as jnp
from jax import random as rand
from jax import jit

import matplotlib.pyplot as plt

from tqdm import trange

from constants import *

from objects import ConvexPolygon, convex_polygon,glue_polygons_together


from visualization import animate_particles

# Create a regular expression pattern to match "yes" or "no" and their variants
yes_no_pattern = re.compile(r'^(y|yes|yeah|yup|yea|no|n|nope)$', re.IGNORECASE)

MANY = 20


initial_random_key = rand.PRNGKey(678912390)

def rotation_noise(rand_key, num_particles: int, rotation_diffusion: float, dt: float) -> Tuple[jnp.ndarray,jnp.ndarray]:
    r"""
    Computes `\delta`-correlated noise used to cause drift in `\theta(t)`.
    Becaus we work in discrete-time, we must ensure that 

    `\int_0^T \langle \xi(t)\xi(t')\rangle = 2D_R`

    We will draw our `\xi` values as IID zero-mean Gaussians, so two values of
    `\xi(t)`, `\xi(t')` will never be correlated, except when `t=t'`. But 
    to ensure that the integral takes on the correct value, we must recall that
    if `\xi` is Norm(0, `\sigma`) distributed, then because it takes this value
    for our discrete time `dt`, we have
    
    `\int_0^T \langle \xi^2(t)\rangle dt = dt\sigma^2 = 2D_R`

    so we must draw our noise terms from a Norm(0,`\sqrt{2D_R/dt}`) distribution.
    """

    key, new_key = rand.split(rand_key)

    return new_key, rand.normal(key, (num_particles,), float) * jnp.sqrt(2*rotation_diffusion / dt)

def translation_noise(rand_key, num_particles: int, translation_diffusion: float, dt: float) -> Tuple[jnp.ndarray,jnp.ndarray]: 
    r"""
    Computes `\delta`-correlated noise used to cause drift in `r(t)`. 
    Because we work in discrete-time, we must ensure that 

    `\int_0^T \langle\zeta(t)\zeta(t')\rangle = 2D_T`

    We will draw our `\zeta` values as IID zero-mean 2D Gaussians, so two components
    `\zeta_i(t)`, `\zeta_j(t')` will never be correltated unless `t=t'` and `i=j`. 
    But to ensure that the integral takes on the correct value, we must recall that
    if our `\zeta_i` is Norm(0,`\sigma`) distributed, then for discrete time 

    `\int_0^T\langle \xi^2_i(t)\rangle dt = dt\sigma^1 = 2D_T`

    so we must draw our noise terms from a Norm(0,`\sqrt{2D_T}/dt`) distribution. 
    """

    key, new_key = rand.split(rand_key)

    return new_key, rand.normal(key, (num_particles,2), float) * jnp.sqrt(2*translation_diffusion / dt)

def get_derivatives(
        r: jnp.ndarray, 
        theta: jnp.ndarray,

        rand_key: jnp.ndarray,
        sim_params: dict = {},
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_particles = r.shape[0]

    dt =                                 sim_params.get("dt",DEFAULT_DT)
    v0 =                                 sim_params.get("v0",DEFAULT_V0)
    translation_gamma =                  sim_params.get("translation_gamma",DEFAULT_TRANSLATION_GAMMA)
    translation_diffusion =              sim_params.get("translation_diffusion",DEFAULT_TRANSLATION_DIFFUSION)
    rotation_gamma =                     sim_params.get("rotation_gamma",DEFAULT_ROTATION_GAMMA)
    rotation_diffusion =                 sim_params.get("rotation_diffusion",DEFAULT_ROTATION_DIFFUSION)
    omega =                              sim_params.get("omega",DEFAULT_OMEGA)

    heading_vector = jnp.array([jnp.cos(theta),jnp.sin(theta)]).transpose()
    rand_key, zeta = translation_noise(rand_key,num_particles,translation_diffusion,dt)
    r_dot = v0 * heading_vector + zeta/translation_gamma # should have shape (n,2).

    rand_key, xi = rotation_noise(rand_key, num_particles, rotation_diffusion, dt)
    theta_dot = omega + xi/rotation_gamma

    return rand_key, r_dot, theta_dot
get_derivatives = jit(get_derivatives)

def run_sim(
        initial_positions: jnp.ndarray, 
        initial_heading_angles: jnp.ndarray,
        polygons: List[ConvexPolygon],
        centroids: List[jnp.ndarray],
        sim_params: dict = {},
        hell: bool = False
        ) -> jnp.ndarray: 
    r"""
    We work with two-dimensional Active Brownian Particles (ABPs). 
    These have a preferred direction of motion, parametrized by a 
    heading variable `\theta(t)` and a constant preferred velocity v0. 

    These particles work in the overdamped limit, where forces translate
    directly into velocities, scaled by the drag coefficient `gamma`. 
    The translational equation of motion for our ABPs is given by 

    `\gamma_T (\dot{r} - v_0n(t)) = \zeta(t)`

    where `\zeta(t)` is a zero-mean, `\delta`-correlated noise term: 

    `\langle \zeta_i(t)\zeta_j(t')\rangle = 2D_T \delta(t-t')\delta_{ij}`. 

    and `n(t) = [\cos(\theta(t)),\sin(\theta(t))]` is the heading vector
    for the ABP. The particle also experiences drift in its heading, which
    we assume is primarily diffusive: 

    `\gamma_R(\dot{\theta} -\omega) = \xi(t)`

    where `\xi(t)` is another `\delta`-correlated zero mean rotation term: 

    `\langle \xi(t)\xi(t')\rangle = 2D_R \delta(t-t')`

    and `\omega` is a natural rotation rate. These particles are also assumed
    to completely re-assign their heading on a Unif[0,2`\pi`] distribution with
    Poisson-like dynamics, parametrized by `p` the rate of transition per unit
    time.
    """

    use_jit =                           sim_params.get("use_jit", True)
    num_particles =                     sim_params.get("num_particles",DEFAULT_NUM_PARTICLES)
    dt =                                sim_params.get("dt",DEFAULT_DT)
    total_time =                        sim_params.get("total_time",DEFAULT_TOTAL_TIME)
    tumble_rate =                       sim_params.get("tumble_rate",DEFAULT_TUMBLE_RATE)
    return_history =                    sim_params.get("return_history", DEFAULT_RETURN_HISTORY)
    timesteps_per_frame =               sim_params.get("timesteps_per_frame",DEFAULT_TIMESTEPS_PER_FRAME)

    assert timesteps_per_frame % MANY == 0

    r_history = []
    theta_history = []
    poly_vertex_history = []
    poly_com_history = []
    poly_angle_history = []

    angles = [0. for _ in polygons]

    for _ in polygons:
        poly_vertex_history.append([])
        poly_com_history.append([])
        poly_angle_history.append([])
        

    rand_key = initial_random_key

    num_steps = int(total_time / dt/MANY)
    
    if hell:
        r_hell = jnp.zeros((num_particles,2),float)
        hell_q = jnp.zeros((num_particles,),float)
    else:
        r_hell = None
        hell_q = None

    r = initial_positions.copy()[:num_particles]
    theta = initial_heading_angles.copy()[:num_particles]

    assert r.shape == (sim_params["num_particles"],2)

    r_history.append(r)
    theta_history.append(theta)
    for i,poly in enumerate(polygons):
        poly_vertex_history[i].append(poly.get_vertices(centroids[i], angles[i]))
        poly_com_history[i].append(centroids[i])
        poly_angle_history[i].append(angles[i])
    

    # We know that angle reassignment is done as a Poisson process, so the time
    # between events is distributed as Expo(tumble_rate).
    
    key, rand_key = rand.split(rand_key)
    time_until_angle_reassignment = rand.exponential(key,(num_particles,),float) / tumble_rate
    next_reassignment_all_particles = (time_until_angle_reassignment/dt).astype(jnp.int32)
    next_reassignment_event = jnp.min(next_reassignment_all_particles)

    for step in trange(num_steps):
        if use_jit:
            sim_update_chunk = _jit_do_many_sim_steps(rand_key, r, theta, sim_params, polygons, centroids, angles, hell_q, r_hell)
        else:
            sim_update_chunk = do_many_sim_steps(rand_key, r, theta, sim_params, polygons, centroids, angles, hell_q, r_hell)

        rand_key, r, theta, centroids, angles, r_hell = sim_update_chunk

        if step % int(timesteps_per_frame/MANY) == 0 and return_history:
            r_history.append(r)
            theta_history.append(theta)
            for i,poly in enumerate(polygons):
                poly_vertex_history[i].append(poly.get_vertices(centroids[i], angles[i]))
                poly_angle_history[i].append(angles[i])
                poly_com_history[i].append(centroids[i])

        if step >= next_reassignment_event:
            reassign_which_particles = (step>=next_reassignment_all_particles)
            num_reassignments = jnp.count_nonzero(reassign_which_particles)
            
            key, rand_key = rand.split(rand_key)
            new_thetas = rand.uniform(key,(num_reassignments,),float,0.,2*jnp.pi)
            theta = theta.at[reassign_which_particles].set(new_thetas)
            
            key, rand_key = rand.split(rand_key)
            next_reassignment_of_reassigned_particles = step + (rand.exponential(key,(num_reassignments,),float) / tumble_rate / dt).astype(jnp.int32)
            next_reassignment_all_particles = next_reassignment_all_particles.at[reassign_which_particles].set(next_reassignment_of_reassigned_particles)

            next_reassignment_event = jnp.min(next_reassignment_all_particles)

    if return_history:
        return (jnp.array(r_history), jnp.array(theta_history),
                poly_vertex_history,poly_com_history, poly_angle_history, r_hell)
    else:
<<<<<<< HEAD
        return r, theta, [sub_poly.get_vertices(centroids[i] + sub_com,angles[i]) for i,poly in enumerate(polygons) for sub_poly,sub_com in zip(poly.polygon_list,poly.get_relative_centroids(angles[i]))]
        
        # return r, theta, [poly.get_vertices(centroids[i],angles[i]) for i,poly in enumerate(polygons)]
=======
        return r, theta, [sub_poly.get_vertices(centroids[i]+sub_com,angles[i]) for i,poly in enumerate(polygons) for sub_poly,sub_com in zip(poly.polygon_list,poly.get_relative_centroids(angles[i]))]
>>>>>>> 39df6a0 (MONOTILE)

def do_many_sim_steps(
        rand_key: jnp.ndarray, 
        r: jnp.ndarray, 
        theta: jnp.ndarray, 
        sim_params: dict, 
        polygons: List[ConvexPolygon],
        centroids: List[jnp.ndarray], 
        angles: List[float],
        hell_q: jnp.ndarray = None,
        r_hell: jnp.ndarray = None,
        hell_cutoff: float = 0.1
        ) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
    
    dt = sim_params.get("dt", DEFAULT_DT)
    box_size = sim_params.get("box_size", DEFAULT_BOX_SIZE)
    translation_gamma = sim_params.get("translation_gamma", DEFAULT_TRANSLATION_GAMMA)
    pbc_size = sim_params.get("pbc_size", DEFAULT_PERIODIC_BOUNDARY_SIZE)

    for sub_step in range(MANY):
        rand_key, r_dot, theta_dot = get_derivatives(r,theta,rand_key,sim_params)
        delta_r = r_dot * dt
        delta_theta = theta_dot * dt
        for poly_indx, poly in enumerate(polygons):
            mpv_corrections = poly.get_min_particle_push_vector(centroids[poly_indx], angles[poly_indx], r)
            poly_com_adjustment = -jnp.sum(mpv_corrections,axis=0) * translation_gamma / poly.pos_gamma
            poly_angle_adjustment = poly.get_rotation_from_wall_particle_interaction(centroids[poly_indx], angles[poly_indx], r, mpv_corrections, translation_gamma)

            delta_r += mpv_corrections
            centroids[poly_indx] += poly_com_adjustment
            angles[poly_indx] += poly_angle_adjustment
            
            wall_com_correction, wall_rot_correction = collide_ow(poly, centroids[poly_indx], angles[poly_indx], box_size/2) # correct for object-wall collisions
            angles[poly_indx] += wall_rot_correction
            centroids[poly_indx] += wall_com_correction * 0.1

            for op_indx in range(poly_indx):
                correct_op, correct_poly = polygons[op_indx].collide_oo_wrapper(centroids[op_indx],angles[op_indx],poly,centroids[poly_indx],angles[poly_indx])
                centroids[poly_indx] -= correct_poly
                centroids[op_indx] -= correct_op
                
            if r_hell is not None:
                hell_q = hell_q + poly.hell_query(centroids[poly_indx], angles[poly_indx], r, cutoff=hell_cutoff)
        
        
        r = jax.lax.clamp(-box_size/2,r + delta_r,box_size/2)
        theta = theta + delta_theta
        
        if pbc_size is not None:
            r = jnp.mod(r + pbc_size/2., pbc_size) - pbc_size/2.
            
        if r_hell is not None:
            r_abs = jnp.abs(r)
            hell_q += jnp.max(jnp.heaviside(hell_cutoff - (box_size/2 - r_abs), 0.), axis=1) # Summon demons from the walls
            
            hell_q = jax.lax.clamp(0., hell_q - 1., 1.) # Only send particles with multiple collisions to hell
            r_hell += jnp.outer(hell_q, jnp.array([0.,-(box_size * 2.)])) # Hell is below the box
            r += r_hell
        
    return rand_key, r, theta, centroids, angles, r_hell
_jit_do_many_sim_steps = jit(do_many_sim_steps)

def get_initial_fill_shape(
        geometry_name: str,
        shape_list: List[jnp.ndarray], 
        box_size: float,
        overwrite_cache: bool = False,
        sim_params: dict = {},
        ) -> Tuple[jnp.ndarray,jnp.ndarray]:
    num_particles = sim_params.get("num_particles",DEFAULT_NUM_PARTICLES)
    while True:
        if os.path.exists(os.path.join(PROJECT_DIR,"particle_distributions",f"{geometry_name}_r.npy")) and not overwrite_cache:
            initial_positions = jnp.load(os.path.join(PROJECT_DIR,"particle_distributions",f"{geometry_name}_r.npy"),allow_pickle=True)
            initial_headings = jnp.load(os.path.join(PROJECT_DIR,"particle_distributions",f"{geometry_name}_theta.npy"),allow_pickle=True)
            comparison_list = jnp.load(os.path.join(PROJECT_DIR,"particle_distributions",f"{geometry_name}_shape.npy"),allow_pickle=True)
            
            try:
                assert jnp.all(jnp.array([jnp.allclose(shape,shape_to_compare) for shape,shape_to_compare in zip(shape_list,comparison_list)]))
                # Break out of while loop if assertion passes
                break
            except:
                clear_cache = input('Mismatch between shape and cached version. Clear cache? [Y/n]: ') or "y"
                if yes_no_pattern.match(clear_cache):
                    if clear_cache.lower() in ['y', 'yes', 'yeah', 'yup', 'yea']:
                        for filename in os.listdir(os.path.join(PROJECT_DIR, 'particle_distributions')):
                            # Check if the filename contains the shape name
                            if geometry_name in filename:
                                # Remove the file
                                os.remove(os.path.join(PROJECT_DIR, 'particle_distributions', filename))
                        overwrite_cache = True
                    else:
                        raise AssertionError('Fatal mismatch between shape and cached version.')
                else:
                    # User input is invalid
                    print("Invalid input. Please enter 'yes' or 'no'.")
        else:
            poly_and_com_list = [convex_polygon(shape, return_centroid=True) for shape in shape_list]
            
            rand_key = initial_random_key
            initial_positions = rand.uniform(rand_key,(DEFAULT_NUM_PARTICLES,2), float,-box_size/2,box_size/2)
            initial_headings = rand.uniform(rand_key,(DEFAULT_NUM_PARTICLES,),float,0,2*jnp.pi)
            
            which_are_inside = jnp.any(jnp.array([poly.is_inside(com,0.,initial_positions) for poly,com in poly_and_com_list]),axis=0)

            while jnp.count_nonzero(which_are_inside)!=0:
                print(f"Refreshing again. {jnp.count_nonzero(which_are_inside)} remain.")
                rand_key, key = rand.split(rand_key)
                
                valid_initial_conditions = initial_positions[jnp.logical_not(which_are_inside)]
                new_initial_positions = rand.uniform(key,(jnp.count_nonzero(which_are_inside),2),float,-box_size/2,box_size/2)
                
                initial_positions = jnp.concatenate((valid_initial_conditions, new_initial_positions), axis=0)
                which_are_inside = jnp.any(jnp.array([poly.is_inside(com,0.,initial_positions) for poly,com in poly_and_com_list]),axis=0)
            
            jnp.save(os.path.join(PROJECT_DIR, f"particle_distributions/{geometry_name}_r.npy"),initial_positions,allow_pickle=True)
            jnp.save(os.path.join(PROJECT_DIR, f"particle_distributions/{geometry_name}_theta.npy"),initial_headings,allow_pickle=True)
            jnp.save(os.path.join(PROJECT_DIR, f"particle_distributions/{geometry_name}_shape.npy"),np_objarr(shape_list),allow_pickle=True)

            # Break out of while loop if cache clear corrects assertion error
            break

    return initial_positions[:num_particles],initial_headings[:num_particles]



# AVERY WHY DID YOU DEFINE GLOBAL VARIABLES IN THE RUN PARAMETERS AAAAAAAA
# LOOK AT ME I'M A GOOD LITTLE BOY


def main():
    # t1 = jnp.array([[-10.,-10.],[10.,-10.],[5.,-5],[-10.,-5]])
    # t2 = jnp.array([[-10.,-10.],[-5.,-10.],[-5.,5.],[-10.,10.]])

    # t3 = jnp.array([[-10.,-10.],[10.,-10.],[5.,-5],[-10.,-5]]) - jnp.array([6.,6.])
    # t4 = jnp.array([[-10.,-10.],[-5.,-10.],[-5.,5.],[-10.,10.]]) - jnp.array([6.,6.])

    # r_0, theta_0 = get_initial_fill_shape(
    #     "two_arrows",
    #     [t1, t2, t3, t4],
    #     DEFAULT_BOX_SIZE,
    #     overwrite_cache=True
    #     )

    # glu1, c1 = glue_polygons_together([t1,t2])
    # glu2, c2 = glue_polygons_together([t3,t4])


    mono_sub_1 = jnp.array([[1.25, 0.433013],[0.5, 0.866025],[-0.25, 0.433013],[0., 0.],[1.,0.]]) * 5 + jnp.array([[5,5]])
    mono_sub_2 = jnp.array([[1.25, 0.433013],[2., 0.],[2.75, 0.433013],[2.5, 0.866025],[1.5, 0.866025]]) * 5 + jnp.array([[5,5]])
    mono_sub_3 = jnp.array([[1., 1.73205],[0.5, 1.73205],[0.5, 0.866025],[1.25, 0.433013],[1.5, 0.866025]]) * 5 + jnp.array([[5,5]])
    mono_sub_4 = jnp.array([[2., 0.866025],[2., 1.73205],[1.25, 2.16506],[1., 1.73205],[1.5, 0.866025]]) * 5 + jnp.array([[5,5]])
    
    mono2_sub_1 = jnp.array([[1.25, 0.433013],[0.5, 0.866025],[-0.25, 0.433013],[0., 0.],[1.,0.]]) * 5 + jnp.array([[-10,-10]])
    mono2_sub_2 = jnp.array([[1.25, 0.433013],[2., 0.],[2.75, 0.433013],[2.5, 0.866025],[1.5, 0.866025]]) * 5 + jnp.array([[-10,-10]])
    mono2_sub_3 = jnp.array([[1., 1.73205],[0.5, 1.73205],[0.5, 0.866025],[1.25, 0.433013],[1.5, 0.866025]]) * 5 + jnp.array([[-10,-10]])
    mono2_sub_4 = jnp.array([[2., 0.866025],[2., 1.73205],[1.25, 2.16506],[1., 1.73205],[1.5, 0.866025]]) * 5 + jnp.array([[-10,-10]])
    
    mono3_sub_1 = jnp.array([[1.25, 0.433013],[0.5, 0.866025],[-0.25, 0.433013],[0., 0.],[1.,0.]]) * 5 + jnp.array([[-15,10]])
    mono3_sub_2 = jnp.array([[1.25, 0.433013],[2., 0.],[2.75, 0.433013],[2.5, 0.866025],[1.5, 0.866025]]) * 5 + jnp.array([[-15,10]])
    mono3_sub_3 = jnp.array([[1., 1.73205],[0.5, 1.73205],[0.5, 0.866025],[1.25, 0.433013],[1.5, 0.866025]]) * 5 + jnp.array([[-15,10]])
    mono3_sub_4 = jnp.array([[2., 0.866025],[2., 1.73205],[1.25, 2.16506],[1., 1.73205],[1.5, 0.866025]]) * 5 + jnp.array([[-15,10]])
    
    mono4_sub_1 = jnp.array([[1.25, 0.433013],[0.5, 0.866025],[-0.25, 0.433013],[0., 0.],[1.,0.]]) * 5 + jnp.array([[15,-10]])
    mono4_sub_2 = jnp.array([[1.25, 0.433013],[2., 0.],[2.75, 0.433013],[2.5, 0.866025],[1.5, 0.866025]]) * 5 + jnp.array([[15,-10]])
    mono4_sub_3 = jnp.array([[1., 1.73205],[0.5, 1.73205],[0.5, 0.866025],[1.25, 0.433013],[1.5, 0.866025]]) * 5 + jnp.array([[15,-10]])
    mono4_sub_4 = jnp.array([[2., 0.866025],[2., 1.73205],[1.25, 2.16506],[1., 1.73205],[1.5, 0.866025]]) * 5 + jnp.array([[15,-10]])
    
    mmono_sub_1 = jnp.array([[1.25, 0.433013],[0.5, 0.866025],[-0.25, 0.433013],[0., 0.],[1.,0.]]) * -5 + jnp.array([[35,15]])
    mmono_sub_2 = jnp.array([[1.25, 0.433013],[2., 0.],[2.75, 0.433013],[2.5, 0.866025],[1.5, 0.866025]]) * -5 + jnp.array([[35,15]])
    mmono_sub_3 = jnp.array([[1., 1.73205],[0.5, 1.73205],[0.5, 0.866025],[1.25, 0.433013],[1.5, 0.866025]]) * -5 + jnp.array([[35,15]])
    mmono_sub_4 = jnp.array([[2., 0.866025],[2., 1.73205],[1.25, 2.16506],[1., 1.73205],[1.5, 0.866025]]) * -5 + jnp.array([[35,15]])
    
    mmono2_sub_1 = jnp.array([[1.25, 0.433013],[0.5, 0.866025],[-0.25, 0.433013],[0., 0.],[1.,0.]]) * -5 + jnp.array([[-30,-10]])
    mmono2_sub_2 = jnp.array([[1.25, 0.433013],[2., 0.],[2.75, 0.433013],[2.5, 0.866025],[1.5, 0.866025]]) * -5 + jnp.array([[-30,-10]])
    mmono2_sub_3 = jnp.array([[1., 1.73205],[0.5, 1.73205],[0.5, 0.866025],[1.25, 0.433013],[1.5, 0.866025]]) * -5 + jnp.array([[-30,-10]])
    mmono2_sub_4 = jnp.array([[2., 0.866025],[2., 1.73205],[1.25, 2.16506],[1., 1.73205],[1.5, 0.866025]]) * -5 + jnp.array([[-30,-10]])
    
    mmono3_sub_1 = jnp.array([[1.25, 0.433013],[0.5, 0.866025],[-0.25, 0.433013],[0., 0.],[1.,0.]]) * -5 + jnp.array([[-15,30]])
    mmono3_sub_2 = jnp.array([[1.25, 0.433013],[2., 0.],[2.75, 0.433013],[2.5, 0.866025],[1.5, 0.866025]]) * -5 + jnp.array([[-15,30]])
    mmono3_sub_3 = jnp.array([[1., 1.73205],[0.5, 1.73205],[0.5, 0.866025],[1.25, 0.433013],[1.5, 0.866025]]) * -5 + jnp.array([[-15,30]])
    mmono3_sub_4 = jnp.array([[2., 0.866025],[2., 1.73205],[1.25, 2.16506],[1., 1.73205],[1.5, 0.866025]]) * -5 + jnp.array([[-15,30]])
    
    mmono4_sub_1 = jnp.array([[1.25, 0.433013],[0.5, 0.866025],[-0.25, 0.433013],[0., 0.],[1.,0.]]) * -5 + jnp.array([[15,-30]])
    mmono4_sub_2 = jnp.array([[1.25, 0.433013],[2., 0.],[2.75, 0.433013],[2.5, 0.866025],[1.5, 0.866025]]) * -5 + jnp.array([[15,-30]])
    mmono4_sub_3 = jnp.array([[1., 1.73205],[0.5, 1.73205],[0.5, 0.866025],[1.25, 0.433013],[1.5, 0.866025]]) * -5 + jnp.array([[15,-30]])
    mmono4_sub_4 = jnp.array([[2., 0.866025],[2., 1.73205],[1.25, 2.16506],[1., 1.73205],[1.5, 0.866025]]) * -5 + jnp.array([[15,-30]])
    
    triangle = jnp.array([[-10.,-10.],[10.,-10.],[-10.,10.]])
    r_0, theta_0 = get_initial_fill_shape(
        "monotile_1",
        [mono_sub_1,mono_sub_2,mono_sub_3,mono_sub_4,
         mono2_sub_1,mono2_sub_2,mono2_sub_3,mono2_sub_4,
         mono3_sub_1,mono3_sub_2,mono3_sub_3,mono3_sub_4,
         mono4_sub_1,mono4_sub_2,mono4_sub_3,mono4_sub_4,
         mmono_sub_1,mmono_sub_2,mmono_sub_3,mmono_sub_4,
         mmono2_sub_1,mmono2_sub_2,mmono2_sub_3,mmono2_sub_4,
         mmono3_sub_1,mmono3_sub_2,mmono3_sub_3,mmono3_sub_4,
         mmono4_sub_1,mmono4_sub_2,mmono4_sub_3,mmono4_sub_4],
        DEFAULT_BOX_SIZE,
        overwrite_cache=True
    )
    glu, c = glue_polygons_together([mono_sub_1,mono_sub_2,mono_sub_3,mono_sub_4])
    glu2, c2 = glue_polygons_together([mono2_sub_1,mono2_sub_2,mono2_sub_3,mono2_sub_4])
    glu3, c3 = glue_polygons_together([mono3_sub_1,mono3_sub_2,mono3_sub_3,mono3_sub_4])
    glu4, c4 = glue_polygons_together([mono4_sub_1,mono4_sub_2,mono4_sub_3,mono4_sub_4])
    mglu, mc = glue_polygons_together([mmono_sub_1,mmono_sub_2,mmono_sub_3,mmono_sub_4])
    mglu2, mc2 = glue_polygons_together([mmono2_sub_1,mmono2_sub_2,mmono2_sub_3,mmono2_sub_4])
    mglu3, mc3 = glue_polygons_together([mmono3_sub_1,mmono3_sub_2,mmono3_sub_3,mmono3_sub_4])
    mglu4, mc4 = glue_polygons_together([mmono4_sub_1,mmono4_sub_2,mmono4_sub_3,mmono4_sub_4])
    
    angle_hist_fig, angle_hist_ax = plt.subplots(figsize=(6,6))
    com_hist_fig, com_hist_ax = plt.subplots(figsize=(6,6))

<<<<<<< HEAD
    for rotation_diffusion in [1e-5,1e-4,3e-4,6e-4,1e-3,3e-3,6e-3,1e-2,1e-1]:
        sim_params = {
            "dt": 0.01,
            "num_particles": 10000,
            "total_time": 5000.,
            "do_animation": True,
            "return_history": True,
            "rotation_diffusion":rotation_diffusion,
            "use_jit": True,
            "timesteps_per_frame": 10000
=======
    for v0 in [6.,]:
        sim_params = {
            "dt": 0.01,
            "num_particles": 10000,
            "total_time": 2.,
            "do_animation": True,
            "return_history": True,
            "v0": v0,
            "use_jit": False,
            "timesteps_per_frame": 2000
>>>>>>> 39df6a0 (MONOTILE)
            }

        r_history, theta_history, poly_history, com_history, angle_history, _ = run_sim(r_0, theta_0, [glu,glu2,glu3,glu4,mglu,mglu2,mglu3,mglu4], [c,c2,c3,c4,mc,mc2,mc3,mc4], sim_params, hell=False)

        times = jnp.linspace(0,sim_params["total_time"],len(r_history))

        param_name = r"$D_R$"
        scan_param_title = param_name + f" = {sim_params['rotation_diffusion']}"

        angle_hist_ax.set_title(f"Effect of {param_name} on polygon rotation\n"+get_parameter_report(param_name))
        angle_hist_ax.set_xlabel("Time (a.u.)")
        angle_hist_ax.set_ylabel("Polygon Angle (rad)")
        angle_hist_ax.plot(times,jnp.array(angle_history[0]),label=scan_param_title)
        angle_hist_ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        angle_hist_fig.savefig(f"{PROJECT_DIR}/plots/angle_history.png",bbox_inches="tight")

        max_displacement = DEFAULT_BOX_SIZE/jnp.sqrt(2)
        com_hist_ax.set_title(f"Effect of {param_name} on Corner Docking Rate\n"+get_parameter_report(param_name))
        com_hist_ax.set_xlabel("Time (a.u.)")
        com_hist_ax.set_ylabel(r"Total Polygon Displacement (Normalized to $L/\sqrt{2}$)")
        com_hist_ax.plot(times,jnp.linalg.norm(jnp.array(com_history[0])/max_displacement,axis=1),label=scan_param_title)
        com_hist_ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        com_hist_fig.savefig(f"{PROJECT_DIR}/plots/com_history.png",bbox_inches="tight")

        # r_0 = 0*r_0
        # theta_0 = 0*theta_0 - 3*jnp.pi/4
        # r_history, theta_history, poly_history, r_hell = run_sim(r_0, theta_0, [glu1,glu2], [c1,c2], sim_params, hell=False)

        animate_particles(r_history, theta_history, poly_history, DEFAULT_BOX_SIZE,title=scan_param_title)

if __name__ == "__main__":
    main()