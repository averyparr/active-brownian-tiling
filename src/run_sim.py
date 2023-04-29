from typing import Tuple

# import jax.numpy as jnp
import jax.numpy as jnp
from jax import random as rand
from jax import jit

import matplotlib.pyplot as plt

from tqdm import trange

from constants import *

from multiprocessing.pool import Pool


from visualization import animate_particles

MANY = 50

initial_random_key = rand.PRNGKey(678912390)
global total_time

def rotation_noise(rand_key, num_particles: int, rotationDiffusion: float, dt: float) -> Tuple[jnp.ndarray,jnp.ndarray]:
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

    return new_key, rand.normal(key, (num_particles,), float) * jnp.sqrt(2*rotationDiffusion / dt)
rotation_noise = jit(rotation_noise,static_argnums=(1,2,3))

def translation_noise(rand_key, num_particles: int, translationDiffusion: float, dt: float) -> Tuple[jnp.ndarray,jnp.ndarray]: 
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

    return new_key, rand.normal(key, (num_particles,2), float) * jnp.sqrt(2*translationDiffusion / dt)
translation_noise = jit(translation_noise,static_argnums=(1,2,3))

def get_derivatives(
        r: jnp.ndarray, 
        theta: jnp.ndarray,

        rand_key: jnp.ndarray,
        sim_params: dict = {},
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_particles = r.shape[0]

    dt =                                sim_params.get("dt",DEFAULT_DT)
    v0 =                                sim_params.get("v0",DEFAULT_V0)
    translationGamma =                  sim_params.get("translationGamma",DEFAULT_TRANSLATION_GAMMA)
    translationDiffusion =              sim_params.get("translationDiffusion",DEFAULT_TRANSLATION_DIFFUSION)
    rotationGamma =                     sim_params.get("rotationGamma",DEFAULT_ROTATION_GAMMA)
    rotationDiffusion =                 sim_params.get("rotationDiffusion",DEFAULT_ROTATION_DIFFUSION)
    omega =                             sim_params.get("omega",DEFAULT_OMEGA)

    heading_vector = jnp.array([jnp.cos(theta),jnp.sin(theta)]).transpose()
    rand_key, zeta = translation_noise(rand_key,num_particles,translationDiffusion,dt)
    r_dot = v0 * heading_vector + zeta/translationGamma # should have shape (n,1,2).

    rand_key, xi = rotation_noise(rand_key, num_particles, rotationDiffusion, dt)
    theta_dot = omega + xi/rotationGamma

    return rand_key, r_dot, theta_dot
get_derivatives = jit(get_derivatives)

class WallHolder:
    """
    Suppose we have W walls. Walls are parametrized as two (W,2) arrays 
    describing their start and end points, with the line between these points
    being the "wall". We treat particles as 0-size, and walls as having a
    thickness Th. At any given timestep, we determine if a particle is nearby
    to wall `w_i` parametrized by points `s_i,e_i` using two vectors: 
    a vector `f_i = (e_i - s_i)/||e_i-s_i||`, and a normal unit vector `n_i` 
    to the wall. Suppose a particle is at position r. We describe its position in 
    the basis of "along the wall" and "normal from the wall". In particular, 
    we compute  `f(r) = (r - m_i) @ f_i` as our distance along the wall, where
    `m_i` is the midpoint `m_i = (e_i+s_i)/2` normalized so `f(s_i) = -0.5` and 
    `f(e_i) = ||e_i-s_i|| = -0.5`. We additionally compute `n(r) = (r-s_i) @ n_i` 
    to give our distance from the wall. The particle is in contact with the 
    wall if abs(f(r)) < 0.5 and abs(n(r)) < Th. 

    If a particle tries to move _through_ a wall and our timestep `dt` is fine
    enough, then we can catch this motion by checking if (r+dr) is in contact
    with the wall. If that is the case, we presume that the wall exerts a normal
    force on the particle just enough to ensure that its final position is not
    within Th of the wall by setting 

    `r -> r + (Th - (r @ n_i)) n_i`
    
    which ensures (r @ n_i) = Th. 

    Because we have many different vectors to encode, it makes good sense to 
    keep them all in a single class (oh, would that Python have structs). 
    """
    def __init__(self, wall_starts:jnp.ndarray, wall_ends:jnp.ndarray) -> None:
        """
        Creates a WallHolder object, including computing required midpoint,
        paralell, and normal vectors. Walls are constructed between 
        `wall_starts[i]` and `wall_ends[i]`. 

        Parameters
        ----------
        wall_starts: jnp.ndarray
            A (W, 2) arrary encoding the starting points of each wall
        wall_ends: jnp.ndarray
            A (W, 2) array encoding the stopping points of each wall. 
        
        Returns
        ----------
        WallHolder object with all necessary vectors cached. 
        """

        self.wall_midpoints = (wall_starts + wall_ends)/2.
        self.wall_starts = wall_starts
        
        wall_diffs = wall_ends - wall_starts
        wall_lengths = jnp.apply_along_axis(jnp.linalg.norm,1,wall_diffs)
        self.wall_thickness = 1.
        self.max_horizontal_distance_from_wall_center = 0.5 * wall_lengths + self.wall_thickness

        self.fraction_along_wall_vec = (wall_diffs.transpose() / wall_lengths).transpose()
        rot90_arr = jnp.array([[0,-1],[1,0]])
        self.distance_from_wall_vec = (wall_diffs.transpose() / wall_lengths).transpose() @ rot90_arr

    def correct_for_collisions(self,
            r: jnp.ndarray,
            delta_r: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the updated value of `r`, taking into account whether
        any collisions occur. If none do, returns r + delta_r. Ensures
        that abs(r @ n_i) >= Th for all particles after evolution. 
        
        Parameters
        ----------
        r: jnp.ndarray
            Current positions of all particles. (n, 1, 2) Array.
        delta_r: jnp.ndarray
            Proposed update to positions of all particles, and
            should be equal to r_dot * dt. 

        Returns
        ----------
        correction: jnp.ndarray
            (W,2) Correction to delta_r that should ensure that no 
            final positions collide with the walls in this WallHolder. 
        """

        return WallHolder._jit_get_collision_correction(
            r,
            delta_r,
            self.wall_starts,
            self.wall_ends,
            ) 
    @jit
    def _jit_get_collision_correction(
            r: jnp.ndarray,
            delta_r: jnp.ndarray,
            wall_starts: jnp.ndarray,
            wall_ends: jnp.ndarray,
            ) -> jnp.ndarray:
        """
        Helper function for correct_for_collisions. Formatted
        so that it can be JIT-compiled by JAX. 

        See documentation for WallHelper.correct_for_collisions. 
        """

        wall_midpoints = (wall_starts + wall_ends) / 2.
        
        wall_diffs = wall_ends - wall_starts
        wall_lengths = jnp.linalg.norm(wall_diffs, axis=1)
        wall_thickness = 3.
        max_horizontal_distance_from_wall_center = 0.5 * wall_lengths + wall_thickness

        fraction_along_wall_vec = (wall_diffs.transpose() / wall_lengths).transpose()
        rot90_arr = jnp.array([[0, -1], [1, 0]])
        distance_from_wall_vec = (wall_diffs.transpose() / wall_lengths).transpose() @ rot90_arr

        r_relative_to_wall_starts = r[:,None,:] - wall_midpoints

        r_dot_normal_wall = jnp.sum(r_relative_to_wall_starts * distance_from_wall_vec,axis=-1)
        r_dot_horizontal_wall = jnp.sum(r_relative_to_wall_starts * fraction_along_wall_vec,axis=2)

        dr_dot_normal_wall = jnp.sum(delta_r[:,None,:] * distance_from_wall_vec, axis=-1)

        normal_dist_from_walls = jnp.abs(r_dot_normal_wall + dr_dot_normal_wall)

        is_within_wall_parallel = jnp.abs(r_dot_horizontal_wall) < max_horizontal_distance_from_wall_center
        is_close_to_wall_normal = normal_dist_from_walls < wall_thickness


        hits_walls = is_within_wall_parallel & is_close_to_wall_normal

        wall_correction = jnp.sign(dr_dot_normal_wall) * (normal_dist_from_walls - wall_thickness)
        wall_correction = (wall_correction * hits_walls)[:,:,None] * distance_from_wall_vec
        wall_correction = jnp.sum(wall_correction,axis=1)

        return wall_correction


def run_sim(
        initial_positions: jnp.ndarray, 
        initial_heading_angles: jnp.ndarray,
        sim_params: dict = {},
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

    dt =                                sim_params.get("dt",DEFAULT_DT)
    total_time =                        sim_params.get("total_time",DEFAULT_TOTAL_TIME)
    poissonAngleReassignmentRate =      sim_params.get("poissonAngleReassignmentRate",DEFAULT_POISSON_ANGLE_REASSIGNMENT_RATE)
    wall_starts =                       sim_params.get("wall_starts", None)
    wall_ends =                         sim_params.get("wall_ends", None)
    return_history =                    sim_params.get("return_history", True)
    pbc_size =                          sim_params.get("pbc_size", DEFAULT_PERIODIC_BOUNDARY_SIZE)
    
    assert ((wall_starts is None and wall_ends is None) or (wall_starts.shape==wall_ends.shape))

    r_history = []
    theta_history = []

    rand_key = initial_random_key

    num_particles = initial_heading_angles.shape[0]
    assert initial_positions.shape == (num_particles,2)
    num_steps = int(total_time / dt/MANY)

    r = initial_positions.copy()
    theta = initial_heading_angles.copy()

    # We know that angle reassignment is done as a Poisson process, so the time
    # between events is distributed as Expo(poissonAngleReassignmentRate).
    
    key, rand_key = rand.split(rand_key)
    time_until_angle_reassignment = rand.exponential(key,(num_particles,),float) / poissonAngleReassignmentRate
    next_reassignment_all_particles = (time_until_angle_reassignment/dt).astype(jnp.int32)
    next_reassignment_event = jnp.min(next_reassignment_all_particles)

    if wall_starts is not None:
        walls = WallHolder(wall_starts,wall_ends)

    for step in trange(num_steps):
        rand_key, r, theta, wall_starts, wall_ends = do_many_sim_steps(rand_key, r, theta, sim_params, dt, wall_starts, wall_ends, pbc_size)

        if step % int(50/MANY) == 0 and return_history:
            r_history.append(r)
            theta_history.append(theta)

        if step >= next_reassignment_event:
            reassign_which_particles = (step>=next_reassignment_all_particles)
            num_reassignments = jnp.count_nonzero(reassign_which_particles)
            
            key, rand_key = rand.split(rand_key)
            new_thetas = rand.uniform(key,(num_reassignments,),float,0.,2*jnp.pi)
            theta = theta.at[reassign_which_particles].set(new_thetas)
            
            key, rand_key = rand.split(rand_key)
            next_reassignment_of_reassigned_particles = step + (rand.exponential(key,(num_reassignments,),float) / poissonAngleReassignmentRate / dt).astype(jnp.int32)
            next_reassignment_all_particles = next_reassignment_all_particles.at[reassign_which_particles].set(next_reassignment_of_reassigned_particles)

            next_reassignment_event = jnp.min(next_reassignment_all_particles)

    if return_history:
        return (jnp.array(r_history), jnp.array(theta_history),jnp.array(walls_history))
    else:
        return r, theta, None

@jit
def do_many_sim_steps(rand_key: jnp.ndarray, r: jnp.ndarray, theta: jnp.ndarray, sim_params: dict, dt: float, wall_starts: jnp.ndarray, wall_ends: jnp.ndarray, pbc_size: float) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        
    for sub_step in range(MANY):
        rand_key, r_dot, theta_dot = get_derivatives(r,theta,rand_key,sim_params)
        delta_r = r_dot * dt
        delta_theta = theta_dot * dt
        
        if wall_starts is not None:
            for wall_indx, fluid_drag in enumerate(wall_fluid_drag_list):
                correction = WallHolder._jit_get_collision_correction(r,delta_r,wall_starts[wall_indx],wall_ends[wall_indx])
        
                delta_r += correction
        r += delta_r
        theta = theta + delta_theta
        
        if pbc_size is not None:
            r = jnp.mod(r + pbc_size/2., pbc_size) - pbc_size/2.
    return rand_key, r, theta, wall_starts, wall_ends


def simulate_with_walls(angle: float, gap_fraction: float, n_walls: int = 5, box_size: float = 80.) -> float:
    chevron_starts, chevron_ends = chevron_walls(n_walls,box_size,angle,gap_fraction)

    wall_starts = jnp.array([jnp.append((box_size/2)*BOUNDING_BOX_STARTS,chevron_starts,axis=0)[:4]])/1.5
    wall_ends = jnp.array([jnp.append((box_size/2)*BOUNDING_BOX_ENDS,chevron_ends,axis=0)[:4]])/1.5

    sim_params = {
        "total_time": total_time,
        "pbc_size": box_size,
        "return_history": True,
        "do_animation": True,
        "wall_starts": wall_starts,
        "wall_ends": wall_ends,
    }

    nparticles = 10000
    initial_positions = rand.uniform(initial_random_key,(nparticles,2),float,-box_size/2.,box_size/2.)
    initial_headings = rand.uniform(initial_random_key,(nparticles,),float,0,2*jnp.pi)

    r_history,theta_history = run_sim(initial_positions,initial_headings,sim_params)
    
    plt.plot(jnp.count_nonzero(r_history.squeeze()[:,:,1] < 0.,axis=1),label="Bottom Half")
    plt.plot(jnp.count_nonzero(r_history.squeeze()[:,:,0] < 0.,axis=1),label="Left Half")
    plt.legend()

    plt.savefig("Accumulation in bottom")
    plt.figure()

    do_animation = sim_params.get("do_animation", DEFAULT_DO_ANIMATION)
    if do_animation:
        animate_particles(r_history,theta_history,jnp.array([wall_starts,wall_ends]), 1.25*box_size,1.25*box_size)

    final_r = jnp.mean(r_history[-100:],axis=0)

    return (jnp.tanh(-10*final_r.squeeze()[:,1])+1)/2 @ jnp.ones(nparticles) / nparticles # very close to (final_r.squeeze()[:,1] < 0.) @ jnp.ones(nparticles) but continuous

from jax import value_and_grad

total_time = 400.
sim_grad = value_and_grad(simulate_with_walls,argnums=(0,1))
theta_0 = 2*jnp.pi/3
fraction_0 = 0.05

# learning_rate = 0.5
# theta = theta_0
# fraction = fraction_0
# for round in range(100):
#     val,(grad_theta, grad_fraction) = sim_grad(theta, fraction)
#     theta += learning_rate * grad_theta
#     fraction += learning_rate * grad_fraction

#     print(val,theta/jnp.pi,fraction,grad_theta,grad_fraction)

# for theta in jnp.pi*jnp.linspace(0.6,0.8,51):
#     fraction = simulate_with_walls(theta,0.1)
#     print(fraction,theta/jnp.pi)

for num_walls in range(5,6):
    lower_fraction = simulate_with_walls(0.7 * jnp.pi, 0.05, num_walls, 80.,)
    print(lower_fraction,num_walls)



