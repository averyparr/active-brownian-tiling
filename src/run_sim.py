from typing import Tuple
import jax

# import jax.numpy as jnp
import jax.numpy as jnp
from jax import random as rand
from jax import jit

import matplotlib.pyplot as plt

from tqdm import trange

from constants import *

from multiprocessing.pool import Pool


from visualization import animate_particles

TIMESTEPS_PER_FRAME = 500
MANY = 50
STEPS_PER_ROTATION_TRANSFORM = 50
assert TIMESTEPS_PER_FRAME % MANY == 0
assert MANY % STEPS_PER_ROTATION_TRANSFORM == 0

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

    dt =                                sim_params.get("dt",DEFAULT_DT)
    v0 =                                sim_params.get("v0",DEFAULT_V0)
    translation_gamma =                  sim_params.get("translation_gamma",DEFAULT_TRANSLATION_GAMMA)
    translation_diffusion =              sim_params.get("translation_diffusion",DEFAULT_TRANSLATION_DIFFUSION)
    rotation_gamma =                     sim_params.get("rotation_gamma",DEFAULT_ROTATION_GAMMA)
    rotation_diffusion =                 sim_params.get("rotation_diffusion",DEFAULT_ROTATION_DIFFUSION)
    omega =                             sim_params.get("omega",DEFAULT_OMEGA)

    heading_vector = jnp.array([jnp.cos(theta),jnp.sin(theta)]).transpose()
    rand_key, zeta = translation_noise(rand_key,num_particles,translation_diffusion,dt)
    r_dot = v0 * heading_vector + zeta/translation_gamma # should have shape (n,1,2).

    rand_key, xi = rotation_noise(rand_key, num_particles, rotation_diffusion, dt)
    theta_dot = omega + xi/rotation_gamma

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
        
    @jit
    def _jit_update_wall_positions(wall_correction_to_dr: jnp.ndarray, particle_gamma: float, wall_fluid_drag: float) -> jnp.ndarray:
        """
        Translates the entire WallHolder according to the total amount of impulse
        it has experienced in a given timestep. Suppose each particle i experiences 
        forces `f_i` from non-wall sources in time `dt`. Due to its fluid drag, it 
        experiences velocity `f_i/\gamma_i` and moves by `f_i dt/\gamma_i`. Suppose
        it collides with the wall for `d\tau <= dt` time. The wall exerts normal
        force `n_i` and so leads to a correction of `c_i = n_i d\tau/\gamma_i` in 
        the particle's position. The wall experiences a force `-n_i` for `d\tau` 
        and has fluid drag `\gamma_w`. It then moves by `-n_id\tau/\gamma_w`. We 
        can express this as 

        `wall_correction = -n_i d\tau/\gamma_w = -c_i \gamma_i / \gamma_w`


        Parameters
        ----------
        wall_correction_to_dr: jnp.ndarray
            (W,2) Array specifying our wall's modificaiton to particles' position 
            changes. wall_correction_to_dr[i,0] = `c_i`. 
        particle_gamma: float
            Specifies the fluid drag coefficient of the particles. 
        """

        return -jnp.sum(wall_correction_to_dr,axis=0) * particle_gamma / wall_fluid_drag



        ROTATIONAL / TORQUE EFFECTS NOT YET IMPLEMENTED
        """

        com_translation = -jnp.sum(wall_correction_to_dr,axis=0) * particle_gamma / self.fluid_drag
        self.wall_starts += com_translation
        self.wall_ends += com_translation

        
    @jit
    def _jit_update_wall_positions(wall_correction_to_dr: jnp.ndarray, particle_gamma: float, wall_fluid_drag: float) -> jnp.ndarray:
        return -jnp.sum(wall_correction_to_dr,axis=0) * particle_gamma / wall_fluid_drag
        

    @jit
    def _jit_get_collision_correction(
            r: jnp.ndarray,
            delta_r: jnp.ndarray,
            wall_starts: jnp.ndarray,
            wall_ends: jnp.ndarray,
            ) -> jnp.ndarray:
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

        wall_midpoints = (wall_starts + wall_ends) / 2.
        
        wall_diffs = wall_ends - wall_starts
        wall_lengths = jnp.linalg.norm(wall_diffs, axis=1)
        wall_thickness = 1.
        max_horizontal_distance_from_wall_center = 0.5 * wall_lengths + wall_thickness

        fraction_along_wall_vec = (wall_diffs.transpose() / wall_lengths).transpose()
        rot90_arr = jnp.array([[0, -1], [1, 0]])
        distance_from_wall_vec = (wall_diffs.transpose() / wall_lengths).transpose() @ rot90_arr.transpose()

        r_relative_to_wall_starts = r[:,None,:] - wall_midpoints

        r_dot_normal_wall = jnp.sum(r_relative_to_wall_starts * distance_from_wall_vec,axis=-1)
        r_dot_horizontal_wall = jnp.sum(r_relative_to_wall_starts * fraction_along_wall_vec,axis=2)

        dr_dot_normal_wall = jnp.sum(delta_r[:,None,:] * distance_from_wall_vec, axis=-1)

        normal_dist_from_walls = jnp.abs(r_dot_normal_wall + dr_dot_normal_wall)

        is_within_wall_parallel = jnp.abs(r_dot_horizontal_wall) < max_horizontal_distance_from_wall_center
        is_close_to_wall_normal = normal_dist_from_walls < wall_thickness


        hits_walls = is_within_wall_parallel & is_close_to_wall_normal

        # wall_correction = jnp.sign(dr_dot_normal_wall) * (normal_dist_from_walls - wall_thickness)
        wall_correction = (wall_thickness - normal_dist_from_walls)
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
    tumble_rate =                       sim_params.get("tumble_rate",DEFAULT_TUMBLE_RATE)
    wall_starts =                       sim_params.get("wall_starts", None)
    wall_ends =                         sim_params.get("wall_ends", None)
    return_history =                    sim_params.get("return_history", True)
    pbc_size =                          sim_params.get("pbc_size", DEFAULT_PERIODIC_BOUNDARY_SIZE)
    
    assert ((wall_starts is None and wall_ends is None) or (wall_starts.shape==wall_ends.shape))

    r_history = []
    theta_history = []
    walls_history = []

    rand_key = initial_random_key

    num_particles = initial_heading_angles.shape[0]
    assert initial_positions.shape == (num_particles,2)
    num_steps = int(total_time / dt/MANY)

    r = initial_positions.copy()
    theta = initial_heading_angles.copy()

    # We know that angle reassignment is done as a Poisson process, so the time
    # between events is distributed as Expo(tumble_rate).
    
    key, rand_key = rand.split(rand_key)
    time_until_angle_reassignment = rand.exponential(key,(num_particles,),float) / tumble_rate
    next_reassignment_all_particles = (time_until_angle_reassignment/dt).astype(jnp.int32)
    next_reassignment_event = jnp.min(next_reassignment_all_particles)

    # walls = WallHolder(wall_starts,wall_ends,wall_fluid_drag_coefficient=10)

    for step in trange(num_steps):
        rand_key, r, theta, wall_starts, wall_ends = do_many_sim_steps(rand_key, r, theta, sim_params, dt, wall_starts, wall_ends, pbc_size)

        if step % int(TIMESTEPS_PER_FRAME/MANY) == 0 and return_history:
            r_history.append(r)
            theta_history.append(theta)
            walls_history.append([wall_starts.copy(),wall_ends.copy()])

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
        return (jnp.array(r_history), jnp.array(theta_history),jnp.array(walls_history))
    else:
        return r, theta, None

@jit
def do_many_sim_steps(rand_key: jnp.ndarray, r: jnp.ndarray, theta: jnp.ndarray, sim_params: dict, dt: float, wall_starts: jnp.ndarray, wall_ends: jnp.ndarray, pbc_size: float) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
    particle_gamma = sim_params.get("translation_gamma", DEFAULT_TRANSLATION_GAMMA)
    wall_gamma_list = sim_params.get("wall_gamma_list", [DEFAULT_WALL_GAMMA]*len(wall_starts))
    for sub_step in range(MANY):
        rand_key, r_dot, theta_dot = get_derivatives(r,theta,rand_key,sim_params)
        delta_r = r_dot * dt
        delta_theta = theta_dot * dt
        
        if wall_starts is not None:
            for wall_indx, fluid_drag in enumerate(wall_gamma_list):
                correction = WallHolder._jit_get_collision_correction(r,delta_r,wall_starts[wall_indx],wall_ends[wall_indx])

                wall_com_adjustment = WallHolder._jit_update_wall_positions(correction,particle_gamma,fluid_drag)

                wall_starts = wall_starts.at[wall_indx].set(wall_starts[wall_indx] + wall_com_adjustment)
                wall_ends = wall_ends.at[wall_indx].set(wall_ends[wall_indx] + wall_com_adjustment)

                correction = WallHolder._jit_get_collision_correction(r,delta_r,wall_starts[wall_indx],wall_ends[wall_indx])
                delta_r += correction
        r += delta_r
        theta = theta + delta_theta
        
        if pbc_size is not None:
            r = jnp.mod(r + pbc_size/2., pbc_size) - pbc_size/2.
    return rand_key, r, theta, wall_starts, wall_ends

def wall_vecs_from_points(wall_points: jnp.ndarray,ordering=1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert ordering in [-1,1]
    return wall_points[None,:,:],jnp.roll(wall_points,ordering,axis=0)[None,:,:]


def get_initial_fill_shape(
        shape_name: str,
        shape, 
        initial_positions, 
        overwrite_cache: bool = False
        ) -> Tuple[jnp.ndarray,jnp.ndarray]:
    import os
    if os.path.exists(os.path.join(PROJECT_DIR,"particle_distributions",f"{shape_name}_r.npy")) and not overwrite_cache:
        # print(PROJECT_DIR)
        # exit()
        starting_positions = jnp.load(os.path.join(PROJECT_DIR,"particle_distributions",f"{shape_name}_r.npy"))
        starting_angles = jnp.load(os.path.join(PROJECT_DIR,"particle_distributions",f"{shape_name}_theta.npy"))
        shape_to_compare = jnp.load(os.path.join(PROJECT_DIR,"particle_distributions",f"{shape_name}_shape.npy"))
        assert jnp.allclose(shape,shape_to_compare)
    else:
        wall_starts, wall_ends = wall_vecs_from_points(shape)
        sim_params = {
            "total_time": 100.,
            "v0": 0.,
            "tumble_rate": 1e-9,
            "translation_gamma": 1,
            "rotation_gamma": 0.1,
            "wall_gamma_list": jnp.array([jnp.inf] * len(shape)),
            "pbc_size": 1e5,
            "return_history": True,
            "do_animation": True,
            "wall_starts": wall_starts,
            "wall_ends": wall_ends,
        }

        initial_headings = rand.uniform(initial_random_key,(nparticles,),float,0,2*jnp.pi)

        r_history,theta_history,wall_history = run_sim(initial_positions,initial_headings,sim_params)

        starting_positions,starting_angles = r_history[-1], theta_history[-1]
        
        jnp.save(f"../particle_distributions/{shape_name}_r.npy",starting_positions)
        jnp.save(f"../particle_distributions/{shape_name}_theta.npy",starting_angles)
        jnp.save(f"../particle_distributions/{shape_name}_shape.npy",shape)

        do_animation = sim_params.get("do_animation", DEFAULT_DO_ANIMATION)
        if do_animation:
            animate_particles(r_history,theta_history,wall_history, 1.5*box_size,1.5*box_size,gif_filename=f"../particle_distributions/{shape_name}_ani.gif")



    return starting_positions,starting_angles


def simulate_spike(box_size: float = 80., sim_params: dict = {}):
    x_offset = 0.1
    half_height = 0.5
    spike_shape = (box_size/2) * jnp.array([
            [-1+x_offset, -half_height],
            [-1+x_offset, half_height],
            [1+x_offset, half_height - half_height/3],
            [-0.5+x_offset, half_height - 2*half_height/3],
            [1+x_offset, half_height - 3*half_height/3],
            [-0.5+x_offset, half_height - 4*half_height/3],
            [1+x_offset, half_height - 5*half_height/3],
        ])

    initial_positions = rand.uniform(initial_random_key,(nparticles,2),float,-0.5,0.5)
    initial_positions = initial_positions - jnp.array([0.5,0])[None,:]
    initial_positions = initial_positions * jnp.array([0.3,0.3])[None,:] * box_size

    initial_positions, initial_headings = get_initial_fill_shape("triple_spike",spike_shape,initial_positions)




def simulate_with_walls(angle: float, gap_fraction: float, n_walls: int = 5, box_size: float = 80.) -> float:
    x_offset = 0.1
    half_height = 0.5

    # triple spike
    wall_shape = (box_size/2) * jnp.array([
            [-1+x_offset, -half_height],
            [-1+x_offset, half_height],
            [1+x_offset, half_height - half_height/3],
            [-0.5+x_offset, half_height - 2*half_height/3],
            [1+x_offset, half_height - 3*half_height/3],
            [-0.5+x_offset, half_height - 4*half_height/3],
            [1+x_offset, half_height - 5*half_height/3],
        ])

    wall_starts,wall_ends = wall_vecs_from_points(
        wall_shape
    )


    # # Triangle
    # wall_starts = (box_size/2) * jnp.array([[
    #     jnp.array([-1+x_offset,half_height]),
    #     jnp.array([-1+x_offset,-half_height]),
    #     jnp.array([1+x_offset,-half_height + half_height/4]),
    #     jnp.array([1+x_offset,0]),
    # ]])
    # wall_ends = (box_size/2) * jnp.array([[
    #     jnp.array([-1+x_offset,-half_height]),
    #     jnp.array([1+x_offset,0]),
    #     jnp.array([-1+x_offset,half_height]),
    # ]])

    sim_params = {
        "total_time": total_time,
        "pbc_size": box_size*1000,
        "return_history": True,
        "do_animation": True,
        "wall_starts": wall_starts,
        "wall_ends": wall_ends,
    }
    initial_positions = jnp.array([0.7,0.1])*rand.uniform(initial_random_key,(nparticles,2),float,-0.9*box_size/2.,0.9*box_size/2.) * 0
    initial_headings = rand.uniform(initial_random_key,(nparticles,),float,0,2*jnp.pi)

    r_history,theta_history,wall_history = run_sim(initial_positions,initial_headings,sim_params)
    
    plt.figure()
    plt.plot(jnp.count_nonzero(r_history.squeeze()[:,:,1] < 0.,axis=1),label="Bottom Half")
    plt.plot(jnp.count_nonzero(r_history.squeeze()[:,:,0] < 0.,axis=1),label="Left Half")
    plt.legend()

    plt.savefig("Accumulation in bottom")
    plt.figure()

    wall_mid_x,wall_mid_y = jnp.mean(wall_history - wall_history[0],axis=(1,3)).squeeze().transpose()
    plt.plot(wall_mid_x,label="Delta Mean X")
    plt.plot(wall_mid_y/5,label="Delta Mean Y/5")
    plt.legend()
    plt.savefig("triangle_motion.png")

    plt.figure()

    do_animation = sim_params.get("do_animation", DEFAULT_DO_ANIMATION)
    if do_animation:
        animate_particles(r_history,theta_history,wall_history, 1.5*box_size,1.5*box_size)

    final_r = jnp.mean(r_history[-100:],axis=0)

    return (jnp.tanh(-10*final_r.squeeze()[:,1])+1)/2 @ jnp.ones(nparticles) / nparticles # very close to (final_r.squeeze()[:,1] < 0.) @ jnp.ones(nparticles) but continuous

from jax import value_and_grad

total_time = 100.
nparticles = 10000
sim_grad = value_and_grad(simulate_with_walls,argnums=(0,1))
theta_0 = 0.7*jnp.pi
fraction_0 = 0.4
num_walls = 10
box_size = 100

# for _ in range(10):
#     sim_grad(theta_0,fraction_0,num_walls,box_size)

simulate_spike()
