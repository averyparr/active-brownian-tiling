from typing import Tuple
import jax.numpy as jnp

DEFAULT_DT = 0.01
DEFAULT_TOTAL_TIME = 1000.
DEFAULT_BOX_SIZE = 100.
DEFAULT_NUM_PARTICLES = 10000
DEFAULT_V0 = 4.
DEFAULT_TRANSLATION_GAMMA = 1.
DEFAULT_TRANSLATION_DIFFUSION = 0.1
DEFAULT_ROTATION_GAMMA = 0.1
DEFAULT_ROTATION_DIFFUSION = 0.0008
DEFAULT_OMEGA = 0.
DEFAULT_TUMBLE_RATE = 1e-4
DEFAULT_WALL_GAMMA = 250.
DEFAULT_WALL_ROTATIONAL_GAMMA = 25000.

DEFAULT_TIMESTEPS_PER_FRAME = 500

DEFAULT_GIF_FPS = 30.
DEFAULT_DO_ANIMATION = True
DEFAULT_RETURN_HISTORY = True

DEFAULT_PERIODIC_BOUNDARY_SIZE = 100*DEFAULT_BOX_SIZE

import os
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

BOUNDING_BOX_VERTICES = DEFAULT_BOX_SIZE * jnp.array([
    [0.5,0.5],
    [-0.5,0.5],
    [-0.5,-0.5],
    [0.5,-0.5],
])


# DEPRECATED -- DOES NOT WORK WITH VERTEX BASED WALLING
# COULD IN PRINCIPLE BE MADE TO WORK, BUT THIS IS NOT HIGH PRIORITY
def chevron_walls(
        num_chevrons: int, 
        box_size: float, 
        chevron_angle: float, 
        gap_fraction: float
        ) -> Tuple[jnp.ndarray,jnp.ndarray]:
    center_positions = jnp.linspace(-box_size/2,box_size/2,num_chevrons)

    half_chevron_width = (1-gap_fraction) * box_size / (2*num_chevrons-2)
    half_wall_length = (half_chevron_width) / jnp.sin(chevron_angle/2)
    chevron_height = half_wall_length * jnp.cos(chevron_angle/2)
    
    start_positions = [[pos_x, chevron_height/2] for pos_x in center_positions] + [[pos_x, chevron_height/2] for pos_x in center_positions]
    end_positions = [[pos_x + half_chevron_width, -chevron_height/2] for pos_x in center_positions] + [[pos_x - half_chevron_width, -chevron_height/2] for pos_x in center_positions]

    return jnp.array(start_positions),jnp.array(end_positions)
