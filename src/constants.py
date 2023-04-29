from typing import Tuple
import jax.numpy as jnp

DEFAULT_DT = 0.01
DEFAULT_TOTAL_TIME = 10.
DEFAULT_V0 = 3.
DEFAULT_TRANSLATION_GAMMA = 1.
DEFAULT_TRANSLATION_DIFFUSION = 1e-1
DEFAULT_ROTATION_GAMMA = 0.1
DEFAULT_ROTATION_DIFFUSION = 1e-4
DEFAULT_OMEGA = 0.
DEFAULT_POISSON_ANGLE_REASSIGNMENT_RATE = 1e-2
DEFAULT_WALL_GAMMA = 100.

DEFAULT_GIF_FPS = 30.
DEFAULT_DO_ANIMATION = True

DEFAULT_PERIODIC_BOUNDARY_SIZE = None

BOUNDING_BOX_STARTS = jnp.array([
    [1.,1.],
    [-1.,1.],
    [-1.,-1.],
    [1.,-1.],
])
BOUNDING_BOX_ENDS = jnp.array([
    [-1.,1.],
    [-1.,-1.],
    [1.,-1.],
    [1.,1.],
])

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
