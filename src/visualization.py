from typing import Tuple
from tqdm import tqdm

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from PIL import Image
import io

from constants import DEFAULT_GIF_FPS

def animate_particles(
        r: jnp.ndarray, 
        theta: jnp.ndarray, 
        wall_history: jnp.ndarray, 
        width: float, 
        height: float, 
        show_arrows: bool=False,
        gif_filename: str="particles.gif"
        ):
    """
    Create an animated GIF of particles with their positions in every frame and optionally display arrows
    representing their headings at each frame.

    Parameters
    ----------
    r : jnp.ndarray
        A 3D array of particle positions with shape (n_frames, n_particles, 2).
    theta : jnp.ndarray
        A 2D array of particle headings with shape (n_frames, n_particles).
    wall_history : jnp.ndarray
        A 4D arary of wall positions with shape (n_frames, 2, W, 2). wall_history[:,0] is 
        start positions and wall_history[:,1] is end positions.
    width : float
        The width of the region to be animated.
    height : float
        The height of the region to be animated.
    show_arrows : bool, optional
        If True, arrows representing particle headings will be displayed in the animation. Default is False.
    gif_filename : str, optional
        The filename for the output animated GIF. Default is "particles.gif".

    Returns
    -------
    None
        The function saves an animated GIF file with the specified filename.
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    wall_history = [np.asarray(one_hist) for one_hist in wall_history]
    
    n_frames, n_particles, _ = r.shape

    # Create an array to store the frames
    frames = []

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-width/2, width/2)
    ax.set_ylim(-height/2, height/2)

    for frame in (pbar := tqdm(range(n_frames))):
        pbar.set_description(f"Processing {frame}")
        
        # Clear the axis
        ax.clear()
        ax.set_xlim(-width/2, width/2)
        ax.set_ylim(-height/2, height/2)

        # Plot the particle positions
        positions = r[frame]
        ax.scatter(positions[:, 0], positions[:, 1],s=1)

        if show_arrows:
            # Calculate the heading vectors
            headings = np.column_stack((np.cos(theta[frame]), np.sin(theta[frame])))
            # Plot the arrows using quiver
            ax.quiver(positions[:, 0], positions[:, 1], headings[:, 0], headings[:, 1], color='red', angles='xy', scale_units='xy', scale=1)
        
        for wall_object in wall_history: # wall_object is a (n_frames, 2, W, 2) Array
            wall_start_list, wall_end_list = wall_object[frame] # each a (W,2) Array
            for w1,w2 in zip(wall_start_list,wall_end_list):
                ax.plot([w1[0],w2[0]], [w1[1],w2[1]],c="k")#, linewidth=5)

        # Save the frame to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        frames.append(buf.getvalue())
        buf.close()

    # Save the frames as an animated GIF
    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    images[0].save(gif_filename, save_all=True, append_images=images[1:], loop=0, duration=1000/DEFAULT_GIF_FPS)