import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import time

class AnimationBox:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particles = []
        self.arrows = []
        self.walls = []

    def add_particles(self, r, theta):
        for r_i, theta_i in zip(r, theta):
            self.particles.append({"r": r_i, "theta": theta_i, "history": [], "theta_history": []})

    def add_walls(self, wall_points):
        self.walls.append(wall_points)

    def update_particles(self, r, theta, t):
        for i, (r_i, theta_i) in enumerate(zip(r, theta)):
            self.particles[i]["r"] = r_i
            self.particles[i]["theta"] = theta_i
            self.particles[i]["history"].append(r_i)
            self.particles[i]["theta_history"].append(theta_i)

    def animate(self, show_arrows=False, output_file='animation.gif'):
        if not self.particles or not self.particles[0]["history"]:
            print("No particles or frames found. Cannot create animation.")
            return

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        for wall_points in self.walls:
            ax.add_line(Line2D(wall_points[:, 0], wall_points[:, 1], color='k'))

        scatters = [ax.scatter(particle["r"][0], particle["r"][1]) for particle in self.particles]

        if show_arrows:
            for particle in self.particles:
                arrow = patches.Arrow(particle["r"][0], particle["r"][1],
                                      np.cos(particle["theta"]), np.sin(particle["theta"]),
                                      width=0.3, color='C0')
                self.arrows.append(ax.add_patch(arrow))

        def update(frame):
            for i, particle in enumerate(self.particles):
                scatters[i].set_offsets([particle["history"][frame]])
                if show_arrows:
                    self.arrows[i].remove()
                    arrow = patches.Arrow(particle["history"][frame][0], particle["history"][frame][1],
                                          np.cos(particle["theta_history"][frame][0]), np.sin(particle["theta_history"][frame][0]),
                                          width=0.3, color=scatters[i].get_facecolor().tolist()[0])
                    self.arrows[i] = ax.add_patch(arrow)

        num_frames = len(self.particles[0]["history"])
        ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

        ani.save(output_file, writer='imagemagick', fps=15)
        time.sleep(0.1)
        plt.close(fig)