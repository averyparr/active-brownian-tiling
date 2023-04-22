import unittest
from run_sim import *
import jax.numpy as jnp

class TestRandomForces(unittest.TestCase):
    def test_random_angles(self):
        DR = 3.
        dt = 0.03

        rand_key = initial_random_key

        samples = []

        for i in range(1000):
            rand_key, noise = angle_noise(rand_key, 10000, DR, dt)
            samples.append(noise)
        samples = jnp.array(samples)
        
        assert jnp.mean(samples) < 0.1
        assert jnp.mean(samples[0] * samples[1]) < 0.1
        second_moment = jnp.mean(jnp.mean(samples**2,axis=0) - jnp.mean(samples,axis=0)**2)
        assert abs(second_moment - 2*DR/dt) < 1.
    def test_random_translation(self):
        DT = 6.
        dt = 0.02

        rand_key = initial_random_key
        samples = []

        for i in range(1000):
            rand_key, noise = translation_noise(rand_key, 10000, DT, dt)
            samples.append(noise)
        samples = jnp.array(samples)

        assert jnp.mean(samples) < 0.1
        assert jnp.mean(samples[0] * samples[1]) < 0.1
        second_moment = jnp.mean(jnp.mean(samples**2,axis=0) - jnp.mean(samples,axis=0),axis=0)
        assert abs(second_moment[0] - 2*DT/dt) < 1.
        assert abs(second_moment[1] - 2*DT/dt) < 1.


if __name__=="__main__":
    unittest.main()