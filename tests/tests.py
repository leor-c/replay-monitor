import os
import unittest
from replay_monitor.monitor import Monitor
import gym
from replay_monitor.db import DBReader


class ComplexEnv(gym.Env):
    def render(self, mode='human'):
        pass

    def __init__(self):
        self.env1 = gym.make('Breakout-v0')
        self.env2 = gym.make('CartPole-v0')

        self.observation_space = gym.spaces.Tuple([self.env1.observation_space, self.env2.observation_space])
        self.action_space = gym.spaces.MultiDiscrete([self.env1.action_space.n, self.env2.action_space.n])
        super().__init__()

    def reset(self):
        return self.env1.reset(), self.env2.reset()

    def step(self, action):
        a1, a2 = action
        res1 = self.env1.step(a1)
        res2 = self.env2.step(a2)
        s, r, done, info = zip(res1, res2)
        return s, sum(r), any(done), info


class MyTestCase(unittest.TestCase):
    def test_db_write(self):
        env = gym.make('CartPole-v0')

        monitor = Monitor(env=env,
                          performance_metrics=[],
                          log_to_db=True)

        monitor.reset()
        for i in range(20):
            s, r, done, info = monitor.step(env.action_space.sample())

            if done:
                break

        db_file = monitor.db_file_path
        del monitor

        reader = DBReader(db_file=db_file)
        log_id = reader.get_logs_ids()[-1]
        self.assertEqual(reader.get_num_of_trajectories(log_id), 1)
        self.assertEqual(reader.get_trajectories_lengths(log_id)[0], i+1)

    def test_db_write_image_state_env(self):
        env = gym.make('Breakout-v0')

        monitor = Monitor(env=env,
                          performance_metrics=[],
                          log_to_db=True)

        n_trajectories = 1
        monitor.reset()
        for i in range(2000):
            action = env.action_space.sample()
            s, r, done, info = monitor.step(action)

            if done:
                monitor.reset()
                n_trajectories += 1

        db_file = monitor.db_file_path
        del monitor

        reader = DBReader(db_file=db_file)
        log_id = reader.get_logs_ids()[-1]
        self.assertEqual(reader.get_num_of_trajectories(log_id), n_trajectories)

    def test_complex_state_env(self):
        env = ComplexEnv()
        print(f'obs space: {env.observation_space}')
        env = Monitor(env=env,
                      performance_metrics=[],
                      log_to_db=True)

        n_trajectories = 1
        env.reset()
        for i in range(300):
            action = env.action_space.sample()
            s, r, done, info = env.step(action)

            if done:
                env.reset()
                n_trajectories += 1

        db_file = env.db_file_path
        del env

        reader = DBReader(db_file=db_file)
        log_id = reader.get_logs_ids()[-1]
        self.assertEqual(reader.get_num_of_trajectories(log_id), n_trajectories)

    def test_start_server(self):
        from replay_monitor.server import _start_server
        _start_server(os.path.join('..' ,'RLMonitorLogs', 'monitor_db.h5'))


if __name__ == '__main__':
    unittest.main()
