import datetime

import gym
import tensorflow as tf
import os
from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy

from replay_monitor.db import DBWriter, PosInTrajectory


class PerformanceMetric(ABC):
    @abstractmethod
    def track(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def value(self):
        """
        Compute and return the value of the performance metric.
        :return:
        """
        pass

    @abstractmethod
    def log_metric_to_tf_summary(self):
        """
        Assume this method will be called with a default TF Summary Writer, output the relevant tf.summary calls.
        :return:
        """
        pass

    @abstractmethod
    def reset_state(self):
        pass

    def new_episode(self):
        """
        Optional. A method that is being called whenever a new episode starts
        :return:
        """
        pass


class PerStepPerformanceMetric(PerformanceMetric):
    """
    This class is intended for metrics that rely on 'per step' data such as observation, action, reward, info, etc.
    """

    @abstractmethod
    def track(self, observation, action, next_observation, reward, done, info, *args, **kwargs):
        pass


def get_space_shape(space: gym.spaces.Space):
    if isinstance(space, gym.spaces.Tuple):
        return tuple(get_space_shape(space) for space in space.spaces)
    else:
        return space.shape


DEFAULT_LOGS_DIR = 'monitor-logs'
DEFAULT_DB_FILENAME = 'monitor_db.h5'


class Monitor(gym.core.Wrapper):
    """
    This class is intended to provide a convenient way of tracking the training / running process of a Reinforcement
    Learning algorithm over an environment that follows the OpenAI's Gym conventions.
    """
    # TODO: Support Vector Envs.

    def __init__(self,
                 env: gym.Env,
                 performance_metrics: List[PerformanceMetric] = None,
                 log_to_db: bool = False,
                 logging_directory: str = DEFAULT_LOGS_DIR,
                 *arg, **kwargs):
        # initialize the directory for the logs:
        self.run_id = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        logging_path = self._generate_logging_path(logging_directory=logging_directory, run_id=self.run_id)

        self.summary_writer = tf.summary.create_file_writer(logging_path)
        self.performance_metrics: List[PerformanceMetric] = performance_metrics \
            if performance_metrics is not None else []
        self.current_observation = None

        self.log_to_db = log_to_db
        self.db_file_path = os.path.join(logging_directory, DEFAULT_DB_FILENAME)
        self._db_writer = DBWriter(
            state_shape=get_space_shape(env.observation_space),
            action_shape=get_space_shape(env.action_space),
            is_action_discrete=isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)),
            db_file=self.db_file_path,
            total_steps_estimate=None
        )
        self._last_observation = None
        self._is_first_observation = False

        super().__init__(env=env)

    def add_performance_metrics(self, performance_metrics: List[PerformanceMetric]):
        self.performance_metrics.extend(performance_metrics)

    def step(self, action):
        next_observation, reward, done, info = self.env.step(action)
        if self.log_to_db:
            # determine position in trajectory:
            self._db_writer.store_transition(
                state=self._last_observation,
                action=action,
                reward=reward,
                next_state=next_observation,
                info=info,
                pos_in_trajectory=self._determine_pos_in_trajectory(done)
            )
        self._is_first_observation = False

        self._track_performance(action=action, next_observation=next_observation, reward=reward, done=done, info=info)
        self.current_observation = deepcopy(next_observation)
        return next_observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)

        self._last_observation = observation
        self._is_first_observation = True

        self.current_observation = deepcopy(observation)
        for metric in self.performance_metrics:
            metric.new_episode()
        return observation

    def log_performance_metrics(self):
        with self.summary_writer.as_default():
            for metric in self.performance_metrics:
                metric.log_metric_to_tf_summary()

    def log_hparams(self, hparams: dict):
        """
        Logs the hyper parameters to a text table in Tensorboard.
        :param hparams: dictionary where the keys are strings with the names of the hyper parameters and the values
        are the corresponding values of the hyper parameters.
        """
        with self.summary_writer.as_default():
            tf.summary.text(name='Hyper Parameters', data=self.generate_hparams_md_table(hparams), step=0)

    def reset_metrics_states(self):
        for metric in self.performance_metrics:
            metric.reset_state()

    def _track_performance(self, action, next_observation, reward, done, info):
        for metric in self.performance_metrics:
            metric.track(self.current_observation,
                         action,
                         next_observation,
                         reward,
                         done,
                         info)

    @staticmethod
    def _generate_logging_path(logging_directory: str, run_id: str):
        logging_path = os.path.join(logging_directory, f'run_{run_id}')
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
        return logging_path

    @staticmethod
    def generate_hparams_md_table(hparams: dict):
        header = '| Hyper Parameter Name | Value | \n | --- | --- | \n'
        lines = [f'| {hp_k} | {hp_v} | \n' for hp_k, hp_v in hparams.items()]
        return header + ' '.join(lines)

    def _determine_pos_in_trajectory(self, done) -> PosInTrajectory:
        if self._is_first_observation:
            if done:
                pos = PosInTrajectory.ONLY
            else:
                pos = PosInTrajectory.FIRST
        else:
            if done:
                pos = PosInTrajectory.LAST
            else:
                pos = PosInTrajectory.MID
        return pos

