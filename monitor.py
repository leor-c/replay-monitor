import datetime

import gym
import tensorflow as tf
import os
from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy


class PerformanceMetric(ABC):
    @abstractmethod
    def track(self, *args, **kwargs):
        pass

    @abstractmethod
    @property
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


class RLMonitor(gym.core.Wrapper):
    """
    This class is intended to provide a convenient way of tracking the training / running process of a Reinforcement
    Learning algorithm over an environment that follows the OpenAI's Gym conventions.
    """
    def __init__(self,
                 env: gym.Env,
                 performance_metrics: List[PerformanceMetric],
                 logging_directory: str = 'RLMonitorLogs',
                 *arg, **kwargs):
        # initialize the directory for the logs:
        logging_path = self._generate_logging_path(logging_directory=logging_directory)

        self.summary_writer = tf.summary.create_file_writer(logging_path)
        self.performance_metrics: List[PerformanceMetric] = performance_metrics
        self.current_observation = None

        super().__init__(env=env)

    def add_performance_metrics(self, performance_metrics: List[PerformanceMetric]):
        self.performance_metrics.extend(performance_metrics)

    def step(self, action):
        next_observation, reward, done, info = self.env.step(action)
        self._track_performance(action=action, next_observation=next_observation, reward=reward, done=done, info=info)
        self.current_observation = deepcopy(next_observation)
        return next_observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.current_observation = deepcopy(observation)
        for metric in self.performance_metrics:
            metric.new_episode()
        return observation

    def log_performance_metrics(self):
        with self.summary_writer.as_default():
            for metric in self.performance_metrics:
                metric.log_metric_to_tf_summary()

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
    def _generate_logging_path(logging_directory: str):
        current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        logging_path = os.path.join(logging_directory, f'run_{current_time}')
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
        return logging_path

