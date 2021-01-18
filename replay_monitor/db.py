from dataclasses import dataclass

import tables as tb
import time
import numpy as np
from typing import Dict, Tuple, Union, List
import os
from enum import Enum, auto


@dataclass
class StateData:
    observation: Tuple[np.ndarray]
    additional_data: Dict[str, np.ndarray]
    terminal: bool


LOGS_GROUP = 'logs'
LOG_ID_GROUP = 'log_id'
TRAJECTORIES_ARRAY = 'trajectories_start_idx'
LOG_STATES_GROUP = 'states'
STATE_ELEM = 'state_elem_'
LOG_ACTIONS_ARRAY = 'actions'
LOG_REWARDS_ARRAY = 'rewards'


DB_STRUCTURE = {
    LOGS_GROUP: {
        LOG_ID_GROUP: {

            LOG_STATES_GROUP: {
                STATE_ELEM: np.ndarray
            }
        }
    }
}


class PosInTrajectory(Enum):
    FIRST = auto()
    LAST = auto()
    MID = auto()
    ONLY = auto()


class DBWriter:
    """
    Intended for easy to use interface with the DB to perform the necessary DB write transactions.
    An object of this class records all interactions as a single run.
    To record different runs, use different objects.

    * avoid messy function signatures by avoiding to require configurations in function params

    """
    def __init__(
            self,
            state_shape: Union[Tuple[int], Tuple[Tuple[int]]],
            action_shape: Tuple[int],
            is_action_discrete: bool = True,
            db_file: str = 'monitor_db.h5',
            total_steps_estimate: int = None
    ):
        self.state_shape = state_shape
        if isinstance(self.state_shape[0], int):
            self.state_shape = (self.state_shape,)
        self.action_shape = action_shape
        self.is_action_discrete = is_action_discrete
        self.total_steps_estimate = total_steps_estimate

        self.db_file = db_file
        open_mode = 'a' if os.path.exists(self.db_file) else 'w'
        self.db = tb.open_file(self.db_file, mode=open_mode, title="RL Monitor DB")

        self.run_id = self._generate_run_id()
        self._initialize_monitor_run()

        self._start_new_trajectory_flag = False

    def __del__(self):
        self.db.close()

    def store_transition(
            self,
            state: Union[np.ndarray, Tuple[np.ndarray]],
            action,
            reward: float,
            next_state: Union[np.ndarray, Tuple[np.ndarray]],
            info = None,
            pos_in_trajectory: PosInTrajectory = PosInTrajectory.MID
    ):
        # If this state begins a new trajectory, store that information as well:
        if pos_in_trajectory == PosInTrajectory.FIRST or pos_in_trajectory == PosInTrajectory.ONLY:
            if self._start_new_trajectory_flag:
                self._begin_new_trajectory()
            self._store_state(state)
        self._start_new_trajectory_flag = False

        # self._store_state(state)
        self._store_action(action)
        self._store_reward(reward)
        self._store_state(next_state)

        if pos_in_trajectory == PosInTrajectory.LAST or pos_in_trajectory == PosInTrajectory.ONLY:
            self._start_new_trajectory_flag = True

    def _store_state(self, env_state: Union[np.ndarray, Tuple[np.ndarray]]):
        # wrap single obs states in a tuple for uniform handling:
        if not isinstance(env_state, tuple):
            assert isinstance(env_state, np.ndarray)
            env_state = (env_state,)

        states_group_node = self.db.get_node(f'/{LOGS_GROUP}/{self.run_id}/{LOG_STATES_GROUP}')
        assert states_group_node._v_nchildren == len(env_state)

        for i, state_elem in enumerate(env_state):
            state_elem_arr = self.db.get_node(f'/{LOGS_GROUP}/{self.run_id}/{LOG_STATES_GROUP}/{STATE_ELEM}{i}')
            state_elem_arr.append(state_elem)

    def _store_action(self, agent_action):
        if isinstance(agent_action, (int, float)):
            agent_action = np.array([agent_action])
        actions_arr = self.db.get_node(f'/{LOGS_GROUP}/{self.run_id}/{LOG_ACTIONS_ARRAY}')
        actions_arr.append(agent_action)

    def _store_reward(self, reward: float):
        rewards_arr = self.db.get_node(f'/{LOGS_GROUP}/{self.run_id}/{LOG_REWARDS_ARRAY}')
        rewards_arr.append(np.array([reward]))

    def _begin_new_trajectory(self):
        n_states = self.db.get_node(f'/{LOGS_GROUP}/{self.run_id}/{LOG_STATES_GROUP}/{STATE_ELEM}{0}').nrows
        n_transitions = self.db.get_node(f'/{LOGS_GROUP}/{self.run_id}/{LOG_REWARDS_ARRAY}').nrows
        self.db.get_node(f'/{LOGS_GROUP}/{self.run_id}/{TRAJECTORIES_ARRAY}').append((n_states, n_transitions))

    @staticmethod
    def _generate_run_id() -> str:
        return LOG_ID_GROUP + str(time.time()).replace('.', '_')

    def _initialize_monitor_run(self):
        root_children = [v._v_name for v in self.db.get_node('/')]
        if LOGS_GROUP not in root_children:
            self.db.create_group('/', LOGS_GROUP)
        log_root = self.db.create_group(f'/{LOGS_GROUP}', f"{self.run_id}")

        # create inner structure to store the data of the current run:
        self.db.create_vlarray(
            log_root,
            TRAJECTORIES_ARRAY,
            tb.UInt64Atom(shape=(2,), dflt=0),
            'An array of the indexes of the first state of each trajectory'
        )

        log_states_group = self.db.create_group(log_root, LOG_STATES_GROUP)
        for i, state_elem_shape in enumerate(self.state_shape):
            self.db.create_vlarray(
                log_states_group,
                f"{STATE_ELEM}{i}",
                tb.Atom.from_dtype(np.dtype((np.float32, state_elem_shape))),
                f'Element #{i} of the observation (state)',
                expectedrows=self.total_steps_estimate
            )

        action_atom = tb.UInt64Atom(shape=self.action_shape, dflt=0) if self.is_action_discrete else \
            tb.Atom.from_dtype(np.dtype((np.float32, self.action_shape)))
        self.db.create_vlarray(
            log_root,
            LOG_ACTIONS_ARRAY,
            atom=action_atom
        )

        self.db.create_vlarray(
            log_root,
            LOG_REWARDS_ARRAY,
            atom=tb.Float32Atom(shape=(), dflt=0.0)
        )


class DBReader:
    def __init__(self, db_file: str = 'monitor_db.h5'):
        self.db_file = db_file

    @property
    def db(self):
        return tb.open_file(self.db_file, mode='r')

    def get_logs_ids(self) -> List[str]:
        with self.db as db:
            logs_node = db.get_node(f'/{LOGS_GROUP}')
            return [v._v_name for v in logs_node._f_list_nodes()]

    def get_num_of_trajectories(self, log_id: str):
        if self._get_num_of_states(log_id=log_id) == 0:
            return 0

        with self.db as db:
            n_traj_rows = db.get_node(f'/{LOGS_GROUP}/{log_id}/{TRAJECTORIES_ARRAY}').nrows
        return n_traj_rows + 1

    def get_trajectories_lengths(self, log_id: str) -> List[int]:
        """
        Useful to know how many trajectories are there, and their lengths.
        Here, trajectory length is defined as the number of transitions (= number of actions).
        :return:
        """
        n_states = self._get_num_of_states(log_id=log_id)
        if n_states == 0:
            return []

        trajectories_start_indices = self._get_log_trajectories(log_id=log_id)
        trajectories_lengths = []
        for i in range(len(trajectories_start_indices)-1):
            trajectories_lengths.append((trajectories_start_indices[i+1][0]-1) - trajectories_start_indices[i][0])
        trajectories_lengths.append((n_states-1) - trajectories_start_indices[-1][0])

        return trajectories_lengths

    def get_transition_data(self, log_id: str, trajectory_index: int, transition_index: int):
        trajectories_start_indices = self._get_log_trajectories(log_id=log_id)

        with self.db as db:
            states_group_node = db.get_node(f'/{LOGS_GROUP}/{log_id}/{LOG_STATES_GROUP}')
            state_elements_nodes = states_group_node._f_list_nodes()
            state_idx = trajectories_start_indices[trajectory_index][0] + transition_index
            state = tuple(s[state_idx] for s in state_elements_nodes)
            next_state = tuple(s[state_idx+1] for s in state_elements_nodes)

            action_idx = trajectories_start_indices[trajectory_index][1] + transition_index
            action = db.get_node(f'/{LOGS_GROUP}/{log_id}/{LOG_ACTIONS_ARRAY}')[action_idx]
            reward = db.get_node(f'/{LOGS_GROUP}/{log_id}/{LOG_REWARDS_ARRAY}')[action_idx]

        return state, action, reward, next_state

    def get_trajectory_rewards(self, log_id: str, trajectory_index: int):
        trajectories_start_indices = self._get_log_trajectories(log_id=log_id)
        trajectories_lengths = self.get_trajectories_lengths(log_id=log_id)
        if len(trajectories_lengths) == 0:
            return None
        with self.db as db:
            start_index = trajectories_start_indices[trajectory_index][1]
            end_index = trajectories_start_indices[trajectory_index][1] + trajectories_lengths[trajectory_index]
            rewards = db.get_node(f'/{LOGS_GROUP}/{log_id}/{LOG_REWARDS_ARRAY}')[start_index:end_index]
            return rewards

    def get_num_of_state_elements(self, log_id: str):
        with self.db as db:
            return db.get_node(f'/{LOGS_GROUP}/{log_id}/{LOG_STATES_GROUP}')._v_nchildren

    def _get_num_of_states(self, log_id: str):
        with self.db as db:
            states_group_node = db.get_node(f'/{LOGS_GROUP}/{log_id}/{LOG_STATES_GROUP}')
            state_elements_nodes = states_group_node._f_list_nodes()
            if len(state_elements_nodes) == 0:
                return 0

            return db.get_node(f'/{LOGS_GROUP}/{log_id}/{LOG_STATES_GROUP}/{STATE_ELEM}{0}').nrows

    def _get_log_trajectories(self, log_id: str) -> List[Tuple[int, int]]:
        if self._get_num_of_states(log_id=log_id) == 0:
            return []

        trajectories = [(0, 0)]
        with self.db as db:
            stored_data = db.get_node(f'/{LOGS_GROUP}/{log_id}/{TRAJECTORIES_ARRAY}').read()
            if len(stored_data) > 0:
                trajectories.extend(np.array(stored_data, dtype=int).squeeze(1))
        return trajectories

