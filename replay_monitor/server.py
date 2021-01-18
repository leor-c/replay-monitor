import os
from enum import Enum, auto
from typing import Tuple

import numpy as np
from bokeh.models import Panel, Tabs, Slider, ColumnDataSource, Select, Spinner, Div
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.server.server import Server

from replay_monitor.db import DBReader

from .monitor import DEFAULT_LOGS_DIR, DEFAULT_DB_FILENAME


def create_state_data_dict_from_state(state: Tuple[np.ndarray]):
    state_data_dicts = []
    for element in state:
        if element.shape[0] == 1:
            element = element[0]
        data = {}
        plot_type = determine_state_element_plot_type(element)
        if plot_type == StateElementPlotType.BAR:
            values = element.flatten()
            data['x'] = list(range(len(values)))
            data['state_element'] = values
            # data['state_x_range_1d'] = range(element.shape[0])
        elif plot_type == StateElementPlotType.MATRIX:
            data['state_element'] = [element]
            # data['state_x_range_2d'] = element.shape[1]
        elif plot_type == StateElementPlotType.COLOR_IMAGE:
            assert element.shape[2] == 3  # Assume color channel is last!
            xdim, ydim = element.shape[0:2]
            img = np.empty((xdim, ydim), dtype=np.uint32)
            view = img.view(dtype=np.uint8).reshape((xdim, ydim, 4))
            view[:, :, :-1] = np.flipud(element)
            view[:, :, -1] = 255
            data['state_element'] = [img]
        state_data_dicts.append(data)
    return state_data_dicts


def create_rewards_data_dict(rewards: np.ndarray):
    return {
        'rewards': rewards,
        'x': list(range(len(data_manager.trajectory_rewards)))
    }


def step_slider_change_handler(attr, old, new):
    global data_manager, ui_manager
    data_manager.change_transition(new)

    ui_manager.step_spinner.value = new
    ui_manager.step_slider.value = new
    ui_manager.update_ui_due_to_transition_change()


def trajectory_changed_handler(attr, old, new):
    global data_manager, ui_manager
    data_manager.change_transition(0, new)

    ui_manager.update_ui_due_to_trajectory_change()


def log_select_change_handler(attr, old, new):
    global data_manager, ui_manager
    data_manager.change_transition(0, 0, new)
    ui_manager.update_ui_due_to_log_change()


class StateElementPlotType(Enum):
    BAR = auto()
    MATRIX = auto()
    COLOR_IMAGE = auto()


def determine_state_element_plot_type(state_elem: np.ndarray) -> StateElementPlotType:
    if len(state_elem.shape) > 1 and state_elem.shape[0] == 1:
        state_elem = state_elem[0]

    if len(state_elem.shape) == 1:
        return StateElementPlotType.BAR
    elif len(state_elem.shape) == 2:
        if 1 in state_elem.shape:
            return StateElementPlotType.BAR
        else:
            return StateElementPlotType.MATRIX
    elif len(state_elem.shape) == 3 and state_elem.shape[2] == 3:
        return StateElementPlotType.COLOR_IMAGE

def create_state_layout(state, data_sources, title):
    tabs_list = []
    for i, element in enumerate(state):
        if len(element.shape) > 1 and element.shape[0] == 1:
            element = element[0]

        plot_type = determine_state_element_plot_type(element)
        if plot_type == StateElementPlotType.BAR:
            fig = figure(plot_width=600, plot_height=600)
            fig.vbar(x='x', width=0.5, bottom=0, top='state_element', source=data_sources[i])
        elif plot_type == StateElementPlotType.MATRIX:
            range_max = max(element.shape[1], element.shape[0])
            fig = figure(plot_width=600, plot_height=600, x_range=(0, range_max), y_range=(0, range_max))
            fig.image(image='state_element', x=0, y=0, dw=element.shape[1], dh=element.shape[0], palette="Spectral11",
                      source=data_sources[i])
        elif plot_type == StateElementPlotType.COLOR_IMAGE:
            range_max = max(element.shape[1], element.shape[0])
            fig = figure(plot_width=600, plot_height=600, x_range=(0, range_max), y_range=(0, range_max))
            fig.image_rgba(image='state_element', x=0, y=0, dw=element.shape[1], dh=element.shape[0],
                           source=data_sources[i])
        else:
            continue

        tabs_list.append(Panel(child=fig, title=f"State Element {i}"))
    state_layout = column(
        Div(text=f"<h2>{title}</h2>"),
        Tabs(tabs=tabs_list)
    )
    return state_layout


class DataManager:
    def __init__(self, db_file_path: str):
        self.db_file_path = db_file_path
        db_reader = DBReader(db_file=db_file_path)
        self.log_ids = db_reader.get_logs_ids()
        self.current_log = self.log_ids[0]
        self.trajectory_index = 0
        self.transition_index = 0
        self.trajectories_lengths = db_reader.get_trajectories_lengths(self.current_log)
        self.n_state_elements = db_reader.get_num_of_state_elements(self.current_log)

        self.trajectory_rewards = db_reader.get_trajectory_rewards(log_id=self.current_log,
                                                                   trajectory_index=self.trajectory_index)

        self.s, self.a, self.r, self.s2 = self.get_transition(self.current_log, self.trajectory_index,
                                                              self.transition_index)

        self.state_elements_sizes = [state_elem.size for state_elem in self.s]

    def get_transition(self, log_id: str, trajectory_index: int, transition_index: int):
        db_reader = DBReader(db_file=self.db_file_path)
        data = db_reader.get_transition_data(log_id, trajectory_index=trajectory_index,
                                             transition_index=transition_index)
        return data

    def change_transition(self, transition_index: int, trajectory_index: int = None, log_id: str = None):
        is_log_changed = False
        if log_id is not None:
            is_log_changed = log_id != self.current_log
            self.current_log = log_id

        if trajectory_index is not None:
            is_different_trajectory = self.trajectory_index != trajectory_index
            self.trajectory_index = trajectory_index
            if is_different_trajectory or is_log_changed:
                db_reader = DBReader(db_file=self.db_file_path)
                self.trajectory_rewards = db_reader.get_trajectory_rewards(log_id=self.current_log,
                                                                           trajectory_index=self.trajectory_index)

        self.transition_index = transition_index

        self.s, self.a, self.r, self.s2 = self.get_transition(self.current_log, self.trajectory_index,
                                                              self.transition_index)

        if log_id is not None and is_log_changed:
            db_reader = DBReader(db_file=self.db_file_path)
            self.trajectories_lengths = db_reader.get_trajectories_lengths(self.current_log)
            self.n_state_elements = db_reader.get_num_of_state_elements(self.current_log)
            self.state_elements_sizes = [state_elem.size for state_elem in self.s]

    def get_current_trajectory_length(self):
        return self.trajectories_lengths[self.trajectory_index]


class UIManager:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

        self.log_select = Select(title="Choose a log:", value=data_manager.current_log, options=data_manager.log_ids)
        self.log_select.on_change('value', log_select_change_handler)

        self.step_slider = None
        self._generate_step_slider()

        self.trajectory_select = None
        self._generate_trajectory_select()

        self.state_data_sources = None
        self.next_state_data_sources = None
        self.state_layout = None
        self.next_state_layout = None
        self.states_layout = None
        self._create_states_ui()

        self.rewards_data_source = ColumnDataSource(data=create_rewards_data_dict(self.data_manager.trajectory_rewards))
        self.trajectory_rewards_fig = None
        self._create_trajectory_rewards_ui()

        self.textual_action_reward = Div(text=self._generate_textual_action_reward_str())

        self.states_layout = row(self.state_layout, self.next_state_layout)
        self.sliders_layout = column(row(self.step_slider, self.step_spinner), self.trajectory_select, sizing_mode='stretch_width')
        self.layout = column(self.log_select,
                             self.trajectory_rewards_fig,
                             self.textual_action_reward,
                             self.states_layout,
                             self.sliders_layout)

    def update_ui_due_to_log_change(self):
        self._create_states_ui()
        self.states_layout.children = [self.state_layout, self.next_state_layout]

        self.rewards_data_source.stream(new_data=create_rewards_data_dict(self.data_manager.trajectory_rewards),
                                        rollover=len(self.data_manager.trajectory_rewards))

        self.trajectory_select.high = len(self.data_manager.trajectories_lengths)-1
        self.trajectory_select.value = self.data_manager.trajectory_index

        self.step_slider.value = self.data_manager.transition_index
        self.step_slider.end = self.data_manager.get_current_trajectory_length()-1
        self.step_spinner.high = self.step_slider.end
        self.step_spinner.value = self.step_slider.value

    def update_ui_due_to_transition_change(self):
        state_data_dicts = create_state_data_dict_from_state(self.data_manager.s)
        next_state_data_dicts = create_state_data_dict_from_state(self.data_manager.s2)
        for i, element in enumerate(self.data_manager.s):
            self.state_data_sources[i].stream(new_data=state_data_dicts[i],
                                                    rollover=self.data_manager.state_elements_sizes[i])
            self.next_state_data_sources[i].stream(new_data=next_state_data_dicts[i],
                                                         rollover=self.data_manager.state_elements_sizes[i])

        self.textual_action_reward.text = self._generate_textual_action_reward_str()

    def update_ui_due_to_trajectory_change(self):
        self.update_ui_due_to_transition_change()

        self.rewards_data_source.stream(new_data=create_rewards_data_dict(self.data_manager.trajectory_rewards),
                                        rollover=len(self.data_manager.trajectory_rewards))

        self.step_slider.value = self.data_manager.transition_index
        self.step_slider.end = self.data_manager.get_current_trajectory_length() - 1
        self.step_spinner.high = self.step_slider.end
        self.step_spinner.value = self.step_slider.value

    def _generate_step_slider(self):
        self.step_slider = Slider(start=0, end=self.data_manager.get_current_trajectory_length()-1,
                                  value=self.data_manager.transition_index, step=1, title="Time Step",
                                  sizing_mode='stretch_width')
        self.step_spinner = Spinner(low=0, high=self.step_slider.end, step=1, value=self.step_slider.value, width=100)
        self.step_slider.on_change('value', step_slider_change_handler)
        self.step_spinner.on_change('value', step_slider_change_handler)

    def _generate_trajectory_select(self):
        self.trajectory_select = Spinner(
            title="Choose a trajectory:",
            low=0,
            high=len(self.data_manager.trajectories_lengths)-1,
            step=1,
            value=self.data_manager.trajectory_index,
        )
        self.trajectory_select.on_change('value', trajectory_changed_handler)

    def _create_states_ui(self):
        self.state_data_sources = [ColumnDataSource(data=data_dict)
                                   for data_dict in create_state_data_dict_from_state(self.data_manager.s)]
        self.next_state_data_sources = [ColumnDataSource(data=data_dict)
                                        for data_dict in create_state_data_dict_from_state(self.data_manager.s2)]
        self.state_layout = create_state_layout(self.data_manager.s, self.state_data_sources, 'State')
        self.next_state_layout = create_state_layout(self.data_manager.s2, self.next_state_data_sources, 'Next State')

    def _create_trajectory_rewards_ui(self):
        fig = figure(plot_width=600, plot_height=300, sizing_mode='stretch_width')
        fig.circle(x='x', y='rewards', source=self.rewards_data_source, name='trajectory_rewards_plot')
        self.trajectory_rewards_fig = fig
        return fig

    def _generate_textual_action_reward_str(self):
        text = f'<h4>Reward: {self.data_manager.r[0]}<br>Action: {self.data_manager.a[0]}</h4>'
        return text


db_file_path = os.path.join(DEFAULT_LOGS_DIR, DEFAULT_DB_FILENAME)
data_manager = None
ui_manager = None


def start_app(doc):
    global data_manager, ui_manager, db_file_path
    data_manager = DataManager(db_file_path)
    ui_manager = UIManager(data_manager=data_manager)
    doc.add_root(ui_manager.layout)


def _start_server(db_path: str = None):
    global db_file_path

    if db_path is not None:
        db_file_path = db_path

    server = Server({'/': start_app}, num_procs=1)
    server.start()

    print('Opening Bokeh application on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def start_server():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', required=False)
    args = parser.parse_args()

    _start_server(args.db_path)


if __name__ == '__main__':
    start_server()