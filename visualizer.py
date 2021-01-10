import argparse
from enum import Enum, auto
from typing import Tuple, List

import numpy as np
from bokeh.io import curdoc
# from bokeh.io import show
from bokeh.models import Panel, Tabs, Slider, ColumnDataSource, Select, Spinner
from bokeh.plotting import figure
from bokeh.layouts import column, row

from db import DBReader

parser = argparse.ArgumentParser()
parser.add_argument('--db_file')
try:
    args = parser.parse_args()
    kwargs = {'db_file': args.db_file}
except Exception as e:
    print(e)
    kwargs = {}

def get_transition(log_id: str, trajectory_index: int, transition_index: int):
    db_reader = DBReader(**kwargs)
    data = db_reader.get_transition_data(log_id, trajectory_index=trajectory_index, transition_index=transition_index)
    return data


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


def step_slider_change_handler(attr, old, new):
    global data_manager, ui_manager
    data_manager.change_transition(new)

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

def create_state_layout(state, data_sources):
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
    return Tabs(tabs=tabs_list)


class DataManager:
    def __init__(self):
        db_reader = DBReader(**kwargs)
        self.log_ids = db_reader.get_logs_ids()
        self.current_log = self.log_ids[0]
        self.trajectory_index = 0
        self.transition_index = 0
        self.trajectories_lengths = db_reader.get_trajectories_lengths(self.current_log)
        self.n_state_elements = db_reader.get_num_of_state_elements(self.current_log)

        self.s, self.a, self.r, self.s2 = get_transition(self.current_log, self.trajectory_index, self.transition_index)

        self.state_elements_sizes = [state_elem.size for state_elem in self.s]

    def change_transition(self, transition_index: int, trajectory_index: int = None, log_id: str = None):
        if trajectory_index is not None:
            self.trajectory_index = trajectory_index
        self.transition_index = transition_index
        is_log_changed = False
        if log_id is not None:
            is_log_changed = log_id != self.current_log
            self.current_log = log_id
        self.s, self.a, self.r, self.s2 = get_transition(self.current_log, self.trajectory_index, self.transition_index)
        if log_id is not None and is_log_changed:
            db_reader = DBReader(**kwargs)
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

        self.states_layout = row(self.state_layout, self.next_state_layout)
        self.sliders_layout = column(self.step_slider, self.trajectory_select, sizing_mode='stretch_width')
        self.layout = column(self.log_select, self.states_layout, self.sliders_layout)

    def update_ui_due_to_log_change(self):
        self._create_states_ui()
        self.states_layout.children = [self.state_layout, self.next_state_layout]

        self.step_slider.value = self.data_manager.transition_index
        self.step_slider.end = self.data_manager.get_current_trajectory_length()-1

    def update_ui_due_to_transition_change(self):
        state_data_dicts = create_state_data_dict_from_state(self.data_manager.s)
        next_state_data_dicts = create_state_data_dict_from_state(self.data_manager.s2)
        for i, element in enumerate(self.data_manager.s):
            self.state_data_sources[i].stream(new_data=state_data_dicts[i],
                                                    rollover=self.data_manager.state_elements_sizes[i])
            self.next_state_data_sources[i].stream(new_data=next_state_data_dicts[i],
                                                         rollover=self.data_manager.state_elements_sizes[i])

    def update_ui_due_to_trajectory_change(self):
        self.update_ui_due_to_transition_change()

        self.step_slider.value = self.data_manager.transition_index
        self.step_slider.end = self.data_manager.get_current_trajectory_length() - 1

    def _generate_step_slider(self):
        print(f'slider end: {self.data_manager.get_current_trajectory_length()}')
        self.step_slider = Slider(start=0, end=self.data_manager.get_current_trajectory_length()-1,
                                  value=self.data_manager.transition_index, step=1, title="Time Step")
        self.step_slider.on_change('value', step_slider_change_handler)

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
        self.state_layout = create_state_layout(self.data_manager.s, self.state_data_sources)
        self.next_state_layout = create_state_layout(self.data_manager.s2, self.next_state_data_sources)

data_manager = DataManager()

ui_manager = UIManager(data_manager=data_manager)

curdoc().add_root(ui_manager.layout)