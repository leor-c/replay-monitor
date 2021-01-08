import argparse
from enum import Enum, auto
from typing import Tuple

import numpy as np
from bokeh.io import curdoc
# from bokeh.io import show
from bokeh.models import Panel, Tabs, Slider, ColumnDataSource
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

def read_metadata():
    db_reader = DBReader(**kwargs)
    log_ids = db_reader.get_logs_ids()

    first_id = log_ids[0]
    trajectories_lengths = db_reader.get_trajectories_lengths(first_id)

    n_state_elements = db_reader.get_num_of_state_elements(first_id)
    return log_ids, trajectories_lengths, n_state_elements

def get_transition(log_id: str, trajectory_index: int, transition_index: int):
    db_reader = DBReader(**kwargs)
    data = db_reader.get_transition_data(log_id, trajectory_index=trajectory_index, transition_index=transition_index)
    return data


def create_state_data_dict_from_state(state: Tuple[np.ndarray]):
    state_data_dicts = []
    for element in state:
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
        state_data_dicts.append(data)
    return state_data_dicts


def step_slider_change_handler(attr, old, new):
    trajectory_index = 0
    s, a, r, s2 = get_transition(log_ids[0], trajectory_index, new)
    # print(f'state={s}, shape={s[0].shape}')
    state_data_dicts = create_state_data_dict_from_state(s)
    next_state_data_dicts = create_state_data_dict_from_state(s2)
    for i, element in enumerate(s):
        state_data_sources[i].stream(new_data=state_data_dicts[i], rollover=state_elements_sizes[i])
        next_state_data_sources[i].stream(new_data=next_state_data_dicts[i], rollover=state_elements_sizes[i])

    print(f'old value={old}, new value={new}. state={s}')


class StateElementPlotType(Enum):
    BAR = auto()
    MATRIX = auto()


def determine_state_element_plot_type(state_elem: np.ndarray) -> StateElementPlotType:
    if len(state_elem.shape) == 1:
        return StateElementPlotType.BAR
    elif len(state_elem.shape) == 2:
        if 1 in state_elem.shape:
            return StateElementPlotType.BAR
        else:
            return StateElementPlotType.MATRIX

def create_state_layout(state, data_sources):
    tabs_list = []
    for i, element in enumerate(state):
        plot_type = determine_state_element_plot_type(element)
        if plot_type == StateElementPlotType.BAR:
            fig = figure(plot_width=600, plot_height=600)
            fig.vbar(x='x', width=0.5, bottom=0, top='state_element', source=data_sources[i])
        elif plot_type == StateElementPlotType.MATRIX:
            range_max = max(element.shape[1], element.shape[0])
            fig = figure(plot_width=600, plot_height=600, x_range=(0, range_max), y_range=(0, range_max))
            fig.image(image='state_element', x=0, y=0, dw=element.shape[1], dh=element.shape[0], palette="Spectral11",
                      source=data_sources[i])
        else:
            continue

        tabs_list.append(Panel(child=fig, title=f"State Element {i}"))
    return Tabs(tabs=tabs_list)

log_ids, trajectories_lengths, n_state_elements = read_metadata()

s, a, r, s2 = get_transition(log_ids[0], 0, 0)

state_elements_sizes = [state_elem.size for state_elem in s]

state_data_sources = [ColumnDataSource(data=data_dict) for data_dict in create_state_data_dict_from_state(s)]
next_state_data_sources = [ColumnDataSource(data=data_dict) for data_dict in create_state_data_dict_from_state(s2)]

slider = Slider(start=0, end=trajectories_lengths[0], value=0, step=1, title="Time Step")
slider.on_change('value', step_slider_change_handler)

# p1 = figure(plot_width=300, plot_height=300)
# p1.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)
# tab1 = Panel(child=p1, title="circle")
#
# p2 = figure(plot_width=300, plot_height=300)
# p2.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=3, color="navy", alpha=0.5)
# tab2 = Panel(child=p2, title="line")
##Tabs(tabs=[tab1, tab2])

state_layout = create_state_layout(s, state_data_sources)
next_state_layout = create_state_layout(s2, next_state_data_sources)

layout = column(row(state_layout, next_state_layout), slider)

curdoc().add_root(layout)