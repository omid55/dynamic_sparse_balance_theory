# Omid55
# Date:     16 Oct 2018
# Author:   Omid Askarisichani
# Email:    omid55@cs.ucsb.edu
# General utility module.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Dict
from typing import List
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pk
import networkx as nx
import shelve


def print_dict_pretty(input_dict: Dict) -> None:
    """Prints the input dictionary line by line and key sorted.

    Args:
        input_dict: Dictionary to be printed.

    Returns:
        None

    Raises:
        None
    """
    sorted_keys = sorted(input_dict.keys())
    for key in sorted_keys:
        print('{}: {}'.format(key, input_dict[key]))


def check_required_columns(
        data: pd.DataFrame, columns: List[str]) -> None:
    """Checks whether input dataframe includes all required columns.

    Args:
        input_dict: Dataframe to be checked.

        columns: List of names for columns to be checked in dataframe.

    Returns:
        None

    Raises:
        ValueError: If input data does not include any of required columns.
    """
    missing_columns = list(set(columns) - set(data.columns))
    if missing_columns:
        raise ValueError('Missing required columns: {}.'.format(
            ', '.join(map(str, missing_columns))))


def graph_equals(g1: nx.DiGraph, g2: nx.DiGraph) -> bool:
    """Checks if two graphs are equal.

    TODO(omid55): check for a possible mapping or node names if not same.

    Args:
        g1: First graph to be compared.

        g2: Second graph to be compared.

    Returns:
        Boolean whether g1 equals g2 or not.

    Raises:
        None.
    """
    return g1.nodes() == g2.nodes() and g1.edges() == g2.edges()


def sub_adjacency_matrix(
        adj_matrix: np.ndarray,
        rows: List[int]) -> np.ndarray:
    """Computes a desired subset of given adjacency matrix.

    Args:
        adj_matrix: Given adjacencey matrix.

        rows: List of desired rows and same columns for being in the subgraph.

    Returns:
        Adjancy matrix only including the desired rows and columns.

    Raises:
        None.
    """
    return adj_matrix[np.ix_(rows, rows)]


def swap_nodes_in_matrix(
        matrix: np.ndarray,
        node1: int,
        node2: int) -> np.ndarray:
    """Swaps two nodes in a matrix and return the resulting matrix.

    Args:
        matrix: Input matrix to be swapped.

        node1: First node to be swapped with second one.

        node2: Second node to be swapped with first one.

    Returns:
        Matrix with swapped nodes.

    Raises:
        None.
    """
    result = np.copy(matrix)
    result[:, [node1, node2]] = result[:, [node2, node1]]
    result[[node1, node2], :] = result[[node2, node1], :]
    return result


def make_matrix_row_stochastic(matrix: np.ndarray) -> np.ndarray:
    """Makes the matrix row-stochastic (sum of each row is 1)

    Args:
        matrix: Input matrix.

    Returns:
        Matrix which its rows sum up to 1.

    Raises:
        None.
    """
    return np.nan_to_num(matrix.T / np.sum(matrix, axis=1)).T


def fully_savefig(
        fig_object: matplotlib.figure.Figure,
        file_path: str) -> None:
    """Fully saves the figure in pdf and pkl format for later modification.

    This function saves the figure in a pkl and pdf such that later can
        be loaded and easily be modified.
        To have the figure object, one can add the following line of the code
        to the beginning of their code:
            fig_object = plt.figure()

    Args:
        fig_object: Figure object (computed by "plt.figure()")

        file_path: String file path without file extension.

    Returns:
        None.

    Raises:
        None.
    """
    # Saves as pdf.
    plt.savefig(file_path + '.pdf', dpi=fig_object.dpi)
    # Also saves as pickle.
    with open(file_path + '.pkl', 'wb') as handle:
        pk.dump(fig_object, handle, protocol=pk.HIGHEST_PROTOCOL)


def fully_loadfig(file_path: str) -> matplotlib.figure.Figure:
    """Fully loads the saved figure to be able to be modified.

    It can be easily showed by:
        fig_object.show()

    Args:
        file_path: String file path without file extension.

    Returns:
        Figure object.

    Raises:
        None.
    """
    with open(file_path + '.pkl', 'rb') as handle:
        fig_object = pk.load(handle)
    return fig_object


def save_all_variables_of_current_session(
        locals_: dict,
        file_path: str) -> None:
    """Saves all defined variables in the current session to be used later.

    It works similar to save_all in MATLAB. It is super useful when one is
    trying to save everything in a notebook for later runs of a subset of cells
    of the notebook.

    Args:
        locals_: Just call this as the first parameter ALWAYS: locals()
        file_path: String file path (with extension).

    Returns:
        None.

    Raises:
        None.
    """
    my_shelf = shelve.open(file_path, 'n')
    # for key in dir():
    for key, value in locals_.items():
        if not key.startswith('__') and key != 'self':
            try:
                my_shelf[key] = value  # globals()[key]
            except TypeError:
                print('ERROR shelving: {0}'.format(key))
    my_shelf.close()


def load_all_variables_of_saved_session(
        globals_: dict,
        file_path: str) -> None:
    """Loads all defined variables from a saved session into current session.

    It should be used after running "save_all_variables_of_current_session".

    Args:
        globals_: Just call this as the first parameter ALWAYS: globals()

        file_path: String file path (with extension).

    Returns:
        None.

    Raises:
        None.
    """
    my_shelf = shelve.open(file_path)
    for key in my_shelf:
        globals_[key] = my_shelf[key]
    my_shelf.close()
