# Omid55
# Date:     16 Oct 2018
# Author:   Omid Askarisichani
# Email:    omid55@cs.ucsb.edu
# General utility module.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pk
import networkx as nx
import shelve
# import enforce
from typing import Dict
from typing import List


# @enforce.runtime_validation
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


# @enforce.runtime_validation
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


# @enforce.runtime_validation
def graph_equals(
        g1: nx.DiGraph,
        g2: nx.DiGraph,
        weight_column_name: str = 'weight') -> bool:
    """Checks if two graphs are equal.

    If weight_column_name is None, then it does not check weight values.

    Args:
        g1: First graph to be compared.

        g2: Second graph to be compared.

        weight_column_name: The name of weight column.

    Returns:
        Boolean whether g1 equals g2 or not.

    Raises:
        None.
    """
    if g1.nodes() != g2.nodes():
        return False
    if g1.edges() != g2.edges():
        return False
    if weight_column_name:
        for edge in g1.edges():
            w1 = g1.get_edge_data(edge[0], edge[1])[weight_column_name]
            w2 = g2.get_edge_data(edge[0], edge[1])[weight_column_name]
            if w1 != w2:
                return False
    return True


# @enforce.runtime_validation
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


# @enforce.runtime_validation
def swap_nodes_in_matrix(
        matrix: np.ndarray,
        node1: int,
        node2: int,
        inplace: bool = False) -> np.ndarray:
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
    if not inplace:
        modified_matrix = np.copy(matrix)
    else:
        modified_matrix = matrix
    modified_matrix[:, [node1, node2]] = modified_matrix[:, [node2, node1]]
    modified_matrix[[node1, node2], :] = modified_matrix[[node2, node1], :]
    return modified_matrix


# @enforce.runtime_validation
def make_matrix_row_stochastic(
        matrix: np.ndarray) -> np.ndarray:
    """Makes the matrix row-stochastic (sum of each row is 1)

    Args:
        matrix: Input matrix.

    Returns:
        Matrix which its rows sum up to 1.

    Raises:
        None.
    """
    # matrix += eps
    # return np.nan_to_num(matrix.T / np.sum(matrix, axis=1)).T
    matrix = np.array(matrix)  # To make sure it is numpy array and not matrix.
    n, _ = matrix.shape
    for i in range(n):
        row_pointer = matrix[i, :]
        if any(row_pointer < 0):
            row_pointer -= np.min(row_pointer)
        if np.sum(row_pointer) == 0:
            row_pointer += 0.01
    return np.nan_to_num(matrix.T / np.sum(matrix, axis=1)).T


# @enforce.runtime_validation
def save_figure(
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
    fig_object.savefig(file_path + '.pdf', dpi=fig_object.dpi)
    # Also saves as pickle.
    with open(file_path + '.pkl', 'wb') as handle:
        pk.dump(fig_object, handle, protocol=pk.HIGHEST_PROTOCOL)


# @enforce.runtime_validation
def load_figure(file_path: str) -> matplotlib.figure.Figure:
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


# @enforce.runtime_validation
def save_all_variables_of_current_session(
        locals_: dict,
        file_path: str,
        verbose: bool = False) -> None:
    """Saves all defined variables in the current session to be used later.

    It works similar to save_all in MATLAB. It is super useful when one is
    trying to save everything in a notebook for later runs of a subset of cells
    of the notebook.

    Args:
        locals_: Just call this as the first parameter ALWAYS: locals()

        file_path: String file path (with extension).

        verbose: Whether to print the name of variables it is saving.

    Returns:
        None.

    Raises:
        None.
    """
    my_shelf = shelve.open(file_path, 'n')
    # for key in dir():
    for key, value in locals_.items():
        if (not key.startswith('__') and
            not key.startswith('_') and
                key not in ['self', 'exit', 'Out', 'quit', 'imread'] and
                str(type(value)) not in [
                    "<class 'module'>", "<class 'method'>"]):
            try:
                if verbose:
                    print('key: ', key)
                my_shelf[key] = value
            except TypeError:
                print('Just this variable was not saved: {0}'.format(key))
    my_shelf.close()


# @enforce.runtime_validation
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
        try:
            globals_[key] = my_shelf[key]
        except AttributeError:
            print('Just this variable was not loaded: ', key)
    my_shelf.close()


def swap_two_elements_in_matrix(
        matrix: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        inplace: bool = True) -> np.ndarray:
    """Swaps the content of two given elements from the matrix.

    Args:

    Returns:

    Raises:
        ValueError: If any of coordinates did not exist.
    """
    n, m = matrix.shape
    if ((x1 < 0 or x1 >= n) or
            (x2 < 0 or x2 >= n) or
            (y1 < 0 or y1 >= m) or
            (y2 < 0 or y2 >= m)):
        raise ValueError(
            'Given coordinates do not fall into matrix dimensions.'
            ' Matrix size: ({}, {}), Coordinates: ({}, {}), ({}, {}).'.format(
                n, m, x1, y1, x2, y2))
    if not inplace:
        modified_matrix = matrix.copy()
    else:
        modified_matrix = matrix
    first_element_content = modified_matrix[x1, y1]
    modified_matrix[x1, y1] = modified_matrix[x2, y2]
    modified_matrix[x2, y2] = first_element_content
    return modified_matrix


# @enforce.runtime_validation
def dgraph2adjacency(dgraph: nx.DiGraph) -> np.ndarray:
    """Gets the dense adjancency matrix from the graph.

    Args:
        dgraph: Directed graph to compute its adjancency matrix.

    Returns:
        Adjacency matrix of the given dgraph in dense format (np.array(n * n)).

    Raises:
        None.
    """
    return np.array(nx.adjacency_matrix(dgraph).todense())


# @enforce.runtime_validation
def adjacency2digraph(
        adj_matrix: np.ndarray,
        similar_this_dgraph: nx.DiGraph = None) -> nx.DiGraph:
    """Converts the adjacency matrix to directed graph.

    If similar_this_graph is given, then the final directed graph has the same
    node labeling as the given graph has.
    Using dgraph2adjacency and then adjacency2digraph for the same dgraph is
    very practical. Example:
        adj = dgraph2adjacency(dgraph)
        # Then modify adj as wish
        new_dgraph = adjacency2digraph(adj, dgraph)
        # Now new_dgraph has the same node labels as dgraph has before.

    Args:
        adj_matrix: Squared adjancency matrix.

    Returns:
        Directed graph with the adj_matrix and same node names as given dgraph.

    Raises:
        ValueError: If adj_matrix was not squared.
    """
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError('Adjacency matrix is not squared.')

    if similar_this_dgraph:
        node_mapping = {
            i: list(similar_this_dgraph.nodes())[i]
            for i in range(similar_this_dgraph.number_of_nodes())}
        return _adjacency2digraph_with_given_mapping(
            adj_matrix=adj_matrix, node_mapping=node_mapping)
    return _adjacency2digraph_with_given_mapping(adj_matrix=adj_matrix)


# @enforce.runtime_validation
def _adjacency2digraph_with_given_mapping(
        adj_matrix: np.ndarray,
        node_mapping: Dict = None) -> nx.DiGraph:
    """Converts the adjacency matrix to directed graph.

    Args:
        adj_matrix: Squared adjancency matrix.

        node_mapping: Dictionary for every node and their current and new name.

    Returns:
        Directed graph with the adj_matrix and same node names as given dgraph.

    Raises:
        ValueError: If adj_matrix was not squared.
    """
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError('Adjacency matrix is not squared.')
    new_dgrpah = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph())
    if node_mapping:
        return nx.relabel_nodes(new_dgrpah, mapping=node_mapping)
    return new_dgrpah


# @enforce.runtime_validation
def save_it(obj: object, file_path: str, verbose: bool = False) -> None:
    """Saves the input object in the given file path.

    Args:
        file_path: String file path (with extension).

        verbose: Whether to print information about saving successfully or not.

    Returns:
        None.

    Raises:
        None.
    """
    with open(file_path, 'wb') as handle:
        pk.dump(obj, handle, protocol=pk.HIGHEST_PROTOCOL)
    if verbose:
        print('{} is successfully saved.'.format(file_path))


# @enforce.runtime_validation
def load_it(file_path: str, verbose: bool = False) -> object:
    """Loads from the given file path a saved object.

    Args:
        file_path: String file path (with extension).

        verbose: Whether to print info about loading successfully or not.

    Returns:
        The loaded object.

    Raises:
        None.
    """
    obj = None
    with open(file_path, 'rb') as handle:
        obj = pk.load(handle)
    if verbose:
        print('{} is successfully loaded.'.format(file_path))
    return obj


# @enforce.runtime_validation
def plot_box_plot_for_transitions(
        matrix: np.ndarray,
        balanced_ones: np.ndarray,
        with_labels: bool = True,
        fname: str = '',
        ftitle: str = '') -> None:
    """Plots a boxplot for transitoins of a set of balanced/unbalanced states.

    Args:
        matrix: A stochastic transition matrix.

        balanced_ones: Array of boolean of which state is balanced or not.

        fname: File name which if is given, this function saves the figure as.

        ftitle: Figure title if given.

    Returns:
        None.

    Raises:
        ValueError: When the length of matrix and balanced_ones does not match.
    """
    if len(matrix) != len(balanced_ones):
        raise ValueError(
            'Matrix and balanced states should have the same length: '
            'len(matrix): {}, len(balanced_ones): {}.'.format(
                len(matrix), len(balanced_ones)))

    # Computes the transitions.
    probs1 = np.sum(matrix[balanced_ones, :][:, balanced_ones], axis=1)
    probs2 = np.sum(matrix[~balanced_ones, :][:, balanced_ones], axis=1)
    probs3 = np.sum(matrix[~balanced_ones, :][:, ~balanced_ones], axis=1)
    probs4 = np.sum(matrix[balanced_ones, :][:, ~balanced_ones], axis=1)

    # Draws the boxplot.
    labels = None
    if with_labels:
        labels = (
            [r'balanced $\rightarrow$ balanced',
             r'unbalanced $\rightarrow$ balanced',
             r'unbalanced $\rightarrow$ unbalanced',
             r'balanced $\rightarrow$ unbalanced'])
    f = plt.figure()
    bp = plt.boxplot(
        [np.array(probs1),
         np.array(probs2),
         np.array(probs3),
         np.array(probs4)],
        labels=labels,
        vert=False,
        showfliers=False)
    plt.xlabel('Transition proability')
    # Default values:
    #   whis=1.5
    if ftitle:
        plt.title(ftitle)

    # Makes the linewidth larger.
    for box in bp['boxes']:
        # change outline color
        box.set(linewidth=2)
    # Changes the color and linewidth of the whiskers.
    for whisker in bp['whiskers']:
        whisker.set(linewidth=2)
    # Changes the color and linewidth of the caps.
    for cap in bp['caps']:
        cap.set(linewidth=2)
    # Changes color and linewidth of the medians.
    for median in bp['medians']:
        median.set(linewidth=2)

    # If filename is given then saves the file.
    if fname:
        f.savefig('Figures/' + fname+'.pdf', bbox_inches='tight')
        f.savefig('Figures/' + fname+'.png', bbox_inches='tight')


def draw_from_empirical_distribution(
    data_points: np.ndarray,
    nbins: int = 10) -> float:
    """Draws a sample from the empricial distribution of the given data points.

    Args:
        data_points: Array of one dimensional data points.

        nbins: Number of bins to consider for empirical distribution.

    Returns:
        A drawn sample from the same emprical distribution of data points.

    Raises:
        ValueError: When nbins is not positive.
            Also when the number of data_points is less than nbins.
    """
    if nbins <= 0:
        raise ValueError('Number of bins should be positive. '
                         'It was {}.'.format(nbins))
    if len(data_points) < nbins:
        raise ValueError('Number of data points should be more than '
                         'number of bins. '
                         '#data points = {}, #bins = {}.'.format(
                             len(data_points), nbins))
    if not data_points:
        raise ValueError('Data points is empty.')
    bin_volume, bin_edges = np.histogram(data_points, bins=nbins)
    probability = bin_volume / np.sum(bin_volume)
    selected_bin_index = np.random.choice(range(nbins), 1, p=probability)
    drawn_sample = np.random.uniform(
        low=bin_edges[selected_bin_index],
        high=bin_edges[selected_bin_index + 1],
        size=1)[0]
    return drawn_sample