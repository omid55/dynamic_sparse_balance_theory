# Omid55
# Start date:     16 Oct 2018
# Modified date:  22 Nov 2018
# Author:   Omid Askarisichani
# Email:    omid55@cs.ucsb.edu
# Dynamic networks and specificly structural balance theory utility module.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import datetime
import math
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
# import enforce
from typing import Dict
from typing import List
from typing import Union
from typing import Set
from typing import Tuple

import utils


# @enforce.runtime_validation
def extract_graph(
        selected_edge_list: pd.DataFrame,
        sum_multiple_edge: bool = False) -> nx.DiGraph:
    """Extracts a list of graphs in each period of time.

    If there were multiple edges between two nodes, then if sum_multiple_edge
    is True, the edge weight is assigned as the summation of all edge weights.
    If sum_multiple_edge is False, then the latest edge weight is assigned as
    the weight.

    Args:
        selected_edge_list: Dataframe of edges containing required columns.

        sum_multiple_edge: Whether to pick the latest or sum multiple edges.

    Returns:
        Directed graph created from selected edge list.

    Raises:
        ValueError: if it does not contain required columns.
    """
    utils.check_required_columns(
        selected_edge_list, ['source', 'target', 'weight'])
    if not sum_multiple_edge:
        # Considers the weight of the latest edge weight as the weight.
        dataframe = selected_edge_list
    else:
        # Considers summation of edge weights as the weight.
        dataframe = selected_edge_list.groupby(
            ['source', 'target'], as_index=False)['weight'].sum()
    return nx.from_pandas_edgelist(
        dataframe,
        source='source',
        target='target',
        edge_attr='weight',
        create_using=nx.DiGraph())


# @enforce.runtime_validation
def extract_graphs(
        edge_list: pd.DataFrame,
        weeks: int = 4,
        accumulative: bool = False,
        sum_multiple_edge: bool = False) -> List[nx.DiGraph]:
    """Extracts a list of graphs in each period of time.

    It extracts graph structure for periods with the duration of given number
    of weeks. Separated networks are created from the edges only happened
    in that period; However, accumulative ones are created from all edges
    since the beginnig until that period.

    Args:
        edge_list: Dataframe of edges containing required columns.

        weeks: The number of weeks for the desired period length.

        accumulative: Whether we need separated networks or accumulative ones.

        sum_multiple_edge: Whether to pick the latest or sum multiple edges.

    Returns:
        List of directed graphs created based on accumulative or separated.

    Raises:
        ValueError: if it does not contain required columns.
    """
    utils.check_required_columns(
        edge_list, ['edge_date', 'source', 'target', 'weight'])
    start_date = min(edge_list['edge_date'])
    end_date = max(edge_list['edge_date'])
    dgraphs = []
    periods_num = int(np.floor((end_date - start_date).days / (weeks * 7)))
    # For every period, we extract a directed weighted signed network.
    for period_index in range(periods_num):
        period_start = (
            start_date + period_index * datetime.timedelta(weeks * 7))
        period_end = period_start + datetime.timedelta(weeks * 7)
        if not accumulative:
            # Chooses the edges only for this period.
            selected_edges = edge_list[
                (edge_list['edge_date'] >= period_start) &
                (edge_list['edge_date'] < period_end)]
        else:
            # Chooses the edges until this period (accumulative).
            selected_edges = edge_list[edge_list['edge_date'] < period_end]
        dgraph = extract_graph(
            selected_edges, sum_multiple_edge=sum_multiple_edge)
        dgraphs.append(dgraph)
    return dgraphs


# @enforce.runtime_validation
def get_just_periods(
        edge_list: pd.DataFrame,
        weeks: int = 4,
        accumulative: bool = False) -> List:
    """Extracts a list of graphs in each period of time.

    It extracts graph structure for periods with the duration of given number
    of weeks. Separated networks are created from the edges only happened
    in that period; However, accumulative ones are created from all edges
    since the beginnig until that period.

    Args:
        edge_list: Dataframe of edges containing required columns.

        weeks: The number of weeks for the desired period length.

        accumulative: Whether we need separated networks or accumulative ones.

        sum_multiple_edge: Whether to pick the latest or sum multiple edges.

    Returns:
        List of directed graphs created based on accumulative or separated.

    Raises:
        ValueError: if it does not contain required columns.
    """
    utils.check_required_columns(
        edge_list, ['edge_date'])
    periods = []
    start_date = min(edge_list['edge_date'])
    end_date = max(edge_list['edge_date'])
    periods_num = int(np.floor((end_date - start_date).days / (weeks * 7)))
    for period_index in range(periods_num):
        period_start = (
            start_date + period_index * datetime.timedelta(weeks * 7))
        period_end = period_start + datetime.timedelta(weeks * 7)
        if not accumulative:
            if isinstance(period_start, datetime.datetime):
                ps = period_start.date()
                pe = period_end.date()
            else:
                ps = period_start
                pe = period_end
            periods.append([str(ps), str(pe)])
        else:
            if isinstance(start_date, datetime.datetime):
                ps = start_date.date()
                pe = period_end.date()
            else:
                ps = start_date
                pe = period_end
            periods.append([str(ps), str(pe)])
    return periods


# # @enforce.runtime_validation
def get_metrics_for_network(
        dgraph: nx.DiGraph) -> Dict[str, Union[float, int]]:
    """Gets the different metrics of the given directed network.

    Args:
        dgraph: The input network.

    Returns:
        Dictionary of metrics mapped from name to integer value.

    Raises:
        None
    """
    metrics = {}

    # For directed graph.
    n = len(dgraph.nodes())
    e = len(dgraph.edges())
    metrics['#nodes'] = n
    metrics['#edges'] = e
    metrics['#edges/#nodes'] = e / n
    metrics['average in degree'] = np.mean(
        [deg for _, deg in list(dgraph.in_degree)])
    metrics['average out degree'] = np.mean(
        [deg for _, deg in list(dgraph.out_degree)])
    metrics['average w in degree'] = np.mean(
        [deg for _, deg in list(dgraph.in_degree(weight='weight'))])
    metrics['average w out degree'] = np.mean(
        [deg for _, deg in list(dgraph.out_degree(weight='weight'))])
    metrics['average degree'] = np.mean(
        [deg for _, deg in list(dgraph.degree)])
    metrics['average load'] = np.mean(list(
        nx.load_centrality(dgraph).values()))
    metrics['average eigenvector'] = np.mean(list(
        nx.eigenvector_centrality(dgraph, max_iter=10000).values()))
    metrics['average harmonic'] = np.mean(list(
        nx.harmonic_centrality(dgraph).values()))
    metrics['average closeness'] = np.mean(list(
        nx.closeness_centrality(dgraph).values()))
    metrics['average betweenness'] = np.mean(list(
        nx.betweenness_centrality(dgraph).values()))

    # Directed graphs' weights.
    weights = np.zeros(len(dgraph.edges()))
    for i, edge in enumerate(dgraph.edges()):
        weights[i] = dgraph.get_edge_data(edge[0], edge[1])['weight']
    metrics['weights min'] = min(weights)
    metrics['weights max'] = max(weights)
    metrics['weights average'] = np.mean(weights)
    metrics['weights std'] = np.std(weights)
    metrics['#pos edges'] = len(np.where(weights > 0)[0])
    metrics['#neg edges'] = len(np.where(weights < 0)[0])

    # For undirected version of the given directed graph.
    ugraph = nx.to_undirected(dgraph)
    metrics['average (und) clustering coefficient'] = np.mean(
        list(nx.clustering(ugraph, weight=None).values()))
    metrics['algebraic connectivity'] = nx.algebraic_connectivity(
        ugraph, weight='weight')

    # For Giant Connected Component.
    GCC = ugraph.subgraph(max(c for c in nx.connected_components(ugraph)))
    metrics['#gcc nodes'] = len(GCC.nodes())
    metrics['#gcc edges'] = len(GCC.edges())
    gcc_weights = np.zeros(len(GCC.edges()))
    for i, edge in enumerate(GCC.edges()):
        gcc_weights[i] = GCC.get_edge_data(edge[0], edge[1])['weight']
    metrics['#gcc pos edges'] = len(np.where(gcc_weights > 0)[0])
    metrics['#gcc neg edges'] = len(np.where(gcc_weights < 0)[0])
    metrics['gcc algebraic connectivity'] = nx.algebraic_connectivity(
        GCC, weight='weight')
    metrics['gcc diameter'] = nx.diameter(GCC)

    # My balance metrics.
    edge_balance = compute_edge_balance(
        dgraph, no_isomorph_cycles=True)
    balanced_cycles = 0
    cycles = 0
    for value in edge_balance.values():
        balanced_cycles += value['#balanced']
        cycles += value['#cycle3']
    balanced_ratio = None
    if cycles:
        balanced_ratio = balanced_cycles / cycles
#     metrics['balanced cycles 3 ratio'] = balanced_ratio
    if balanced_ratio is None:
        metrics['unbalanced cycles 3 ratio'] = None
    else:
        metrics['unbalanced cycles 3 ratio'] = 1 - balanced_ratio

    return metrics


# @enforce.runtime_validation
def cartwright_harary_balance(dgraph: nx.DiGraph) -> float:
    """Computes the cartwright and harary balance ratio.

    It computes all cycles in the network. Then for them, it counts
    the number of cycles that have even number of negative signs. By
    Cartwright and Hararry '1956, those are not balanced. This function
    returns the ration of those by all cycles in the given directed graph.

    Args:
        dgraph: Directed graph that is given for computing balance ratio.

    Returns:
        Number of cycles with even number of negative divided by all cycles.

    Raises:
        None
    """
    balanced_cycle_count = 0
    cycle_count = 0
    for cycle in nx.simple_cycles(dgraph):
        cycle_count += 1
        cycle += [cycle[0]]
        # For every cycle we count the number of negative edges.
        negative_count = 0
        for index in range(len(cycle) - 1):
            if dgraph.get_edge_data(
                    cycle[index], cycle[index + 1])['weight'] < 0:
                negative_count += 1
        if negative_count % 2 == 0:
            balanced_cycle_count += 1
    balance_ratio = balanced_cycle_count / cycle_count
    return balance_ratio


# @enforce.runtime_validation
def count_different_signed_edges(dgraph):
    different_signs = 0
    nodes = list(dgraph.nodes())
    for i in range(len(nodes)-1):
        for j in range(i+1, len(nodes)):
            if (dgraph.has_edge(nodes[i], nodes[j]) and
                    dgraph.has_edge(nodes[j], nodes[i])):
                wij = dgraph.get_edge_data(
                    nodes[i], nodes[j])['weight']
                wji = dgraph.get_edge_data(
                    nodes[j], nodes[i])['weight']
                if np.sign(wij) != np.sign(wji):
                    different_signs += 1
    return different_signs


# @enforce.runtime_validation
def compute_edge_balance(
        dgraph: nx.DiGraph,
        no_isomorph_cycles: bool = False) -> Dict[tuple, Dict[str, int]]:
    """Computes edge balance based on Van De Rijt idea.

    With no_isomorph_cycles=True, we can get the number of cycles and edge
        is not meaningful. However, with no_isomorph_cycles=False, for every
        edge which is involved one of cycle3, there will be information in
        the dictionary.

    Args:
        dgraph: Directed weighted graph to apply edge balance.

        no_isomorph_cycles: If true, it does not count
            the isomorph cycle3.

    Returns:
        Dictionary of edges mapped to the number of balanced
            and total number of triads the edge is involved in.

    Raises:
        None
    """
    edge_balance = {}
    cycle3s = set()

    for edge in dgraph.edges():
        triad_count = 0
        balanced_count = 0
        weight_sum = 0
        weight_distance = 0

        nodes = list(set(dgraph.nodes()) - set(edge))
        xij = dgraph.get_edge_data(edge[0], edge[1])['weight']
        for node in nodes:
            if dgraph.has_edge(
                    edge[1], node) and dgraph.has_edge(node, edge[0]):
                triad_str = ','.join((str(edge[1]), str(node), str(edge[0])))
                if not no_isomorph_cycles or (
                        no_isomorph_cycles and (triad_str not in cycle3s)):
                    if triad_str not in cycle3s:
                        triad_isomorph1_str = ','.join(
                            (str(edge[0]), str(edge[1]), str(node)))
                        triad_isomorph2_str = ','.join(
                            (str(node), str(edge[0]), str(edge[1])))
                        cycle3s = cycle3s.union(
                            set([triad_str,
                                 triad_isomorph1_str,
                                 triad_isomorph2_str]))

                    triad_count += 1
                    xik = dgraph.get_edge_data(edge[1], node)['weight']
                    xkj = dgraph.get_edge_data(node, edge[0])['weight']

                    weight_sum += np.sign(xik) * np.sign(xkj)

                    weight_distance += abs(xij - (xik * xkj))
                    if np.sign(xij * xik * xkj) > 0:
                        balanced_count += 1

        if triad_count:
            as_expected_sign = int(np.sign(weight_sum) == np.sign(xij))
            edge_balance[edge] = {
                '#balanced': balanced_count,
                '#cycle3': triad_count,
                'weight_distance': weight_distance,
                'as_expected_sign': as_expected_sign}
    return edge_balance


# @enforce.runtime_validation
def plot_evolving_graphs(
        dgraphs: List[nx.DiGraph],
        titles: List[str] = None,
        aggregated_dgraph: nx.DiGraph = None) -> None:
    """Plots list of given graphs.

    If aggregated_dgraph is None, it computes it by combining
        all graphs together.

    Args:
        dgraphs: List of given evolving graphs.

        titles: List of title content if needed.

        aggregated_dgraph: One aggregated graph.

    Returns:
        None.

    Raises:
        None.
    """
    n = 3
    m = np.ceil(len(dgraphs) / n)
    sns.set(rc={'figure.figsize': (6*n, 6*m)})
    if not aggregated_dgraph:
        aggregated_dgraph = nx.compose_all(dgraphs)
    all_positions = nx.layout.spring_layout(aggregated_dgraph)
    for index, dgraph in enumerate(dgraphs):
        plt.subplot(m, n, index + 1)
        nx.draw(dgraph, pos=all_positions, with_labels=True)
        if titles:
            title_name = titles[index]
        else:
            title_name = 'Period {}'.format(index + 1)
        plt.title('{}\n{} nodes, {} edges'.format(
            title_name, len(dgraph.nodes()), len(dgraph.edges())))


# @enforce.runtime_validation
def compute_fairness_goodness(
        dgraph: nx.DiGraph,
        weight_range: float = 20.0,
        max_iteration: int = 100,
        verbose: bool = True) -> Dict[str, Dict[int, float]]:
    """Computes fairness and goodness per node in a weighted signed graph.

    Args:
        dgraph: Weighted signed graph with weights fall in [-l, l].

        weight_range: Range of weights, for above graph is 2*l.

        max_iteration: The maximum number of iterations if not converge.

        verbose: If we want to print information while computing.

    Returns:
        Dictionary of fairness and goodness as dictionary of values for nodes.

    Raises:
        None.
    """
    # Initializes fairness of all nodes to 1 and goodness of all to 0.
    fairness = {}
    goodness = {}
    nodes = dgraph.nodes()
    for node in nodes:
        fairness[node] = 1
        in_degree = dgraph.in_degree(node)
        if in_degree:
            goodness[node] = dgraph.in_degree(
                node, weight='weight') / in_degree
        else:
            goodness[node] = 0

    nodes = dgraph.nodes()
    for iteration in range(max_iteration):
        fairness_diff = 0
        goodness_diff = 0

        if verbose:
            print('-----------------')
            print("Iteration number", iteration)
            print('Updating goodness')
        for node in nodes:
            inedges = dgraph.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]]*edge[2]
            in_degree = len(inedges)
            if in_degree:
                goodness_diff += abs(g/in_degree - goodness[node])
                goodness[node] = g/in_degree

        if verbose:
            print('Updating fairness')
        for node in nodes:
            outedges = dgraph.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]])/weight_range
            out_degree = len(outedges)
            if out_degree:
                fairness_diff += abs(f/out_degree - fairness[node])
                fairness[node] = f/out_degree

        if verbose:
            print('Differences in fairness and goodness = {}, {}.'.format(
                fairness_diff, goodness_diff))
        if (fairness_diff < math.pow(10, -6)
                and goodness_diff < math.pow(10, -6)):
            break

    return {'fairness': fairness, 'goodness': goodness}


# @enforce.runtime_validation
def is_transitive_balanced(triad: np.ndarray) -> bool:
    """Checks whether input triad matrix is transitively balanced or not.

    Transitive balance is defined on only one rule:
    Friend of friend is friend.

    Args:
        triad: Input triad matrix.

    Returns:
        Boolean result whether triad is transitively balanced or not.

    Raises:
        ValueError: If there is a self loop in the given triad.
    """
    for i in range(3):
        if triad[i, i]:
            raise ValueError('There is a self loop in given triad: {}.'.format(
                triad))

    for i in range(3):
        a = i
        b = (i + 1) % 3
        c = (i + 2) % 3

        # # For every edge between a and b.
        # if triad[a, b] * triad[b, a] < 0:
        #     return False
        # # For every path from a to c.
        # if ((abs(triad[a, b]) and abs(triad[b, c]))
        #         and (triad[a, b]*triad[b, c]*triad[a, c] <= 0)):
        #     return False
        # # For every path from c to a.
        # if ((abs(triad[c, b]) and abs(triad[b, a]))
        #         and (triad[c, b]*triad[b, a]*triad[c, a] <= 0)):
        #     return False

        # For every path from a to c.
        if ((triad[a, b] > 0) and (triad[b, c] > 0)
                and (triad[a, b]*triad[b, c]*triad[a, c] <= 0)):
            return False

        # For every path from c to a.
        if ((triad[c, b] > 0) and (triad[b, a] > 0)
                and (triad[c, b]*triad[b, a]*triad[c, a] <= 0)):
            return False

    return True


# @enforce.runtime_validation
def _get_all_triad_permutations(triad_matrix: np.ndarray) -> Set[str]:
    """Gets all of permutations of nodes in a matrix in string format.

    It computes different matrices with swapping same columns and rows.

    Args:
        triad_matrix: The triad adjacency matrix.

    Returns:
        Set of string permutations of the triad adjacency matrices.

    Raises:
        None.
    """
    permutations = [triad_matrix]
    mat01 = utils.swap_nodes_in_matrix(triad_matrix, 0, 1)
    mat02 = utils.swap_nodes_in_matrix(triad_matrix, 0, 2)
    mat12 = utils.swap_nodes_in_matrix(triad_matrix, 1, 2)
    permutations.extend([mat01, mat02, mat12])
    permutations.extend(
        [utils.swap_nodes_in_matrix(mat01, 0, 2),
         utils.swap_nodes_in_matrix(mat01, 1, 2),
         utils.swap_nodes_in_matrix(mat02, 0, 1),
         utils.swap_nodes_in_matrix(mat02, 1, 2),
         utils.swap_nodes_in_matrix(mat12, 0, 1),
         utils.swap_nodes_in_matrix(mat12, 0, 2)])
    result = set()
    for permutation in permutations:
        result.add(str(permutation))
    return result


# @enforce.runtime_validation
def generate_all_possible_sparse_triads(
        ) -> Tuple[Dict[str, int], List[np.ndarray]]:
    """Generates all possible sparse triads.

    Args:
        None.

    Returns:
        Dictionary of intiallized all sparse triad types.

    Raises:
        None.
    """
    possible_edges = [0, 1, -1]
    adj_matrices = []
    for i1 in possible_edges:
        for i2 in possible_edges:
            for i3 in possible_edges:
                for i4 in possible_edges:
                    for i5 in possible_edges:
                        for i6 in possible_edges:
                            adj_matrices.append(np.array(
                                [[0, i1, i2], [i3, 0, i4], [i5, i6, 0]]))
    triad_map = {}
    triad_list = []
    for adj_matrix in adj_matrices:
        if str(adj_matrix) not in triad_map:
            triad_list.append(adj_matrix)
            triad_index = len(triad_list) - 1
            for permutation in _get_all_triad_permutations(adj_matrix):
                triad_map[str(permutation)] = triad_index
    # triads = {'triad_map': triad_map, 'triad_list': triad_list}
    return triad_map, triad_list


# @enforce.runtime_validation
def _detect_triad_type_for_all_subgraph3(
        dgraph: nx.DiGraph,
        triad_map: Dict[str, int] = None,
        verbose: bool = False) -> Dict[str, int]:
    """Detects triad type for all possible subgraphs of size 3 in given graph.

    Args:
        dgraph: The directed graph.

        triad_map: Initialized sparse triad map (string to triad type index).

        verbose: Whether we want it to print '.' as the finished indicator.

    Returns:
        Dictionary of string name of subgraph to its triad type index.

    Raises:
        None.
    """
    if not triad_map:
        triad_map, _ = generate_all_possible_sparse_triads()
    subgraph2triad_type = {}
    nodes_list = np.array(dgraph.nodes())
    adj_matrix = utils.dgraph2adjacency(dgraph)
    adj_matrix[adj_matrix > 0] = 1
    adj_matrix[adj_matrix < 0] = -1
    triads = list(itertools.combinations(range(len(nodes_list)), 3))
    for triad in triads:
        triad_subgraph_matrix = utils.sub_adjacency_matrix(
            adj_matrix, list(triad))
        triad_subgraph_key = str(np.array(triad_subgraph_matrix, dtype=int))
        if triad_subgraph_key not in triad_map:
            print(triad, 'is not found.')
            print('Their names are:', nodes_list[np.array(triad)])
            print('Simplified subgraph was:', triad_subgraph_matrix)
        else:
            triad_type_index = triad_map[triad_subgraph_key]
            # It is imported to sort the key name unless final dictionary
            #   might have non-unique keys.
            subgraph2triad_type[str(tuple(
                sorted(nodes_list[np.array(triad)])))] = triad_type_index
    if verbose:
        print('.', end='')
    return subgraph2triad_type


# @enforce.runtime_validation
def compute_transition_matrix(
        dgraphs: List[nx.DiGraph],
        unique_triad_num: int,
        triad_map: Dict[str, int] = None,
        verbose: bool = False) -> Dict[
                                    str, Union[List[np.ndarray], List[Dict]]]:
    """Computes transition matrix and triads count for every consequetive graph.

    Args:
        dgraphs: List of graphs in timely order.

        unique_triad_num: Number of unique sparse triads.

        triad_map: Initialized sparse triad map (string to triad type index).

        verbose: If we want to print a . as progress while computing.

    Returns:
        Dictionary of list of transition matrices and list of all subgraphs3
            with their corresponding triad index.

    Raises:
        ValueError: If the size of dgraphs is not at least 2.
    """
    if len(dgraphs) < 2:
        raise ValueError(
            'We need at least 2 directed graphs for computing transition.')

    if not triad_map:
        triad_map, triad_list = generate_all_possible_sparse_triads()
        unique_triad_num = len(triad_list)

    # Detects the sparse triad types of all networks.
    triads_types = [_detect_triad_type_for_all_subgraph3(
            dgraph=dgraph, triad_map=triad_map, verbose=verbose)
            for dgraph in dgraphs]

    transition_matrices = []
    for index in range(len(dgraphs)-1):
        triads_type1 = triads_types[index]        # First graph.
        triads_type2 = triads_types[index + 1]    # Subsequent graph.

        intersected_keys = list(set.intersection(
            set(triads_type1.keys()), set(triads_type2.keys())))

        transition_matrix = np.zeros((unique_triad_num, unique_triad_num))
        for key in intersected_keys:
            transition_matrix[triads_type1[key], triads_type2[key]] += 1

        transition_matrix = utils.make_matrix_row_stochastic(transition_matrix)

        transition_matrices.append(transition_matrix)
    return {'transition_matrices': transition_matrices,
            'triads_types': triads_types}


# @enforce.runtime_validation
def _get_eigen_decomposition_of_markov_transition(
        transition_matrix: np.ndarray,
        aperiodic_irreducible_eps: float = 0.0001) -> Tuple:
    """Gets the eigen value and vectors from transition matrix.

    A Markov chain is irreducible if we can go from any state to any state.
    This entails all transition probabilities > 0.
    A Markov chain is aperiodic if all states are accessible from all other
    states. This entails all transition probabilities > 0.

    Args:
        transition_matrix: Square Markov transition matrix.

        aperiodic_irreducible_eps: To make the matrix aperiodic/irreducible.

    Returns:
        Dictionary of eigen val/vec of irreducible and aperiodic markov chain.

    Raises:
        ValueError: If the matrix was not squared.
    """
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError('Transition matrix is not squared.')
    matrix = transition_matrix.copy()
    matrix = np.nan_to_num(matrix)
    matrix += aperiodic_irreducible_eps
    aperiodic_irreducible_transition_matrix = (
        matrix.T / np.sum(matrix, axis=1)).T
    eigen_values, eigen_vectors = np.linalg.eig(
        aperiodic_irreducible_transition_matrix.T)
    return eigen_values, eigen_vectors


# @enforce.runtime_validation
def get_stationary_distribution(
        transition_matrix: np.ndarray,
        aperiodic_irreducible_eps: float = 0.0001) -> np.ndarray:
    """Gets the stationary distribution of given transition matrix.

    A Markov chain is irreducible if we can go from any state to any state.
    This entails all transition probabilities > 0.
    A Markov chain is aperiodic if all states are accessible from all other
    states. This entails all transition probabilities > 0.

    Args:
        transition_matrix: Square Markov transition matrix.

        aperiodic_irreducible_eps: To make the matrix aperiodic/irreducible.

    Returns:
        Array of size one dimension of matrix.

    Raises:
        ValueError: If the matrix was not squared.
    """
    eigen_values, eigen_vectors = (
        _get_eigen_decomposition_of_markov_transition(
            transition_matrix=transition_matrix,
            aperiodic_irreducible_eps=aperiodic_irreducible_eps))
    index = np.where(eigen_values > 0.99)[0][0]
    stationary_distribution = [item.real for item in eigen_vectors[:, index]]
    stationary_distribution /= sum(stationary_distribution)
    return stationary_distribution


# @enforce.runtime_validation
def get_mixing_time_range(
        transition_matrix: np.ndarray,
        aperiodic_irreducible_eps: float = 0.0001,
        distance_from_stationary_eps: float = 0.01,
        verbose: bool = False) -> np.float64:
    """Gets the mixing time with respect to given distance eps.

    For more information one can look at:
    https://www.math.dartmouth.edu/~pw/M100W11/nathan.pdf
    and
    https://math.dartmouth.edu/~pw/math100w13/kale.pdf
    Argument distance_from_stationary_eps is eps in the document of the
    first link.

    Args:
        transition_matrix: Square Markov transition matrix.

        aperiodic_irreducible_eps: To make the matrix aperiodic/irreducible.

        distance_from_stationary_eps: Distance from stationary distribution.

        verbose: Whether we need it to print about lambda2 and pie_star.

    Returns:
        Number of steps in float type.

    Raises:
        ValueError: If the matrix was not squared.
    """
    eigen_values, eigen_vectors = (
        _get_eigen_decomposition_of_markov_transition(
            transition_matrix=transition_matrix,
            aperiodic_irreducible_eps=aperiodic_irreducible_eps))
    index = np.where(eigen_values > 0.99)[0][0]
    stationary_distribution = [item.real for item in eigen_vectors[:, index]]
    stationary_distribution /= sum(stationary_distribution)
    lambda2 = sorted(eigen_values, reverse=True)[1]
    pie_star = np.min(stationary_distribution)
    if verbose:
        print('\nlambda2: {}\npie_star: {}'.format(
            np.real(lambda2), pie_star))
    tau = (1/(1-lambda2)) * np.log(1/(pie_star*distance_from_stationary_eps))
    return np.real(tau)


# @enforce.runtime_validation
def _randomize_network(
        dgraph: nx.DiGraph,
        switching_count_coef: int = 300) -> nx.DiGraph:
    """Generates randomized network for a given directed graph.

    It preserves the in- and out- degree intact. It keeps the degree
    distribution by randomly switches single and double edges for at least
    MAX_TIMES time nothing changed.

    Args:
        dgraph: The input directed graph.

        switching_count_coef: The coef for number of edge switchings.

    Returns:
        Adjacency matrix of randomized network with same in- and out-degree.

    Raises:
        Exception if the algorithm does not converge.
    """
    # If after MAX_TIMES times, it couldn't switch, we call it consensus
    #   (convergence) and terminate the algorithm.
    MAX_TIMES = 1000

    edge_count = len(dgraph.edges())
    adj = utils.dgraph2adjacency(dgraph=dgraph)
    desired_switching_count = switching_count_coef * edge_count
    switching_count = 0
    prev_switching_count = 0
    counter = 0

    while switching_count < desired_switching_count:
        # Randomly choose 2 edges.
        counter += 1

        binarized_adj = abs(adj.copy())
        binarized_adj[binarized_adj > 0] = 1
        both_double_edges = np.floor((binarized_adj + binarized_adj.T)/2)
        double_edges = np.where(both_double_edges > 0)

        # Double edges.
        i, j = np.random.choice(
            range(len(double_edges[0])), size=2, replace=False)
        s1 = double_edges[0][i]
        t1 = double_edges[1][i]
        s2 = double_edges[0][j]
        t2 = double_edges[1][j]
        if not (adj[s1, t2] or adj[s2, t1] or adj[t2, s1] or adj[t1, s2]
                or s1 == t2 or s1 == s2 or s2 == t1 or t1 == t2):
            utils.swap_two_elements_in_matrix(adj, s1, t1, s1, t2)
            utils.swap_two_elements_in_matrix(adj, t1, s1, t2, s1)
            utils.swap_two_elements_in_matrix(adj, s2, t2, s2, t1)
            utils.swap_two_elements_in_matrix(adj, t2, s2, t1, s2)
            switching_count += 1

        # Single edges.
        # Need to compute it again because adj might have been changed.
        binarized_adj = abs(adj.copy())
        binarized_adj[binarized_adj > 0] = 1
        both_double_edges = np.floor((binarized_adj + binarized_adj.T)/2)
        single_edges = np.where(binarized_adj - both_double_edges > 0)

        i, j = np.random.choice(
            range(len(single_edges[0])), size=2, replace=False)
        s1 = single_edges[0][i]
        t1 = single_edges[1][i]
        s2 = single_edges[0][j]
        t2 = single_edges[1][j]
        if not(adj[s1, t2] or adj[s2, t1]
                or s1 == t2 or s1 == s2 or s2 == t1 or t1 == t2):
            utils.swap_two_elements_in_matrix(adj, s1, t1, s1, t2)
            utils.swap_two_elements_in_matrix(adj, s2, t2, s2, t1)
            switching_count += 1

        if not counter % MAX_TIMES:
            if prev_switching_count == switching_count:
                raise Exception('Not converged.')
            else:
                prev_switching_count = switching_count
    return utils.adjacency2digraph(adj_matrix=adj, similar_this_dgraph=dgraph)


# @enforce.runtime_validation
def compute_randomized_transition_matrix(
        dgraph1: nx.DiGraph,
        dgraph2: nx.DiGraph,
        unique_triad_num: int,
        triad_map: Dict[str, int] = None,
        switching_count_coef: int = 300,
        randomized_num: int = 100) -> List[np.ndarray]:
    """Computes the transition of many randomized versions of two networks.

    Args:
        dgraph1: First directed graph.

        dgraph2: Second directed graph.

        unique_triad_num: Number of unique sparse triads.

        triad_map: Initialized sparse triad map (string to triad type index).

        switching_count_coef: The coef for number of edge switchings.

        randomized_num: Number of transition matrices to generate.

    Returns:
        List of transition matrices from subsequent randomized networks.

    Raises:
        None.
    """
    if not triad_map:
        triad_map, triad_list = generate_all_possible_sparse_triads()
        unique_triad_num = len(triad_list)
    rand_transition_matrices = []
    for _ in range(randomized_num):
        rand_dgraph1 = _randomize_network(
            dgraph=dgraph1, switching_count_coef=switching_count_coef)
        rand_dgraph2 = _randomize_network(
            dgraph=dgraph2, switching_count_coef=switching_count_coef)
        rand_transition_matrices.append(
            compute_transition_matrix(
                dgraphs=[rand_dgraph1, rand_dgraph2],
                unique_triad_num=unique_triad_num,
                triad_map=triad_map)['transition_matrices'][0])
    return rand_transition_matrices


# @enforce.runtime_validation
def get_robustness_of_transitions(
        transition_matrices: List[np.ndarray],
        lnorm: int = 2) -> pd.DataFrame:
    """Gets stationary dist of each transition and dist/corr with average one.

    Args:
        transition_matrices: List of squared transition matrices.

        lnorm: The norm integer (l1 or l2 usually).

    We compute the average transition matrix from all given transition matrices
    and then compute the stationary distribution for that matrix. Then the
    objective is to find out how different/robust all transitions are with
    respect to the stationary distribution from average transtision matrix. We
    compute lnorm distance and also Pearson correlation of them with average
    distribution. Also it returns the list of stationary distributions.

    Returns:
        Dataframe of distance and Pearsons correlation from average transition.

    Raises:
        None.
    """
    n, _ = transition_matrices[0].shape
    avg_transition_matrix = np.zeros((n, n))
    for i in range(len(transition_matrices)):
        avg_transition_matrix += transition_matrices[i]
    avg_transition_matrix /= n

    result = []
    avg_stationary_distribution = get_stationary_distribution(
        avg_transition_matrix)

    for index, transition_matrix in enumerate(transition_matrices):
        matrix_dist_distance = np.linalg.norm(
            avg_transition_matrix - transition_matrix, lnorm)
        matrix_dist_rval, matrix_dist_pval = sp.stats.pearsonr(
            avg_transition_matrix.flatten(), transition_matrix.flatten())

        stationary_dist = get_stationary_distribution(transition_matrix)

        stationary_dist_distance = np.linalg.norm(
            avg_stationary_distribution - stationary_dist, lnorm)
        stationary_dist_rval, stationary_dist_pval = sp.stats.pearsonr(
            avg_stationary_distribution, stationary_dist)

        result.append(
            ['Period {} to Period {}'.format(index+1, index+2),
                matrix_dist_distance,
                matrix_dist_rval,
                matrix_dist_pval,
                stationary_dist_distance,
                stationary_dist_rval,
                stationary_dist_pval])

    result = pd.DataFrame(
        result, columns=[
            'Transitions',
            'Matrix L{}-Norm Dist. from Average'.format(lnorm),
            'Matrix Pearson r-value',
            'Matrix Pearson p-value',
            'Stationary Dist. L{}-Norm Dist. from Average'.format(lnorm),
            'Stationary Dist. Pearson r-value',
            'Stationary Dist. Pearson p-value'])

    return result


# @enforce.runtime_validation
def generate_converted_graphs(
        dgraph: nx.DiGraph,
        convert_from: float = 0.0,
        convert_to: float = 1.0,
        percentage: float = 5.0,
        how_many_to_generate: int = 10) -> List[nx.DiGraph]:
    """Generates a list digraphs with randomly converting a percentage of sign.

    It generates a list of graphs with partially converting a given percentage
    of edge sign from given content to another given content from one given
    directed graph. This has been done to perform a robustness check on whether
    one of edge weights were infered/given wrong (it should have been value
    convert_to; however, it has been set to value convert_from). In this way,
    we randomly convert a small (i.e. 5%) from another edge weight and return
    the graphs to see whether the subsequent analysis is robust with respect to
    this possible noise in the data.

    Args:
        dgraph: Given directed graph.

        percentage: The percentage of edges to do randomly sign conversion.

        convert_from: Converts from this value.

        convert_to: Converts to this value.

        how_many_to_generate: How many new directed graphs to be generated.

    Returns:
        A list of generated directed graphs.

    Raises:
        ValueError: If percentage was wrong or dgraph does not contain
            convert_from value.
    """
    # The process is easier to be applied on adjacency matrix.
    if percentage < 0 or percentage > 100:
        raise ValueError(
            'Inputted percentage was wrong: {}.'.format(percentage))
    original_adj_matrix = utils.dgraph2adjacency(dgraph=dgraph)
    unq_val = max(convert_from, convert_to) + 5  # Value that is not targeted.
    np.fill_diagonal(original_adj_matrix, unq_val)
    from_edges = np.where(original_adj_matrix == convert_from)
    from_edges_cnt = len(from_edges[0])
    if not from_edges_cnt:
        raise ValueError(
            'Inputted directed graph does not contain the edge weight'
            ' equals {}.'.format(convert_from))
    generated_dgraphs = []
    for _ in range(how_many_to_generate):
        adj_matrix = original_adj_matrix.copy()
        selected_indices = np.random.choice(
            from_edges_cnt,
            int(percentage * from_edges_cnt / 100),
            replace=False)
        for index in selected_indices:
            adj_matrix[from_edges[0][index], from_edges[1][index]] = convert_to
        # This is a social network and should not contain any self-loop.
        np.fill_diagonal(adj_matrix, 0)
        generated_dgraphs.append(
            utils.adjacency2digraph(
                adj_matrix=adj_matrix,
                similar_this_dgraph=dgraph))
    return generated_dgraphs
