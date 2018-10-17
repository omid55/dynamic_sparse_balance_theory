# Omid55
# Date:     16 Oct 2018
# Author:   Omid Askarisichani
# Email:    omid55@cs.ucsb.edu
# Dynamic balance theory utils module.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Dict
from typing import List
from typing import Tuple
import numpy as np
import pandas as pd
import datetime
import networkx as nx

import utils


def extract_graphs(
        edge_list: pd.DataFrame,
        weeks: int = 4,
        accumulated: bool = False) -> List[nx.DiGraph]:
    """Extracts a list of graphs in each period of time.

    It extracts graph structure for periods with the duration of given number
    of weeks. Separated networks are created from the edges only happened
    in that period. However, accumulated ones are created from all edges
    since the beginnig until that period.

    Args:
        edge_list: Dataframe of edges containing required columns.

        weeks: The number of weeks for the desired period length.

        accumulated: Whether we need separated networks or accumulated ones.

    Returns:
        List of directed graphs created based on accumulated or separated.

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
        if not accumulated:
            # Chooses the edges only for this period.
            selected_edges = edge_list[
                (edge_list['edge_date'] >= period_start) &
                (edge_list['edge_date'] < period_end)]
        else:
            # Chooses the edges until this period (accumulated).
            selected_edges = edge_list[edge_list['edge_date'] < period_end]
        dgraph = nx.from_pandas_edgelist(
            selected_edges,
            source='source',
            target='target',
            edge_attr='weight',
            create_using=nx.DiGraph())
        dgraphs.append(dgraph)
    return dgraphs


def get_metrics_for_network(directed_graph: nx.graph) -> Dict[str, int]:
    """Gets the different metrics of the given directed network.

    Args:
        directed_graph: The input network.

    Returns:
        Dictionary of metrics mapped from name to integer value.

    Raises:
        None
    """
    metrics = {}

    # For directed graph.
    n = len(directed_graph.nodes())
    e = len(directed_graph.edges())
    metrics['#nodes'] = n
    metrics['#edges'] = e
    metrics['#edges/#nodes'] = e / n
    metrics['average in degree'] = np.mean(
        [deg for _, deg in list(directed_graph.in_degree)])
    metrics['average out degree'] = np.mean(
        [deg for _, deg in list(directed_graph.out_degree)])
    metrics['average w in degree'] = np.mean(
        [deg for _, deg in list(directed_graph.in_degree(weight='weight'))])
    metrics['average w out degree'] = np.mean(
        [deg for _, deg in list(directed_graph.out_degree(weight='weight'))])
    metrics['average degree'] = np.mean(
        [deg for _, deg in list(directed_graph.degree)])
    metrics['average load'] = np.mean(list(
        nx.load_centrality(directed_graph).values()))
    metrics['average eigenvector'] = np.mean(list(
        nx.eigenvector_centrality(directed_graph, max_iter=10000).values()))
    metrics['average harmonic'] = np.mean(list(
        nx.harmonic_centrality(directed_graph).values()))
    metrics['average closeness'] = np.mean(list(
        nx.closeness_centrality(directed_graph).values()))
    metrics['average betweenness'] = np.mean(list(
        nx.betweenness_centrality(directed_graph).values()))

    # Directed graphs' weights.
    weights = np.zeros(len(directed_graph.edges()))
    for i, edge in enumerate(directed_graph.edges()):
        weights[i] = directed_graph.get_edge_data(edge[0], edge[1])['weight']
    metrics['weights min'] = min(weights)
    metrics['weights max'] = max(weights)
    metrics['weights average'] = np.mean(weights)
    metrics['weights std'] = np.std(weights)
    metrics['#pos edges'] = len(np.where(weights > 0)[0])
    metrics['#neg edges'] = len(np.where(weights < 0)[0])

    # For undirected version of the given directed graph.
    undirected_graph = nx.to_undirected(directed_graph)
    metrics['average (und) clustering coefficient'] = np.mean(
        list(nx.clustering(undirected_graph, weight=None).values()))
    metrics['algebraic connectivity'] = nx.algebraic_connectivity(
        undirected_graph, weight='weight')

    # For Giant Connected Component.
    GCC = max(nx.connected_component_subgraphs(undirected_graph), key=len)
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
        directed_graph, no_isomorph_cycles=True)
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


def compute_edge_balance(
        dgraph: nx.DiGraph,
        no_isomorph_cycles=False) -> Dict[Tuple, Dict[str, int]]:
    """Computes edge balance based on Van De Rijt idea.

    Args:

        dgraph: Directed weighted graph to apply edge balance.

        no_isomorph_cycles: If true, it does not count
            the isomorph triads.

    Returns:
        Dictionary of edges mapped to the number of balanced
            and total number of triads the edge is involved in.

    Raises:
        None
    """
    edge_balance = {}
    triads = set()

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
                if no_isomorph_cycles:
                    triad_str = ','.join(
                        (str(edge[1]), str(node), str(edge[0])))
                    if triad_str not in triads:
                        triad_isomorph1_str = ','.join(
                            (str(edge[0]), str(edge[1]), str(node)))
                        triad_isomorph2_str = ','.join(
                            (str(node), str(edge[0]), str(edge[1])))
                        triads = triads.union(
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
                else:
                    triad_count += 1
                    xik = dgraph.get_edge_data(edge[1], node)['weight']
                    xkj = dgraph.get_edge_data(node, edge[0])['weight']
                    xij = dgraph.get_edge_data(edge[0], edge[1])['weight']

                    weight_sum += np.sign(xik) * np.sign(xkj)

                    weight_distance += abs(xij - (xik * xkj))
                    if np.sign(xij * xik * xkj) > 0:
                        balanced_count += 1

        if triad_count:
            as_expected_sign = (np.sign(weight_sum) == np.sign(xij))
            edge_balance[edge] = {
                '#balanced': balanced_count,
                '#cycle3': triad_count,
                'weight_distance': weight_distance,
                'as_expected_sign': as_expected_sign}
    return edge_balance
