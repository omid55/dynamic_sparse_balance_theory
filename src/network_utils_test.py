# Omid55
# Test module for network_utils.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import unittest
from parameterized import parameterized
import networkx as nx
import pandas as pd
import datetime

import utils
import network_utils


class MyTestClass(unittest.TestCase):
    # =========================================================================
    # ==================== extract_graphs =====================================
    # =========================================================================
    def test_extract_graphs_raises_with_missing_columns(self):
        sample_edge_list = pd.DataFrame({'source': [1, 2], 'target': [5, 6]})
        with self.assertRaises(ValueError):
            network_utils.extract_graphs(edge_list=sample_edge_list)

    @parameterized.expand(
        [["seperated graphs", False], ["accumulated graphs", True]])
    def test_extract_graphs(self, name, accumulated):
        # source, target, weight, edge_date
        matrix_edges = [[1, 2, +1, datetime.datetime(2017, 1, 1)],
                        [2, 3, +1, datetime.datetime(2017, 1, 4)],
                        [3, 1, +1, datetime.datetime(2017, 2, 5)],
                        [1, 4, -1, datetime.datetime(2017, 2, 13)],
                        [4, 3, -1, datetime.datetime(2017, 2, 24)],
                        [-1, -1, -1, datetime.datetime(2017, 2, 28)]]
                        # The last one is going to be ignored because fall into
                        #   another period which is neglected.
        sample_edge_list = pd.DataFrame(
            matrix_edges, columns=['source', 'target', 'weight', 'edge_date'])
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 3, 4])
        g2.add_edge(3, 1)
        g2.add_edge(1, 4)
        g2.add_edge(4, 3)
        g3 = nx.DiGraph()
        g3.add_nodes_from([1, 2, 3, 4])
        g3.add_edge(1, 2)
        g3.add_edge(2, 3)
        g3.add_edge(3, 1)
        g3.add_edge(1, 4)
        g3.add_edge(4, 3)
        if not accumulated:
            expected = [g1, g2]
        else:
            expected = [g1, g3]
        computed = network_utils.extract_graphs(
            edge_list=sample_edge_list, weeks=4, accumulated=accumulated)
        for expected_graph, computed_graph in zip(expected, computed):
            self.assertTrue(utils.graph_equals(expected_graph, computed_graph))


    # =========================================================================
    # ==================== get_metrics_for_network ============================
    # =========================================================================
    def test_get_metrics_for_network(self):
        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3, 4])
        DG.add_edge(1, 2, weight=1)
        DG.add_edge(2, 3, weight=1)
        DG.add_edge(3, 1, weight=-1)
        DG.add_edge(1, 3, weight=-1)
        computed = network_utils.get_metrics_for_network(DG)
        expected = {
            '#edges': 4,
            '#edges/#nodes': 1,
            '#gcc edges': 3,
            '#gcc neg edges': 1,
            '#gcc nodes': 3,
            '#gcc pos edges': 2,
            '#neg edges': 2,
            '#nodes': 4,
            '#pos edges': 2,
            'algebraic connectivity': 0,
            'average (und) clustering coefficient': 0.75,
            'average betweenness': 0.0833,
            'average closeness': 0.3888,
            'average degree': 2,
            'average eigenvector': 0.4222,
            'average harmonic': 1.25,
            'average in degree': 1,
            'average w in degree': 0,
            'average w out degree': 0,
            'average load': 0.0833,
            'average out degree': 1,
            'gcc algebraic connectivity': 2.9999,
            'gcc diameter': 1,
            'unbalanced cycles 3 ratio': 1,
            'weights max': 1,
            'weights average': 0,
            'weights min': -1,
            'weights std': 1
        }
#         utils.print_dict_pretty(computed)
#         self.assertDictEqual(computed, expected)
        for key, value in expected.items():
            self.assertAlmostEqual(value, computed[key], places=3)

    # =========================================================================
    # ====================== cartwright_harary_balance ========================
    # =========================================================================
    def test_cartwright_harary_balance(self):
        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3])
        DG.add_edge(1, 2, weight=1)
        DG.add_edge(2, 3, weight=1)
        DG.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance(DG), 0)

        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3])
        DG.add_edge(1, 2, weight=-1)
        DG.add_edge(2, 3, weight=-1)
        DG.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance(DG), 0)

        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3])
        DG.add_edge(1, 2, weight=1)
        DG.add_edge(2, 3, weight=-1)
        DG.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance(DG), 1)

        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3, 4, 5])
        DG.add_edge(1, 2, weight=1)
        DG.add_edge(2, 3, weight=-1)
        DG.add_edge(3, 1, weight=-1)
        DG.add_edge(3, 4, weight=-1)
        DG.add_edge(4, 1, weight=-1)
        DG.add_edge(1, 5, weight=1)
        DG.add_edge(5, 1, weight=-1)
        DG.add_edge(2, 1, weight=1)
        self.assertEqual(network_utils.cartwright_harary_balance(DG), 0.5)

    def test_count_different_signed_edges(self):
        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3])
        DG.add_edge(1, 2, weight=1)
        DG.add_edge(2, 1, weight=1)
        DG.add_edge(3, 1, weight=-5)
        DG.add_edge(1, 3, weight=-2)
        self.assertEqual(network_utils.count_different_signed_edges(DG), 0)

        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3])
        DG.add_edge(1, 2, weight=3)
        DG.add_edge(2, 1, weight=4)
        DG.add_edge(3, 1, weight=1)
        DG.add_edge(1, 3, weight=-1)
        self.assertEqual(network_utils.count_different_signed_edges(DG), 1)

        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3])
        DG.add_edge(1, 2, weight=-1)
        DG.add_edge(2, 1, weight=1)
        DG.add_edge(3, 1, weight=9)
        DG.add_edge(1, 3, weight=-2)
        self.assertEqual(network_utils.count_different_signed_edges(DG), 2)

    # =========================================================================
    # ====================== compute_edge_balance =============================
    # =========================================================================
    @parameterized.expand(
        [["no_isomorph_cycles", False], ["no_isomorph_cycles", True]])
    def test_compute_edge_balance_small_graph(self, name, no_isomorph_cycles):
        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3])
        DG.add_edge(1, 2, weight=1)
        DG.add_edge(2, 1, weight=1)
        DG.add_edge(2, 3, weight=-5)
        DG.add_edge(3, 1, weight=-2)
        if no_isomorph_cycles:
            expected = {
                (1, 2): {
                    '#balanced': 1,
                    '#cycle3': 1,
                    'weight_distance': 9,
                    'as_expected_sign': True}}
        else:
            expected = {
                (1, 2): {
                    '#balanced': 1,
                    '#cycle3': 1,
                    'weight_distance': 9,
                    'as_expected_sign': True},
                (3, 1): {
                    '#balanced': 1,
                    '#cycle3': 1,
                    'weight_distance': 3,
                    'as_expected_sign': True},
                (2, 3): {
                    '#balanced': 1,
                    '#cycle3': 1,
                    'weight_distance': 3,
                    'as_expected_sign': True}}
        computed = network_utils.compute_edge_balance(
            DG, no_isomorph_cycles=no_isomorph_cycles)
        self.assertDictEqual(computed, expected)

    @parameterized.expand(
        [["no_isomorph_cycles", False],
            ["no_isomorph_cycles", True]])
    def test_compute_edge_balance_allnegative_graph(
            self, name, no_isomorph_cycles):
        DG = nx.DiGraph()
        DG.add_nodes_from([1, 2, 3, 4])
        DG.add_edge(1, 2, weight=-1)
        DG.add_edge(2, 3, weight=-1)
        DG.add_edge(3, 1, weight=-1)
        DG.add_edge(1, 4, weight=-5)
        DG.add_edge(4, 3, weight=-2)
        if no_isomorph_cycles:
            expected = {
                (1, 2): {
                    '#balanced': 0,
                    '#cycle3': 1,
                    'weight_distance': 2,
                    'as_expected_sign': False},
                (1, 4): {
                    '#balanced': 0,
                    '#cycle3': 1,
                    'weight_distance': 7,
                    'as_expected_sign': False}}
        else:
            expected = {
                (1, 2): {
                    '#balanced': 0,
                    '#cycle3': 1,
                    'weight_distance': 2,
                    'as_expected_sign': False},
                (1, 4): {
                    '#balanced': 0,
                    '#cycle3': 1,
                    'weight_distance': 7,
                    'as_expected_sign': False},
                (2, 3): {
                    '#balanced': 0,
                    '#cycle3': 1,
                    'weight_distance': 2,
                    'as_expected_sign': False},
                (3, 1): {
                    '#balanced': 0,
                    '#cycle3': 2,
                    'weight_distance': 13,
                    'as_expected_sign': False},
                (4, 3): {
                    '#balanced': 0,
                    '#cycle3': 1,
                    'weight_distance': 7,
                    'as_expected_sign': False}}

        computed = network_utils.compute_edge_balance(
            DG, no_isomorph_cycles=no_isomorph_cycles)
        self.assertDictEqual(computed, expected)


if __name__ == '__main__':
    unittest.main()
