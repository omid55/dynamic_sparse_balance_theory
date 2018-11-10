# Omid55
# Test module for network_utils.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import unittest
from parameterized import parameterized
import networkx as nx
import pandas as pd
import numpy as np
import datetime
import re

import utils
import network_utils


class MyTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.triad_map, cls.triad_list = (
            network_utils.generate_all_possible_sparse_triads())

    @classmethod
    def tearDownClass(cls):
        del cls.triad_map
        del cls.triad_list

    # =========================================================================
    # ==================== extract_graphs =====================================
    # =========================================================================
    def test_extract_graphs_raises_with_missing_columns(self):
        sample_edge_list = pd.DataFrame({'source': [1, 2], 'target': [5, 6]})
        with self.assertRaises(ValueError):
            network_utils.extract_graphs(edge_list=sample_edge_list)

    @parameterized.expand(
        [["seperated graphs", False], ["accumulative graphs", True]])
    def test_extract_graphs(self, name, accumulative):
        # source, target, weight, edge_date
        matrix_edges = [[1, 2, +1, datetime.datetime(2017, 1, 1)],
                        [2, 3, +1, datetime.datetime(2017, 1, 4)],
                        [3, 1, +1, datetime.datetime(2017, 2, 5)],
                        [1, 4, -1, datetime.datetime(2017, 2, 13)],
                        [4, 3, -1, datetime.datetime(2017, 2, 24)],
                        [-1, -1, -1, datetime.datetime(2017, 2, 28)]]
        # The last one is going to be ignored because fall into another period
        #   which is neglected.

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
        if not accumulative:
            expected = [g1, g2]
        else:
            expected = [g1, g3]
        computed = network_utils.extract_graphs(
            edge_list=sample_edge_list, weeks=4, accumulative=accumulative)
        for expected_graph, computed_graph in zip(expected, computed):
            self.assertTrue(utils.graph_equals(expected_graph, computed_graph))

    # =========================================================================
    # ==================== get_metrics_for_network ============================
    # =========================================================================
    def test_get_metrics_for_network(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=-1)
        dg.add_edge(1, 3, weight=-1)
        computed = network_utils.get_metrics_for_network(dg)
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
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance(dg), 0)

        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=-1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance(dg), 0)

        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance(dg), 1)

        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4, 5])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        dg.add_edge(3, 4, weight=-1)
        dg.add_edge(4, 1, weight=-1)
        dg.add_edge(1, 5, weight=1)
        dg.add_edge(5, 1, weight=-1)
        dg.add_edge(2, 1, weight=1)
        self.assertEqual(network_utils.cartwright_harary_balance(dg), 0.5)

    def test_count_different_signed_edges(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(3, 1, weight=-5)
        dg.add_edge(1, 3, weight=-2)
        self.assertEqual(network_utils.count_different_signed_edges(dg), 0)

        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=3)
        dg.add_edge(2, 1, weight=4)
        dg.add_edge(3, 1, weight=1)
        dg.add_edge(1, 3, weight=-1)
        self.assertEqual(network_utils.count_different_signed_edges(dg), 1)

        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=-1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(3, 1, weight=9)
        dg.add_edge(1, 3, weight=-2)
        self.assertEqual(network_utils.count_different_signed_edges(dg), 2)

    # =========================================================================
    # ====================== compute_edge_balance =============================
    # =========================================================================
    @parameterized.expand(
        [["no_isomorph_cycles", False], ["no_isomorph_cycles", True]])
    def test_compute_edge_balance_small_graph(self, name, no_isomorph_cycles):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(2, 3, weight=-5)
        dg.add_edge(3, 1, weight=-2)
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
            dg, no_isomorph_cycles=no_isomorph_cycles)
        self.assertDictEqual(computed, expected)

    @parameterized.expand(
        [["no_isomorph_cycles", False],
            ["no_isomorph_cycles", True]])
    def test_compute_edge_balance_allnegative_graph(
            self, name, no_isomorph_cycles):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        dg.add_edge(1, 2, weight=-1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        dg.add_edge(1, 4, weight=-5)
        dg.add_edge(4, 3, weight=-2)
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
            dg, no_isomorph_cycles=no_isomorph_cycles)
        self.assertDictEqual(computed, expected)

    # =========================================================================
    # ====================== compute_fairness_goodness ========================
    # =========================================================================
    def test_compute_fairness_goodness(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        dg.add_edge(1, 2, weight=1.0)
        dg.add_edge(2, 3, weight=1.0)
        dg.add_edge(3, 1, weight=1.0)
        dg.add_edge(1, 4, weight=2.0)
        dg.add_edge(4, 3, weight=-1.0)
        expected = {'fairness': {1: 1.0, 2: 0.95, 3: 1.0, 4: 0.95},
                    'goodness': {1: 1.0, 2: 1.0, 3: 0.0, 4: 2.0}}
        computed = network_utils.compute_fairness_goodness(dg, verbose=False)
        self.assertDictEqual(computed, expected)

    # =========================================================================
    # ====================== is_transitive_balanced ===========================
    # =========================================================================
    def test_is_transitive_balanced_raises_when_self_loops(self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]])
            network_utils.is_transitive_balanced(triad_with_self_loop)

    @parameterized.expand([
        ["030T", np.array(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]), True],
        ["030Tneg", np.array(
            [[0, 1, -1],
             [0, 0, 1],
             [0, 0, 0]]), False],
        ["030T2neg", np.array(
            [[0, 1, -1],
             [0, 0, -1],
             [0, 0, 0]]), True],
        ["021Uneg", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, -1, 0]]), True],
        ["021D", np.array(
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]]), True],
        ["210", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 1, 0]]), False],
        ["003", np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]), True]]
        )
    def test_is_transitive_balanced(self, name, triad, expected_balance):
        self.assertEqual(
            network_utils.is_transitive_balanced(triad), expected_balance)

    # =========================================================================
    # ====================== get_all_triad_permutations =======================
    # =========================================================================
    def test_get_all_triad_permutations(self):
        triad_adj_matrix = np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 1, 0]])
        expected = set([
            '[[0 1 1]\n [1 0 1]\n [1 0 0]]',
            '[[0 0 1]\n [1 0 1]\n [1 1 0]]',
            '[[0 1 1]\n [0 0 1]\n [1 1 0]]',
            '[[0 1 1]\n [1 0 1]\n [0 1 0]]',
            '[[0 1 1]\n [1 0 0]\n [1 1 0]]',
            '[[0 1 0]\n [1 0 1]\n [1 1 0]]'])
        computed = network_utils._get_all_triad_permutations(triad_adj_matrix)
        self.assertEqual(expected, computed)

    # =========================================================================
    # ====================== generate_all_possible_sparse_triads ==============
    # =========================================================================
    def test_generate_all_possible_sparse_triads(self):
        computed_triad_map, computed_triad_list = (
            network_utils.generate_all_possible_sparse_triads())

        # Testing triad_list
        self.assertTrue(
            len(computed_triad_list) == 138,
            'Length of triad_list is not correct.')
        np.testing.assert_array_equal(
            computed_triad_list[0], np.array(
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]), 'First triad_list is incorrect.')
        np.testing.assert_array_equal(
            computed_triad_list[-1], np.array(
                [[0, -1, -1],
                 [-1,  0, -1],
                 [-1, -1,  0]]), 'Last triad_list is incorrect.')
        np.testing.assert_array_equal(
            computed_triad_list[69], np.array(
                [[0,  0,  1],
                 [1,  0, -1],
                 [1,  0,  0]]), 'Middle triad_list is incorrect.')

        # Testing triad_map.
        expected_key1 = '[[0 0 0]\n [1 0 0]\n [0 0 0]]'
        expected_value1 = 1
        expected_key2 = '[[ 0  1  1]\n [-1  0  1]\n [-1 -1  0]]'
        expected_value2 = 129

        self.assertTrue(
            expected_key1 in computed_triad_map,
            'First key was not found in computed_triad_map.')
        self.assertTrue(
            expected_key2 in computed_triad_map,
            'Second key was not found in computed_triad_map.')
        self.assertEqual(
            computed_triad_map[expected_key1], expected_value1,
            'First value was not found in computed_triad_map.')
        self.assertEqual(
            computed_triad_map[expected_key2], expected_value2,
            'Second value was not found in computed_triad_map.')

    # =========================================================================
    # ====================== detect_triad_type_for_all_subgraph3 ==============
    # =========================================================================
    def test_detect_triad_type_for_all_subgraph3(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=1)
        dg.add_edge(1, 4, weight=2)
        dg.add_edge(4, 3, weight=-5)
        expected = {
            '(1, 2, 3)': 55,
            # [[0, 0, 1],
            #  [1, 0, 0],
            #  [0, 1, 0]]
            '(1, 2, 4)': 3,
            # [[0, 0, 0],
            #  [0, 0, 0],
            #  [1, 1, 0]]
            '(1, 3, 4)': 56,
            # [[0, 0, 1],
            #  [1, 0, 0],
            #  [0,-1, 0]]
            '(2, 3, 4)': 24
            # [[0, 0, 0],
            #  [1, 0, 0],
            #  [-1, 0, 0]]
        }
        computed = network_utils._detect_triad_type_for_all_subgraph3(
            dgraph=dg, triad_map=self.triad_map)
        self.assertDictEqual(expected, computed)

    def test_detect_triad_type_for_all_subgraph3_nodes_with_str_name(self):
        dg = nx.DiGraph()
        dg.add_nodes_from(['b', 'c', 'a', 'd'])
        dg.add_edge('b', 'c', weight=1)
        dg.add_edge('c', 'a', weight=1)
        dg.add_edge('a', 'b', weight=1)
        dg.add_edge('b', 'd', weight=2)
        dg.add_edge('d', 'a', weight=-5)
        expected = {
            "('a', 'b', 'c')": 55,
            "('a', 'b', 'd')": 56,
            "('a', 'c', 'd')": 24,
            "('b', 'c', 'd')": 3
        }
        computed = network_utils._detect_triad_type_for_all_subgraph3(
            dgraph=dg, triad_map=self.triad_map)
        self.assertDictEqual(expected, computed)

    def test_detect_triad_type_for_all_subgraph3_has_unique_keys(self):
        dg = nx.DiGraph()
        dg.add_nodes_from(['b', 'c', 'a', 'd'])
        dg.add_edge('b', 'c', weight=1)
        dg.add_edge('c', 'a', weight=1)
        dg.add_edge('a', 'b', weight=1)
        dg.add_edge('b', 'd', weight=2)
        dg.add_edge('d', 'a', weight=-5)
        computed = network_utils._detect_triad_type_for_all_subgraph3(
            dgraph=dg, triad_map=self.triad_map)
        truncated_keys = []
        for key in list(computed.keys()):
            key = re.sub(r'[^\w]', ' ', key)
            key = key.replace(" ", "")
            truncated_keys.append(''.join(sorted(key)))
        self.assertEqual(len(truncated_keys), len(np.unique(truncated_keys)))

    # =========================================================================
    # ====================== compute_transition_matrix ========================
    # =========================================================================
    def test_compute_transition_matrix(self):
        dg1 = nx.DiGraph()
        dg1.add_nodes_from([1, 2, 3, 4])
        dg1.add_edge(1, 2, weight=1)
        dg1.add_edge(2, 1, weight=1)
        dg1.add_edge(2, 3, weight=1)
        dg1.add_edge(3, 1, weight=-1)
        dg1.add_edge(3, 4, weight=1)
        dg2 = nx.DiGraph()
        dg2.add_nodes_from([1, 2, 3, 4])
        dg2.add_edge(1, 2, weight=1)
        dg2.add_edge(1, 3, weight=1)
        dg2.add_edge(2, 1, weight=1)
        dg2.add_edge(2, 3, weight=1)
        dg2.add_edge(2, 4, weight=1)
        dg2.add_edge(3, 1, weight=1)
        dg2.add_edge(3, 4, weight=1)
        dg2.add_edge(4, 1, weight=1)
        dgraphs = [dg1, dg2]
        triads_types = [
                {'(1, 2, 3)': 76,
                 '(1, 2, 4)': 6,
                 '(1, 3, 4)': 4,
                 '(2, 3, 4)': 8},
                {'(1, 2, 3)': 63,
                 '(1, 2, 4)': 57,
                 '(1, 3, 4)': 57,
                 '(2, 3, 4)': 22}]
        n = len(self.triad_list)
        transition_matrix = np.zeros((n, n))
        transition_matrix[76, 63] = 1
        transition_matrix[6, 57] = 1
        transition_matrix[4, 57] = 1
        transition_matrix[8, 22] = 1
        computed = network_utils.compute_transition_matrix(
            dgraphs=dgraphs,
            unique_triad_num=n,
            triad_map=self.triad_map)
        # self.assertDictEqual(expected, computed)
        self.assertTrue(
            'triads_types' in computed,
            'triads_types was not found in computed transition matrix.')
        self.assertTrue(
            'transition_matrices' in computed,
            'transition_matrices was not found in computed transition matrix.')
        self.assertEqual(
            triads_types,
            computed['triads_types'],
            'Triad types were different.')
        np.testing.assert_array_equal(
            transition_matrix,
            computed['transition_matrices'][0],
            'Transition matrices were different.')

    # =========================================================================
    # ====================== get_stationary_distribution ======================
    # =========================================================================
    def test_get_stationary_distribution_simple(self):
        transition_matrix = np.array(
            [[0, 0, 1],
             [0, 0, 1],
             [0, 0, 1]], dtype=float)
        expected = np.array([0, 0, 1])
        computed = network_utils.get_stationary_distribution(
            transition_matrix, aperiodic_irreducible_eps=0.0)
        np.testing.assert_array_almost_equal(expected, computed, decimal=4)

    def test_get_stationary_distribution_full_matrix(self):
        transition_matrix = np.array(
            [[0.6, 0.1, 0.3],
             [0.1, 0.7, 0.2],
             [0.2, 0.2, 0.6]], dtype=float)
        expected = np.array([0.2759, 0.3448, 0.3793])
        computed = network_utils.get_stationary_distribution(
            transition_matrix, aperiodic_irreducible_eps=0.0)
        np.testing.assert_array_almost_equal(expected, computed, decimal=4)

    def test_get_stationary_distribution_not_row_stochastic(self):
        transition_matrix = np.array(
            [[0, 0, 0],
             [9, 0, 1],
             [1, 0, 3]], dtype=float)
        expected = np.array([0.3571, 0.1191, 0.5238])
        computed = network_utils.get_stationary_distribution(
            transition_matrix, aperiodic_irreducible_eps=0.0001)
        np.testing.assert_array_almost_equal(expected, computed, decimal=4)

    def test_get_stationary_distribution(self):
        transition_matrix = np.array(
            [[0, 0, 0],
             [0.9, 0, 0.1],
             [0.25, 0, 0.75]], dtype=float)
        expected = np.array([0.3571, 0.1191, 0.5238])
        computed = network_utils.get_stationary_distribution(
            transition_matrix, aperiodic_irreducible_eps=0.0001)
        np.testing.assert_array_almost_equal(expected, computed, decimal=4)

    # =========================================================================
    # ====================== get_mixing_time_range ============================
    # =========================================================================
    def test_get_mixing_time_range(self):
        transition_matrix = np.array(
            [[0, 0, 0],
             [0.9, 0, 0.1],
             [0.25, 0, 0.75]], dtype=float)
        expected = 13.7081
        computed = network_utils.get_mixing_time_range(
            transition_matrix,
            aperiodic_irreducible_eps=0.0001,
            distance_from_stationary_eps=0.01)
        self.assertEqual(np.round(expected, 4), np.round(computed, 4))

    # =========================================================================
    # ====================== randomize_network ================================
    # =========================================================================
    def test_randomize_network_with_unweighted_graph(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4, 5, 6])
        dg.add_edge(1, 2)
        dg.add_edge(2, 1)
        dg.add_edge(2, 3)
        dg.add_edge(3, 1)
        dg.add_edge(3, 4)
        dg.add_edge(4, 5)
        dg.add_edge(5, 4)
        dg.add_edge(1, 6)
        dg.add_edge(6, 1)
        dg.add_edge(6, 5)
        computed = network_utils.randomize_network(dg, switching_count_coef=2)
        self.assertEqual(
            sorted(dict(dg.degree()).values()),
            sorted(dict(computed.degree()).values()))

    def test_randomize_network_with_all_positive_weighted_graph(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4, 5, 6])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=2)
        dg.add_edge(3, 4, weight=5)
        dg.add_edge(4, 5, weight=9)
        dg.add_edge(5, 4, weight=6)
        dg.add_edge(1, 6, weight=9)
        dg.add_edge(6, 1, weight=1)
        dg.add_edge(6, 5, weight=16)
        computed = network_utils.randomize_network(dg, switching_count_coef=2)
        self.assertEqual(
            sorted(dict(dg.degree()).values()),
            sorted(dict(computed.degree()).values()))

    def test_randomize_network_with_signed_weighted_graph(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4, 5, 6])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=-2)
        dg.add_edge(3, 4, weight=5)
        dg.add_edge(4, 5, weight=9)
        dg.add_edge(5, 4, weight=-6)
        dg.add_edge(1, 6, weight=-9)
        dg.add_edge(6, 1, weight=1)
        dg.add_edge(6, 5, weight=-16)
        computed = network_utils.randomize_network(dg, switching_count_coef=2)
        self.assertEqual(
            sorted(dict(dg.degree()).values()),
            sorted(dict(computed.degree()).values()))


if __name__ == '__main__':
    unittest.main()
