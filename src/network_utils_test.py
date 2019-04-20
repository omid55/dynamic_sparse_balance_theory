# Omid55
# Test module for network_utils.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import networkx as nx
import pandas as pd
import numpy as np
import unittest
import datetime
import re
from parameterized import parameterized

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
    # ==================== extract_graph ======================================
    # =========================================================================
    @parameterized.expand([
        ["latest_multiple_edge_weight", False],
        ["sum_of_multiple_edge_weights", True]])
    def test_extract_graph(self, name, sum_multiple_edge):
        matrix_edges = [
            [1, 2, +1, datetime.datetime(2017, 1, 1)],
            [1, 2, +5, datetime.datetime(2017, 1, 2)],
            [2, 3, +3, datetime.datetime(2017, 1, 4)],
            [3, 1, +1, datetime.datetime(2017, 2, 5)],
            [2, 3, -2, datetime.datetime(2017, 1, 6)],
            [1, 4, -1, datetime.datetime(2017, 2, 13)],
            [4, 3, -5, datetime.datetime(2017, 2, 22)],
            [4, 3, -5, datetime.datetime(2017, 2, 24)]]
        sample_edge_list = pd.DataFrame(
            matrix_edges, columns=['source', 'target', 'weight', 'edge_date'])
        expected = nx.DiGraph()
        expected.add_nodes_from([1, 2, 3, 4])
        if sum_multiple_edge:
            expected.add_edge(1, 2, weight=6)
        else:
            expected.add_edge(1, 2, weight=5)
        if sum_multiple_edge:
            expected.add_edge(2, 3, weight=1)
        else:
            expected.add_edge(2, 3, weight=-2)
        expected.add_edge(3, 1, weight=1)
        expected.add_edge(1, 4, weight=-1)
        if sum_multiple_edge:
            expected.add_edge(4, 3, weight=-10)
        else:
            expected.add_edge(4, 3, weight=-5)
        computed = network_utils.extract_graph(
            sample_edge_list, sum_multiple_edge=sum_multiple_edge)
        self.assertTrue(
            utils.graph_equals(
                expected,
                computed,
                weight_column_name='weight'))

    # =========================================================================
    # ==================== extract_graphs =====================================
    # =========================================================================
    def test_extract_graphs_raises_with_missing_columns(self):
        sample_edge_list = pd.DataFrame({'source': [1, 2], 'target': [5, 6]})
        with self.assertRaises(ValueError):
            network_utils.extract_graphs(edge_list=sample_edge_list)

    @parameterized.expand(
        [["seperated graphs", False],
         ["accumulative graphs", True]])
    def test_extract_graphs(self, name, accumulative):
        # source, target, weight, edge_date
        matrix_edges = [[1, 2, +1, datetime.datetime(2017, 1, 1)],
                        [2, 3, +3, datetime.datetime(2017, 1, 4)],
                        [3, 1, +1, datetime.datetime(2017, 2, 5)],
                        [1, 4, -1, datetime.datetime(2017, 2, 13)],
                        [4, 3, -5, datetime.datetime(2017, 2, 24)],
                        [-1, -1, -1, datetime.datetime(2017, 2, 28)]]
        # The last one is going to be ignored because fall into another period
        #   which is neglected.

        sample_edge_list = pd.DataFrame(
            matrix_edges, columns=['source', 'target', 'weight', 'edge_date'])
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2, weight=1)
        g1.add_edge(2, 3, weight=3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 3, 4])
        g2.add_edge(3, 1, weight=1)
        g2.add_edge(1, 4, weight=-1)
        g2.add_edge(4, 3, weight=-5)
        g3 = nx.DiGraph()
        g3.add_nodes_from([1, 2, 3, 4])
        g3.add_edge(1, 2, weight=1)
        g3.add_edge(2, 3, weight=3)
        g3.add_edge(3, 1, weight=1)
        g3.add_edge(1, 4, weight=-1)
        g3.add_edge(4, 3, weight=-5)
        if not accumulative:
            expected = [g1, g2]
        else:
            expected = [g1, g3]
        computed = network_utils.extract_graphs(
            edge_list=sample_edge_list, weeks=4, accumulative=accumulative)
        for expected_graph, computed_graph in zip(expected, computed):
            self.assertTrue(
                utils.graph_equals(
                    expected_graph,
                    computed_graph,
                    weight_column_name='weight'))

    # =========================================================================
    # ====================== get_all_degrees ==================================
    # =========================================================================
    def test_get_all_degrees(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4, 5])
        dg.add_edge(1, 1, weight=6)
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(1, 4, weight=-5)
        dg.add_edge(2, 2, weight=-1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=-4)
        dg.add_edge(3, 2, weight=4)
        dg.add_edge(4, 4, weight=-10)
        computed = network_utils.get_all_degrees(dg)
        expected = (
            {1: {'self': 6, 'out': -4, 'in': -4},
             2: {'self': -1, 'out': 1, 'in': 5},
             3: {'self': 0, 'out': 0, 'in': 1},
             4: {'self': -10, 'out': 0, 'in': -5},
             5: {'self': 0, 'out': 0, 'in': 0}})
        self.assertDictEqual(computed, expected)

    # =========================================================================
    # ===================== get_just_periods ==================================
    # =========================================================================
    def test_get_just_periods(self):
        matrix_edges = [[1, 2, +1, datetime.datetime(2017, 1, 1)],
                        [2, 3, +3, datetime.datetime(2017, 1, 4)],
                        [3, 1, +1, datetime.datetime(2017, 2, 5)],
                        [1, 4, -1, datetime.datetime(2017, 2, 13)],
                        [4, 3, -5, datetime.datetime(2017, 2, 24)],
                        [-1, -1, -1, datetime.datetime(2017, 2, 28)]]
        sample_edge_list = pd.DataFrame(
            matrix_edges, columns=['source', 'target', 'weight', 'edge_date'])
        expected = [['2017-01-01', '2017-01-29'], ['2017-01-29', '2017-02-26']]
        computed = network_utils.get_just_periods(
            sample_edge_list, weeks=4, accumulative=False)
        self.assertEqual(expected, computed)

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
        # utils.print_dict_pretty(computed)
#         self.assertDictEqual(computed, expected)
        for key, value in expected.items():
            self.assertAlmostEqual(value, computed[key], places=3)

    # =========================================================================
    # ====================== cartwright_harary_balance_ratio ==================
    # =========================================================================
    def test_cartwright_harary_balance_ratio_notbalanced_graph1(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance_ratio(dg), 0)

    def test_cartwright_harary_balance_ratio_notbalanced_graph2(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=-1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance_ratio(dg), 0)

    def test_cartwright_harary_balance_ratio_balanced_graph(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        self.assertEqual(network_utils.cartwright_harary_balance_ratio(dg), 1)

    def test_cartwright_harary_balance_ratio_halfbalanced_graph(self):
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
        self.assertEqual(
            network_utils.cartwright_harary_balance_ratio(dg), 0.5)

    # =========================================================================
    # ========================= sprase_balance_ratio ==========================
    # =========================================================================
    def test_sparse_balance_ratio_raises_when_incorrect_balance_type(self):
        with self.assertRaises(ValueError):
            network_utils.sprase_balance_ratio(
                dgraph=nx.DiGraph(),
                balance_type=0)

    @parameterized.expand([
        ['CartwrightHarary', 1, [0.3, 3, 7]],
        ['Clustering', 2, [0.5, 5, 5]],
        ['Transitivity', 3, [0.9, 9, 1]]])
    def test_sprase_balance_ratio(
            self,
            name,
            balance_type,
            expected_values):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4, 5])
        dg.add_edge(1, 2, weight=5)
        dg.add_edge(2, 3, weight=-4)
        dg.add_edge(3, 1, weight=-7)
        dg.add_edge(3, 4, weight=-1)
        dg.add_edge(4, 1, weight=-2)
        dg.add_edge(1, 5, weight=9)
        dg.add_edge(5, 1, weight=-11)
        dg.add_edge(2, 1, weight=100)
        computed = network_utils.sprase_balance_ratio(
            dgraph=dg,
            balance_type=balance_type)
        np.testing.assert_array_almost_equal(
            computed, expected_values, decimal=2)

    # =========================================================================
    # ======================= classical_balance_ratio =========================
    # =========================================================================
    def test_classical_balance_ratio_raises_when_incorrect_balance_type(self):
        with self.assertRaises(ValueError):
            network_utils.classical_balance_ratio(
                dgraph=nx.DiGraph(),
                balance_type=0)

    def test_classical_balance_ratio_raises_when_negative_in_dgraph(self):
        with self.assertRaises(ValueError):
            dg = nx.DiGraph()
            dg.add_nodes_from([1, 2])
            dg.add_edge(1, 2, weight=1)
            dg.add_edge(2, 1, weight=-1)
            network_utils.classical_balance_ratio(
                dgraph=dg,
                balance_type=1)

    @parameterized.expand([
        ['Classical', 1, [0.4, 4, 6]],
        ['Clustering', 2, [0.7, 7, 3]],
        ['Transitivity', 3, [0.8, 8, 2]]])
    def test_classical_balance_ratio(
            self,
            name,
            balance_type,
            expected_values):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4, 5])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(5, 1, weight=1)
        dg.add_edge(1, 5, weight=1)
        dg.add_edge(5, 2, weight=1)
        dg.add_edge(2, 5, weight=1)
        dg.add_edge(5, 3, weight=1)
        dg.add_edge(2, 3, weight=1)
        computed = network_utils.classical_balance_ratio(
            dgraph=dg,
            balance_type=balance_type)
        np.testing.assert_array_almost_equal(
            computed, expected_values, decimal=2)

    # =========================================================================
    # ====================== count_different_signed_edges =====================
    # =========================================================================
    def test_count_different_signed_edges(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(3, 1, weight=-5)
        dg.add_edge(1, 3, weight=-2)
        self.assertEqual(network_utils.count_different_signed_edges(dg), 0)

    def test_count_different_signed_edges1(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=3)
        dg.add_edge(2, 1, weight=4)
        dg.add_edge(3, 1, weight=1)
        dg.add_edge(1, 3, weight=-1)
        self.assertEqual(network_utils.count_different_signed_edges(dg), 1)

    def test_count_different_signed_edges2(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=-1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(3, 1, weight=9)
        dg.add_edge(1, 3, weight=-2)
        self.assertEqual(network_utils.count_different_signed_edges(dg), 2)

    # =========================================================================
    # ==================== terzi_sprase_balance_ratio =========================
    # =========================================================================
    def test_terzi_sprase_balance_ratio_notbalanced_graph1(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=-1)
        expected = 0
        computed = network_utils.terzi_sprase_balance_ratio(
            dg, undirected=True)
        np.testing.assert_almost_equal(computed, expected)

    def test_terzi_sprase_balance_ratio_notbalanced_graph2(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=-1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        expected = 0
        computed = network_utils.terzi_sprase_balance_ratio(
            dg, undirected=True)
        np.testing.assert_almost_equal(computed, expected)

    def test_terzi_sprase_balance_ratio_balanced_graph(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        expected = 1
        computed = network_utils.terzi_sprase_balance_ratio(
            dg, undirected=True)
        np.testing.assert_almost_equal(computed, expected)

    def test_terzi_sprase_balance_ratio_halfbalanced_graph(self):
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
        expected = 0.5
        computed = network_utils.terzi_sprase_balance_ratio(
            dg, undirected=True)
        np.testing.assert_almost_equal(computed, expected)

    # =========================================================================
    # ================= kunegis_sprase_balance_ratio ==========================
    # =========================================================================
    def test_kunegis_sprase_balance_ratio_notbalanced_graph1(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=1)
        dg.add_edge(3, 1, weight=-1)
        expected = 0
        computed = network_utils.kunegis_sprase_balance_ratio(
            dg, undirected=True)
        np.testing.assert_almost_equal(computed, expected)

    def test_kunegis_sprase_balance_ratio_notbalanced_graph2(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=-1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        expected = 0
        computed = network_utils.kunegis_sprase_balance_ratio(
            dg, undirected=True)
        np.testing.assert_almost_equal(computed, expected)

    def test_kunegis_sprase_balance_ratio_balanced_graph(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(3, 1, weight=-1)
        expected = 1
        computed = network_utils.kunegis_sprase_balance_ratio(
            dg, undirected=True)
        np.testing.assert_almost_equal(computed, expected)

    def test_kunegis_sprase_balance_ratio_halfbalanced_graph(self):
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
        expected = 0.6
        computed = network_utils.kunegis_sprase_balance_ratio(
            dg, undirected=True)
        np.testing.assert_almost_equal(computed, expected, decimal=1)

    # =========================================================================
    # ====================== compute_vanderijt_edge_balance ===================
    # =========================================================================
    def test_compute_vanderijt_edge_balance_small_graph(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(2, 3, weight=-5)
        dg.add_edge(3, 1, weight=-2)
        expected = {(2, 1): {'#nodes3': 1, '#balanced_node3': 1}}
        computed = network_utils.compute_vanderijt_edge_balance(dg)
        self.assertDictEqual(computed, expected)

    def test_compute_vanderijt_edge_balance_allnegative_graph(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        dg.add_edge(1, 2, weight=-1)
        dg.add_edge(2, 3, weight=-1)
        dg.add_edge(1, 3, weight=-1)
        dg.add_edge(2, 4, weight=-1)
        dg.add_edge(4, 2, weight=-1)
        dg.add_edge(2, 1, weight=1)
        dg.add_edge(3, 2, weight=1)
        dg.add_edge(3, 1, weight=-1)
        dg.add_edge(4, 1, weight=-5)
        dg.add_edge(1, 4, weight=-5)
        dg.add_edge(4, 3, weight=-2)
        dg.add_edge(3, 4, weight=1)
        expected = {
            (1, 2): {'#balanced_node3': 1, '#nodes3': 2},
            (3, 2): {'#balanced_node3': 1, '#nodes3': 2},
            (1, 3): {'#balanced_node3': 0, '#nodes3': 2},
            (3, 4): {'#balanced_node3': 1, '#nodes3': 2},
            (3, 1): {'#balanced_node3': 1, '#nodes3': 2},
            (1, 4): {'#balanced_node3': 1, '#nodes3': 2},
            (2, 3): {'#balanced_node3': 1, '#nodes3': 2},
            (2, 1): {'#balanced_node3': 2, '#nodes3': 2},
            (4, 3): {'#balanced_node3': 0, '#nodes3': 2},
            (4, 2): {'#balanced_node3': 1, '#nodes3': 2},
            (4, 1): {'#balanced_node3': 1, '#nodes3': 2},
            (2, 4): {'#balanced_node3': 2, '#nodes3': 2}}
        computed = network_utils.compute_vanderijt_edge_balance(dg)
        self.assertDictEqual(computed, expected)

    # @parameterized.expand(
    #     [["no_isomorph_cycles", False], ["no_isomorph_cycles", True]])
    # def test_compute_vanderijt_edge_balance_small_graph(
    #         self, name, no_isomorph_cycles):
    #     dg = nx.DiGraph()
    #     dg.add_nodes_from([1, 2, 3])
    #     dg.add_edge(1, 2, weight=1)
    #     dg.add_edge(2, 1, weight=1)
    #     dg.add_edge(2, 3, weight=-5)
    #     dg.add_edge(3, 1, weight=-2)
    #     if no_isomorph_cycles:
    #         expected = {
    #             (1, 2): {
    #                 '#balanced': 1,
    #                 '#cycle3': 1,
    #                 'weight_distance': 9,
    #                 'as_expected_sign': True}}
    #     else:
    #         expected = {
    #             (1, 2): {
    #                 '#balanced': 1,
    #                 '#cycle3': 1,
    #                 'weight_distance': 9,
    #                 'as_expected_sign': True},
    #             (3, 1): {
    #                 '#balanced': 1,
    #                 '#cycle3': 1,
    #                 'weight_distance': 3,
    #                 'as_expected_sign': True},
    #             (2, 3): {
    #                 '#balanced': 1,
    #                 '#cycle3': 1,
    #                 'weight_distance': 3,
    #                 'as_expected_sign': True}}
    #     computed = network_utils.compute_vanderijt_edge_balance(
    #         dg, no_isomorph_cycles=no_isomorph_cycles)
    #     self.assertDictEqual(computed, expected)

    # @parameterized.expand(
    #     [["no_isomorph_cycles", False],
    #         ["no_isomorph_cycles", True]])
    # def test_compute_vanderijt_edge_balance_allnegative_graph(
    #         self, name, no_isomorph_cycles):
    #     dg = nx.DiGraph()
    #     dg.add_nodes_from([1, 2, 3, 4])
    #     dg.add_edge(1, 2, weight=-1)
    #     dg.add_edge(2, 3, weight=-1)
    #     dg.add_edge(3, 1, weight=-1)
    #     dg.add_edge(1, 4, weight=-5)
    #     dg.add_edge(4, 3, weight=-2)
    #     if no_isomorph_cycles:
    #         expected = {
    #             (1, 2): {
    #                 '#balanced': 0,
    #                 '#cycle3': 1,
    #                 'weight_distance': 2,
    #                 'as_expected_sign': False},
    #             (1, 4): {
    #                 '#balanced': 0,
    #                 '#cycle3': 1,
    #                 'weight_distance': 7,
    #                 'as_expected_sign': False}}
    #     else:
    #         expected = {
    #             (1, 2): {
    #                 '#balanced': 0,
    #                 '#cycle3': 1,
    #                 'weight_distance': 2,
    #                 'as_expected_sign': False},
    #             (1, 4): {
    #                 '#balanced': 0,
    #                 '#cycle3': 1,
    #                 'weight_distance': 7,
    #                 'as_expected_sign': False},
    #             (2, 3): {
    #                 '#balanced': 0,
    #                 '#cycle3': 1,
    #                 'weight_distance': 2,
    #                 'as_expected_sign': False},
    #             (3, 1): {
    #                 '#balanced': 0,
    #                 '#cycle3': 2,
    #                 'weight_distance': 13,
    #                 'as_expected_sign': False},
    #             (4, 3): {
    #                 '#balanced': 0,
    #                 '#cycle3': 1,
    #                 'weight_distance': 7,
    #                 'as_expected_sign': False}}

    #     computed = network_utils.compute_vanderijt_edge_balance(
    #         dg, no_isomorph_cycles=no_isomorph_cycles)
    #     self.assertDictEqual(computed, expected)

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
    # ====================== is_sparsely_transitive_balanced ==================
    # =========================================================================
    def test_is_sparsely_transitive_balanced_raises_when_self_loops(self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]])
            network_utils.is_sparsely_transitive_balanced(triad_with_self_loop)

    @parameterized.expand([
        ["120U", np.array(
            [[0, 1, 1],
             [1, 0, 1],
             [-1, -1, 0]]), True],
        ["120D", np.array(
            [[0, 1, -1],
             [1, 0, -1],
             [1, 1, 0]]), True],
        ["0122Z", np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [1, -1, 0]]), True],
        ["030TZ", np.array(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]), True],
        ["003", np.array(
            [[0, -1, -1],
             [-1, 0, -1],
             [-1, -1, 0]]), True],
        ["0032Z", np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [-1, -1, 0]]), True],
        ["030T", np.array(
            [[0, 1, 1],
             [-1, 0, 1],
             [-1, -1, 0]]), True],
        ["021C", np.array(
            [[0, 1, -1],
             [-1, 0, 1],
             [-1, -1, 0]]), False],
        ["030T2negZ", np.array(
            [[0, 1, -1],
             [0, 0, -1],
             [0, 0, 0]]), True],
        ["021UnegZ", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, -1, 0]]), True],
        ["021DZ", np.array(
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]]), True],
        ["210", np.array(
            [[0, 1, -1],
             [1, 0, 1],
             [1, 1, 0]]), False],
        ["210Z", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 1, 0]]), False],
        ["003Z", np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["102Z", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]]), True],
        ["102negZ", np.array(
            [[0, -1, 0],
             [-1, 0, 0],
             [0, 0, 0]]), True],
        ["102posnegZ", np.array(
            [[0, 1, 0],
             [-1, 0, 0],
             [0, 0, 0]]), True],
        ["012Z", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["012", np.array(
            [[0, 1, -1],
             [-1, 0, -1],
             [-1, -1, 0]]), True]]
        )
    def test_is_sparsely_transitive_balanced(
            self, name, triad, expected_balance):
        self.assertEqual(
            network_utils.is_sparsely_transitive_balanced(triad),
            expected_balance)

    # =========================================================================
    # ====================== is_sparsely_cartwright_harary_balanced ===========
    # =========================================================================
    def test_is_sparsely_cartwright_harary_balanced_raises_when_self_loops(
            self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]])
            network_utils.is_sparsely_cartwright_harary_balanced(
                triad_with_self_loop)

    @parameterized.expand([
        ["120U", np.array(
            [[0, 1, 1],
             [1, 0, 1],
             [-1, -1, 0]]), False],
        ["120D", np.array(
            [[0, 1, -1],
             [1, 0, -1],
             [1, 1, 0]]), False],
        ["0122Z", np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [1, -1, 0]]), False],
        ["030TZ", np.array(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]), True],
        ["003", np.array(
            [[0, -1, -1],
             [-1, 0, -1],
             [-1, -1, 0]]), False],
        ["0032Z", np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [-1, -1, 0]]), False],
        ["030T", np.array(
            [[0, 1, 1],
             [-1, 0, 1],
             [-1, -1, 0]]), False],
        ["021C", np.array(
            [[0, 1, -1],
             [-1, 0, 1],
             [-1, -1, 0]]), False],
        ["030T2negZ", np.array(
            [[0, 1, -1],
             [0, 0, -1],
             [0, 0, 0]]), True],
        ["021UnegZ", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, -1, 0]]), True],
        ["021DZ", np.array(
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]]), True],
        ["210", np.array(
            [[0, 1, -1],
             [1, 0, 1],
             [1, 1, 0]]), False],
        ["210Z", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 1, 0]]), False],
        ["003Z", np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["102Z", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]]), True],
        ["102negZ", np.array(
            [[0, -1, 0],
             [-1, 0, 0],
             [0, 0, 0]]), True],
        ["102posnegZ", np.array(
            [[0, 1, 0],
             [-1, 0, 0],
             [0, 0, 0]]), True],
        ["012Z", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["012", np.array(
            [[0, 1, -1],
             [-1, 0, -1],
             [-1, -1, 0]]), False]]
        )
    def test_is_sparsely_cartwright_harary_balanced(
            self, name, triad, expected_balance):
        self.assertEqual(
            network_utils.is_sparsely_cartwright_harary_balanced(triad),
            expected_balance)

    # =========================================================================
    # ====================== is_sparsely_clustering_balanced ==================
    # =========================================================================
    def test_is_sparsely_clustering_balanced_raises_when_self_loops(self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]])
            network_utils.is_sparsely_clustering_balanced(
                triad_with_self_loop)

    @parameterized.expand([
        ["120U", np.array(
            [[0, 1, 1],
             [1, 0, 1],
             [-1, -1, 0]]), False],
        ["120D", np.array(
            [[0, 1, -1],
             [1, 0, -1],
             [1, 1, 0]]), False],
        ["0122Z", np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [1, -1, 0]]), True],
        ["030TZ", np.array(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]), True],
        ["003", np.array(
            [[0, -1, -1],
             [-1, 0, -1],
             [-1, -1, 0]]), True],
        ["0032Z", np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [-1, -1, 0]]), True],
        ["030T", np.array(
            [[0, 1, 1],
             [-1, 0, 1],
             [-1, -1, 0]]), False],
        ["021C", np.array(
            [[0, 1, -1],
             [-1, 0, 1],
             [-1, -1, 0]]), False],
        ["030T2negZ", np.array(
            [[0, 1, -1],
             [0, 0, -1],
             [0, 0, 0]]), True],
        ["021UnegZ", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, -1, 0]]), True],
        ["021DZ", np.array(
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]]), True],
        ["210", np.array(
            [[0, 1, -1],
             [1, 0, 1],
             [1, 1, 0]]), False],
        ["210Z", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 1, 0]]), False],
        ["003Z", np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["102Z", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]]), True],
        ["102negZ", np.array(
            [[0, -1, 0],
             [-1, 0, 0],
             [0, 0, 0]]), True],
        ["102posnegZ", np.array(
            [[0, 1, 0],
             [-1, 0, 0],
             [0, 0, 0]]), True],
        ["012Z", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["012", np.array(
            [[0, 1, -1],
             [-1, 0, -1],
             [-1, -1, 0]]), True]]
        )
    def test_is_sparsely_clustering_balanced(
            self, name, triad, expected_balance):
        self.assertEqual(
            network_utils.is_sparsely_clustering_balanced(triad),
            expected_balance)

    # =========================================================================
    # =================== is_classically_balanced =============================
    # =========================================================================
    def test_is_classically_balanced_raises_when_self_loops(
            self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]])
            network_utils.is_classically_balanced(
                triad_with_self_loop)

    def test_is_classically_balanced_raises_when_negative(
            self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [-1, 0, 0]])
            network_utils.is_classically_balanced(
                triad_with_self_loop)

    @parameterized.expand([
        ["300", np.array(
            [[0, 1, 1],
             [1, 0, 1],
             [1, 1, 0]]), True],
        ["102", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]]), True],
        ["003", np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]), False],
        ["120D", np.array(
            [[0, 0, 1],
             [1, 0, 1],
             [1, 0, 0]]), False],
        ["120U", np.array(
            [[0, 1, 1],
             [0, 0, 0],
             [1, 1, 0]]), False],
        ["030T", np.array(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]), False],
        ["021D", np.array(
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]]), False],
        ["021U", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 1, 0]]), False],
        ["012", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]]), False],
        ["021C", np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [0, 0, 0]]), False],
        ["111U", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 0, 0]]), False],
        ["111D", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 1, 0]]), False],
        ["030C", np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]), False],
        ["201", np.array(
            [[0, 1, 1],
             [1, 0, 0],
             [1, 0, 0]]), False],
        ["120C", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 0, 0]]), False],
        ["210", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 1, 0]]), False]]
            )
    def test_is_classically_balanced(
            self, name, triad, expected_balance):
        self.assertEqual(
            network_utils.is_classically_balanced(triad),
            expected_balance)

    # =========================================================================
    # =============== is_classical_clustering_balanced ========================
    # =========================================================================
    def test_is_classical_clustering_balanced_raises_when_self_loops(
            self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]])
            network_utils.is_classical_clustering_balanced(
                triad_with_self_loop)

    def test_is_classical_clustering_balanced_raises_when_negative(
            self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [-1, 0, 0]])
            network_utils.is_classical_clustering_balanced(
                triad_with_self_loop)

    @parameterized.expand([
        ["300", np.array(
            [[0, 1, 1],
             [1, 0, 1],
             [1, 1, 0]]), True],
        ["102", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]]), True],
        ["003", np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["120D", np.array(
            [[0, 0, 1],
             [1, 0, 1],
             [1, 0, 0]]), False],
        ["120U", np.array(
            [[0, 1, 1],
             [0, 0, 0],
             [1, 1, 0]]), False],
        ["030T", np.array(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]), False],
        ["021D", np.array(
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]]), False],
        ["021U", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 1, 0]]), False],
        ["012", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["021C", np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [0, 0, 0]]), False],
        ["111U", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 0, 0]]), False],
        ["111D", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 1, 0]]), False],
        ["030C", np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]), False],
        ["201", np.array(
            [[0, 1, 1],
             [1, 0, 0],
             [1, 0, 0]]), False],
        ["120C", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 0, 0]]), False],
        ["210", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 1, 0]]), False]]
            )
    def test_is_classical_clustering_balanced(
            self, name, triad, expected_balance):
        self.assertEqual(
            network_utils.is_classical_clustering_balanced(triad),
            expected_balance)

    # =========================================================================
    # ================== is_classical_transitivity_balanced ===================
    # =========================================================================
    def test_is_classical_transitivity_balanced_raises_when_self_loops(
            self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]])
            network_utils.is_classical_transitivity_balanced(
                triad_with_self_loop)

    def test_is_classical_transitivity_balanced_raises_when_negative(
            self):
        with self.assertRaises(ValueError):
            triad_with_self_loop = np.array(
                [[0, 1, 0],
                 [0, 1, 1],
                 [-1, 0, 0]])
            network_utils.is_classical_transitivity_balanced(
                triad_with_self_loop)

    @parameterized.expand([
        ["300", np.array(
            [[0, 1, 1],
             [1, 0, 1],
             [1, 1, 0]]), True],
        ["102", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]]), True],
        ["003", np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["120D", np.array(
            [[0, 0, 1],
             [1, 0, 1],
             [1, 0, 0]]), True],
        ["120U", np.array(
            [[0, 1, 1],
             [0, 0, 0],
             [1, 1, 0]]), True],
        ["030T", np.array(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]), True],
        ["021D", np.array(
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]]), True],
        ["021U", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 1, 0]]), True],
        ["012", np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]]), True],
        ["021C", np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [0, 0, 0]]), False],
        ["111U", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 0, 0]]), False],
        ["111D", np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 1, 0]]), False],
        ["030C", np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]), False],
        ["201", np.array(
            [[0, 1, 1],
             [1, 0, 0],
             [1, 0, 0]]), False],
        ["120C", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 0, 0]]), False],
        ["210", np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [1, 1, 0]]), False]]
            )
    def test_is_classical_transitivity_balanced(
            self, name, triad, expected_balance):
        self.assertEqual(
            network_utils.is_classical_transitivity_balanced(triad),
            expected_balance)

    # # =========================================================================
    # # ===================== is_sparsely_ranked_clustering_balanced ============
    # # =========================================================================
    # def test_is_sparsely_ranked_clustering_balanced_raises_when_self_loops(
    #         self):
    #     with self.assertRaises(ValueError):
    #         triad_with_self_loop = np.array(
    #             [[0, 1, 0],
    #              [0, 1, 1],
    #              [0, 0, 0]])
    #         network_utils.is_sparsely_ranked_clustering_balanced(
    #             triad_with_self_loop)

    # @parameterized.expand([
    #     ["120U", np.array(
    #         [[0, 1, 1],
    #          [1, 0, 1],
    #          [-1, -1, 0]]), True],
    #     ["120D", np.array(
    #         [[0, 1, -1],
    #          [1, 0, -1],
    #          [1, 1, 0]]), True],
    #     ["0122Z", np.array(
    #         [[0, 0, -1],
    #          [-1, 0, 0],
    #          [1, -1, 0]]), True],
    #     ["030TZ", np.array(
    #         [[0, 1, 1],
    #          [0, 0, 1],
    #          [0, 0, 0]]), True],
    #     ["003", np.array(
    #         [[0, -1, -1],
    #          [-1, 0, -1],
    #          [-1, -1, 0]]), True],
    #     ["0032Z", np.array(
    #         [[0, 0, -1],
    #          [-1, 0, 0],
    #          [-1, -1, 0]]), True],
    #     ["030T", np.array(
    #         [[0, 1, 1],
    #          [-1, 0, 1],
    #          [-1, -1, 0]]), False],
    #     ["021C", np.array(
    #         [[0, 1, -1],
    #          [-1, 0, 1],
    #          [-1, -1, 0]]), False],
    #     ["030T2negZ", np.array(
    #         [[0, 1, -1],
    #          [0, 0, -1],
    #          [0, 0, 0]]), True],
    #     ["021UnegZ", np.array(
    #         [[0, 1, 0],
    #          [0, 0, 0],
    #          [0, -1, 0]]), True],
    #     ["021DZ", np.array(
    #         [[0, 0, 0],
    #          [1, 0, 1],
    #          [0, 0, 0]]), True],
    #     ["210", np.array(
    #         [[0, 1, -1],
    #          [1, 0, 1],
    #          [1, 1, 0]]), False],
    #     ["210Z", np.array(
    #         [[0, 1, 0],
    #          [1, 0, 1],
    #          [1, 1, 0]]), False],
    #     ["003Z", np.array(
    #         [[0, 0, 0],
    #          [0, 0, 0],
    #          [0, 0, 0]]), True],
    #     ["102Z", np.array(
    #         [[0, 1, 0],
    #          [1, 0, 0],
    #          [0, 0, 0]]), True],
    #     ["102negZ", np.array(
    #         [[0, -1, 0],
    #          [-1, 0, 0],
    #          [0, 0, 0]]), True],
    #     ["102posnegZ", np.array(
    #         [[0, 1, 0],
    #          [-1, 0, 0],
    #          [0, 0, 0]]), True],
    #     ["012Z", np.array(
    #         [[0, 1, 0],
    #          [0, 0, 0],
    #          [0, 0, 0]]), True],
    #     ["012", np.array(
    #         [[0, 1, -1],
    #          [-1, 0, -1],
    #          [-1, -1, 0]]), True]]
    #     )
    # def test_is_sparsely_ranked_clustering_balanced(
    #         self, name, triad, expected_balance):
    #     self.assertEqual(
    #         network_utils.is_sparsely_ranked_clustering_balanced(triad),
    #         expected_balance)

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
    # ====================== _randomize_network ===============================
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
        computed = network_utils._randomize_network(dg, switching_count_coef=2)
        self.assertEqual(
            sorted(dict(dg.degree()).values()),
            sorted(dict(computed.degree()).values()))
        self.assertEqual(
            sorted(dg.nodes()),
            sorted(computed.nodes()))

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
        computed = network_utils._randomize_network(dg, switching_count_coef=2)
        self.assertEqual(
            sorted(dict(dg.degree()).values()),
            sorted(dict(computed.degree()).values()))
        self.assertEqual(
            sorted(dg.nodes()),
            sorted(computed.nodes()))

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
        computed = network_utils._randomize_network(dg, switching_count_coef=2)
        self.assertEqual(
            sorted(dict(dg.degree()).values()),
            sorted(dict(computed.degree()).values()))
        self.assertEqual(
            sorted(dg.nodes()),
            sorted(computed.nodes()))

    # =========================================================================
    # ================== get_robustness_of_transitions ========================
    # =========================================================================
    def test_get_robustness_of_transitions(self):
        transition_matrices = [
            np.array(
                [[0.9, 0.1, 0],
                 [0.6, 0.2, 0.2],
                 [0.7, 0.1, 0.2]]),
            np.array(
                [[0.1, 0.8, 0.1],
                 [0, 0.9, 0.1],
                 [0.1, 0.1, 0.8]])
        ]
        # Expected dataframe.
        columns = [
            'Transitions',
            'Matrix L2-Norm Dist. from Average',
            'Matrix Pearson r-value',
            'Matrix Pearson p-value',
            'Stationary Dist. L2-Norm Dist. from Average',
            'Stationary Dist. Pearson r-value',
            'Stationary Dist. Pearson p-value']
        expected_df = pd.DataFrame({
            columns[0]: ['Period 1 to Period 2', 'Period 2 to Period 3'],
            columns[1]: [0.8444, 0.8083],
            columns[2]: [0.4256, 0.6522],
            columns[3]: [0.2534, 0.0569],
            columns[4]: [0.5833, 0.4404],
            columns[5]: [0.4637, 0.1319],
            columns[6]: [0.6930, 0.9156],
            },
            columns=columns)
        expected_df = pd.DataFrame(
            expected_df, columns=columns)
        # Computed dataframe.
        computed_df = network_utils.get_robustness_of_transitions(
            transition_matrices, lnorm=2)
        # Comparing computed with expected.
        pd.testing.assert_frame_equal(
            expected_df, computed_df, check_less_precise=2)

    # =========================================================================
    # ================== generate_converted_graphs ============================
    # =========================================================================
    def test_generate_converted_graphs_raises_when_wrong_percentage(self):
        with self.assertRaises(ValueError):
            network_utils.generate_converted_graphs(
                dgraph=nx.DiGraph(),
                percentage=-1)

    def test_generate_converted_graphs_when_it_adds_edges(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        dg.add_edge(1, 2, weight=1)
        dg.add_edge(1, 3, weight=2)
        dg.add_edge(2, 3, weight=5)
        dg.add_edge(3, 1, weight=1)
        percentage = 25
        computed_graphs = network_utils.generate_converted_graphs(
            dgraph=dg,
            convert_from=0,
            convert_to=1,
            percentage=percentage,
            how_many_to_generate=5)
        for computed in computed_graphs:
            # It should contain all nodes.
            self.assertEqual(dg.nodes(), computed.nodes())
            # It should contain all dg's edges.
            self.assertEqual(len(nx.difference(dg, computed).edges()), 0)
            # It should contain percentage% more edges.
            remaining_edges_count = 4 * 3 - 4
            self.assertEqual(
                len(nx.difference(computed, dg).edges()),
                int(percentage*remaining_edges_count/100))

    def test_generate_converted_graphs_when_all_edges_exist(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        dg.add_edge(1, 2, weight=2)
        dg.add_edge(1, 3, weight=-5)
        dg.add_edge(2, 3, weight=-2)
        dg.add_edge(3, 1, weight=2)
        dg.add_edge(4, 1, weight=2)
        dg.add_edge(4, 3, weight=2)
        percentage = 25
        computed_graphs = network_utils.generate_converted_graphs(
            dgraph=dg,
            convert_from=2,
            convert_to=3,
            percentage=percentage,
            how_many_to_generate=2)
        for computed in computed_graphs:
            converted_cnt = 0
            # It should contain all nodes.
            self.assertEqual(dg.nodes(), computed.nodes())
            # It should contain all dg's edges.
            self.assertEqual(dg.edges(), computed.edges())
            # Checking every edge weight.
            for edge in dg.edges():
                w1 = dg.get_edge_data(edge[0], edge[1])['weight']
                w2 = computed.get_edge_data(edge[0], edge[1])['weight']
                if w1 == w2:
                    continue
                if w1 != w2 and w1 == 2 and w2 == 3 and converted_cnt == 0:
                    converted_cnt += 1
                else:
                    self.assertTrue(
                        False, 'Found more converted edges than expeced.')

    def test_generate_converted_graphs(self):
        dg = nx.DiGraph()
        dg.add_nodes_from([1, 2, 3, 4])
        percentage = 10
        computed_graphs = network_utils.generate_converted_graphs(
            dgraph=dg,
            convert_from=0,
            convert_to=1,
            percentage=percentage,
            how_many_to_generate=2)
        for computed in computed_graphs:
            # It should contain all nodes.
            self.assertEqual(dg.nodes(), computed.nodes())
            # It should contain percentage extra edges.
            self.assertEqual(
                len(computed.edges()), int(4 * 3 * percentage / 100))

    def test_generate_converted_graphs_for_large_networks(self):
        n = 100
        m = 300
        dgraph = nx.gnm_random_graph(n=n, m=m, directed=True)
        percentage = 5
        computed_graphs = network_utils.generate_converted_graphs(
            dgraph=dgraph,
            convert_from=0,
            convert_to=1,
            percentage=percentage,
            how_many_to_generate=6)
        for computed in computed_graphs:
            # It should contain all nodes.
            self.assertEqual(dgraph.nodes(), computed.nodes())
            # It should contain percentage extra edges.
            self.assertEqual(
                len(computed.edges()), m + int(
                    (n * (n-1) - m) * percentage / 100))


if __name__ == '__main__':
    unittest.main()
