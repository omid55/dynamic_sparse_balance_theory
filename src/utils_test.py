# Omid55
# Test module for network_utils.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from io import StringIO
import pandas as pd
import numpy as np
import sys
import utils
import networkx as nx
import unittest


class MyTestClass(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.simple_dataframe = pd.DataFrame(
            {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]})

    @classmethod
    def tearDown(cls):
        del cls.simple_dataframe

    # =========================================================================
    # ==================== print_dict_pretty ==================================
    # =========================================================================
    def test_print_dict_pretty(self):
        capturedOutput = StringIO()
        sys.stdout = capturedOutput
        sample_dict = {'key1': 'value1', 'key2': 'value2'}
        utils.print_dict_pretty(sample_dict)
        sys.stdout = sys.__stdout__
        computed = capturedOutput.getvalue()
        expected = 'key1: value1\nkey2: value2\n'
        self.assertEqual(expected, computed)

    # =========================================================================
    # ==================== check_required_columns =============================
    # =========================================================================
    def test_if_check_required_columns_not_raise(self):
        utils.check_required_columns(self.simple_dataframe, ['col1', 'col3'])

    def test_if_check_required_columns_raises_when_missing(self):
        with self.assertRaises(ValueError):
            utils.check_required_columns(
                self.simple_dataframe, ['col1', 'col4'])

    # =========================================================================
    # ==================== graph_equals =======================================
    # =========================================================================
    def test_graph_equals_when_same(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(2, 3)
        g2.add_edge(1, 2)
        self.assertTrue(utils.graph_equals(g1, g2))

    def test_graph_equals_when_not_same(self):
        g1 = nx.DiGraph()
        g1.add_nodes_from([1, 2, 3])
        g1.add_edge(1, 2)
        g1.add_edge(2, 3)
        g2 = nx.DiGraph()
        g2.add_nodes_from([1, 2, 3])
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        g2.add_edge(1, 3)
        self.assertFalse(utils.graph_equals(g1, g2))

    # =========================================================================
    # ==================== sub_adjacency_matrix ===============================
    # =========================================================================
    def test_sub_adjacency_matrix(self):
        adj_matrix = np.arange(16).reshape(4, 4)
        expected = np.array(
            [[0, 1, 3],
             [4, 5, 7],
             [12, 13, 15]])
        computed = utils.sub_adjacency_matrix(adj_matrix, [0, 1, 3])
        np.testing.assert_array_equal(expected, computed)

    # =========================================================================
    # ==================== swap_nodes_in_matrix ===============================
    # =========================================================================
    def test_swap_nodes_in_matrix(self):
        matrix = np.array(
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]])
        node1 = 0
        node2 = 2
        expected = np.array(
            [[8, 7, 6],
             [5, 4, 3],
             [2, 1, 0]])
        computed = utils.swap_nodes_in_matrix(matrix, node1, node2)
        np.testing.assert_array_equal(expected, computed)


if __name__ == '__main__':
    unittest.main()
