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
import networkx as nx


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

    Output:
        Boolean whether g1 equals g2 or not.

    Raises:
        None.
    """
    return g1.nodes() == g2.nodes() and g1.edges() == g2.edges()