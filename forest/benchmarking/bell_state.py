import networkx as nx
from forest.benchmarking.entangled_states import create_ghz_program
from forest.benchmarking.entangled_states import ghz_state_statistics

import warnings
import functools

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}. \r\n \r\n Use the GHZ functions in "
                      "forest.benchmarking.entangled_states instead.\r\n".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

@deprecated
def create_bell_program(tree: nx.DiGraph):
    """
    Create a Bell/GHZ state with CNOTs described by tree.

    :param tree: A tree that describes the CNOTs to perform to create a bell/GHZ state.
    :return: the program
    """
    return create_ghz_program(tree)

@deprecated
def bell_state_statistics(bitstrings):
    """
    Compute statistics bitstrings sampled from a Bell/GHZ state

    :param bitstrings: An array of bitstrings
    :return: A dictionary where bell = number of bitstrings consistent with a bell/GHZ state;
        total = total number of bitstrings.
    """
    return ghz_state_statistics(bitstrings)
