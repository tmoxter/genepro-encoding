import numpy as np
from genepro.node import Node
from genepro.node_impl import Composition, Identity, Xor
from numpy.random import choice as randc
from copy import deepcopy
    
def sample_tree(unary_nodes : list, binary_nodes : list, leaf_nodes : list,
                        max_depth : int = 3, curr_depth : int = 0) -> Node:
    """
    Generate randomly sampled full tree of given depth based on the new 
    composition structure.
    
    Parameters
    -----
    unaryNodes :list: unary atomic functions available,
    binaryNodes :list: binary atomic functions available,
    leafNodes :list: leaf nodes available, max_depth :int: final depth of
    tree, curr_depth :int: used for recursive call only
    
    Returns
    -----
    Returns :Node: root node of newly sampled tree"""

    if np.random.randn() > 0 or curr_depth == max_depth:
        unaryNode = Identity()
    else:
        unaryNode = deepcopy(randc(unary_nodes))
    if curr_depth == max_depth:
        binaryNode = deepcopy(randc(leaf_nodes))
    elif np.random.random() < 0.25 and not isinstance(unaryNode, Identity):
        binaryNode = Xor(0)
    elif np.random.random() < 0.5 and not isinstance(unaryNode, Identity):
        binaryNode = Xor(1)
    else:
        binaryNode = deepcopy(randc(binary_nodes))

    n = Composition(unaryNode, binaryNode)

    if curr_depth != max_depth:
        for _ in range(n.arity):
            c = sample_tree(unary_nodes, binary_nodes, leaf_nodes, 
                                            max_depth, curr_depth+1)
            n.insert_child(c)
    return n
