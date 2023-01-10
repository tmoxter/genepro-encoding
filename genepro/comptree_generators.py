import numpy as np
from genepro.node import Node
from genepro.node_impl import Composition, Identity, Xor
from numpy.random import choice as randc
from copy import deepcopy

def sample_tree_vectorized(unaryNodes : list, binaryNodes : list, leafNodes : list,
                        depth: int, xtrain : np.ndarray):
    """Generate random new full trew in vectorized format. Nodes are already compositin objects,
    And are already evaluated on the training data.

    Parameters
    -----
    unaryNodes :list: unary atomic functions available,
    binaryNodes :list: binary atomic functions available,
    leafNodes :list: leaf nodes available,
    depth :int: depth the binary tree,
    xtrain :np.ndarray: training data
    
    Returns
    -----
    Returns :list: newly sampled tree"""

    unaryNodes = [Composition(unaryNodes[i], [Xor(0), Xor(1)][j])
                        for i in range(len(unaryNodes)) for j in range(2)]
    binaryNodes = [Composition(Identity(), bin) for bin in binaryNodes]
    leafNodes = [Composition(Identity(), lf) for lf in leafNodes]
    depth = depth+1
    tree = [None]*(2**depth)
    # --- fill leaf nodes ---
    for i in range(2**(depth-1), 2**depth):
        choice = deepcopy(randc(leafNodes))
        choice.eval = choice.eval_indiv(xtrain, None)
        tree[i] = choice
    # --- fill rest of the tree recursively ---
    for i in range(2**(depth-1)-1, 0, -1):
        choice = deepcopy(randc(binaryNodes*2 + unaryNodes))
        choice.eval = choice.eval_indiv(tree[2*i].eval, tree[2*i+1].eval)
        tree[i] = choice
    tree[0] = -1e2
    return tree
    
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

def sample_tree_slim(unaryNodes : list, binaryNodes : list, leafNodes : list,
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

    if curr_depth == max_depth:
        node = deepcopy(randc(leafNodes))
    else:
        node = deepcopy(randc(unaryNodes + binaryNodes))

    if curr_depth != max_depth:
        for _ in range(n.arity):
            c = sample_tree_slim(unaryNodes, binaryNodes, leafNodes, 
                                            max_depth, curr_depth+1)
            node.insert_child(c)
    return node