import numpy as np
from copy import deepcopy
from genepro.node import Node
from genepro.node_impl import Composition, Identity, Xor

class Encoder:
    """Encode and decode in and from real-valued representation of syntax trees.
    
    Parameters
    -----
    unaryNodes :list: unary atomic functions available,
    binaryNodes :list: binary atomic functions available,
    leafNodes :list: leaf nodes available """

    def __init__(self, unaryNodes : list, binaryNodes : list, leafNodes : list, 
            train_x : np.ndarray, train_y : np.ndarray, max_depth : int = 4) -> None:
        # --- save original catogorization ---
        self.leafNodes = leafNodes
        self.binaryNodes, self.unaryNodes = binaryNodes, unaryNodes
        self.max_depth = max_depth
        self.train_x, self.train_y = train_x, train_y
        # --- merge to composition node representation ---
        unaryNodes = unaryNodes + [Identity()]
        binaryNodes = binaryNodes + [Xor(0), Xor(1)] + leafNodes
        nUn, nBin = len(unaryNodes), len(binaryNodes)
        self.internalNodes = np.array([[Composition(unaryNodes[j],
                                binaryNodes[i]) for j in range(nUn)]
                                                for i in range(nBin)])
    
    def _assign_domains(self, tree : Node) -> None:
        """Recursively assign domains to each node (interval arithmetics).
        
        Parameters
        -----
        tree :Node: root of subtree,
        x :np.ndarray: train data
        
        Returns
        -----
        None"""

        #self.x = x
        tree.domains = []
        for child in tree._children:
            self._assign_domains(child)
        if tree.binary.arity == 0:
            # --- for leaf nodes the domain has to be added twice since they are
            #     treated as binary nodes, the data only used in terms of its shape ---
            tree.domains.append(self.x[:,0])
            tree.domains.append(self.x[:,0])
        else:
            for child in tree._children:
                child_image = child.eval_indiv(child.domains[0],
                                               child.domains[1])
                tree.domains.append(child_image)
    
    def _softmax(self, x : np.ndarray) -> np.ndarray:
        """Apply softmax numerically a bit more stabilized as
        large values are subtracted before each exponential call.
        
        Parameters
        -----
        x :np.ndarray: data
        
        Returns
        -----
        softmaxed array of same shape :np.ndarray:"""

        return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()
        
    def _compare(self, ax : np.ndarray, bx : np.ndarray,
                                    epsilon : float = 2.) -> float:
        """Metric to determine similarity between atomic functions. The
        epsilon value has a large impact on what makes an adequate step size
        later in traversing the encoded space.
        
        Parameters
        -----
        ax :np.ndarray: left child image, bx :np.ndarray: right child image
        
        Returns
        -----
        similarity :float:"""
        epsilon = 1/max(np.abs(self.train_y))
        return np.exp(-np.mean(np.power(ax-bx, 2)) * epsilon)
        
    def _forward_mapping(self, node : Node) -> np.ndarray:
        """Generate embedding for a single node.

        Parameters
        -----
        node :Node: current node to be decoded
        
        Returns
        -----
        Return :np.ndarray: encoding of single node"""

        nBin, nUn = self.internalNodes.shape
        encoded = np.zeros((nBin, nUn))
        current = node.eval_indiv(node.domains[0], node.domains[1])
        for i in range(nBin):
            for j in range(nUn):
                compareNode = self.internalNodes[i, j]
                try:
                    compare = compareNode.eval_indiv(node.domains[0],
                                                     node.domains[1])
                except RuntimeWarning:
                     # --- Overflow allowed, clipped ---
                    pass
                encoded[i, j] = self._compare(
                    np.clip(current, -1e32, 1e32),
                    np.clip(compare, -1e32, 1e32),
                    )
        
        return self._softmax(encoded)
    
    def encode(self, tree : Node, x : np.ndarray, has_domains : bool = False)\
                                                                -> np.ndarray:                  
        """Generate encoding of tree.

        Parameters
        -----
        tree :Node: root of tree to be encoded,
        x :np.ndarray: x-data 
        
        Returns
        -----
        :np.ndarray: tree representation in encoded space"""
        
        if not has_domains:
            self._assign_domains(tree)
        pref_vector = tree.get_subtree()
        code = []
        for node in pref_vector:
            code.append(self._forward_mapping(node))
        return np.array(code)
    
    def _backward_mapping(self, vector : np.ndarray, forceLeaf = False) -> Node:
        """Reverse mapping from embedding to nodes. Called for individual
        node, trees are reconnected afterwards.

        Parameters
        -----
        vector :np.ndarray: encoded node, forceLeaf :bool: yes for terminals
        
        Returns
        -----
        :Node: node object of embedded vector"""

        if forceLeaf:
            ids = list(np.unravel_index(
                vector[-len(self.leafNodes):].argmax(),
                vector[-len(self.leafNodes):].shape)
                )
            ids[0] += self.internalNodes.shape[0] - len(self.leafNodes)
        else:
            ids = np.unravel_index(vector.argmax(), vector.shape)
        newNode = deepcopy(self.internalNodes[int(ids[0]), int(ids[1])])
        return newNode
    
    def _connect_tree_recursive(self, idx : int, sequence : list, arityCount : list):
        """Reconnect nodes in a vector of nodes that represents a tree.

        Parameters
        -----
        idx : used for recursive call,
        sequence :list: list of nodes to connect,
        arityCount :list: list of the supposed arity for each node

        Returns
        -----
        Nothing of use outside of recursive call
        """
        selfId, runner = idx, idx+1
        while arityCount[selfId] > 0:
            child_id, runner = self._connect_tree_recursive(
                                        runner, sequence, arityCount)
            sequence[selfId]._children.append(sequence[child_id])
            arityCount[selfId] -= 1
        return selfId, runner
    
    def decode(self, code : np.ndarray) -> Node:                                                       
        """Generate node representation of encoded tree.

        Parameters
        -----
        code :np.ndarray: the encoded tree

        Returns
        -----
        :Node: decoded tree"""

        sequence = []
        arCount = [2,2,0,0,2,0,0] # -> hardcoded for depth 2 atm, will be changed
        for i, c in enumerate(code):
            sequence.append(self._backward_mapping(c, forceLeaf = arCount[i] == 0))
        
        self._connect_tree_recursive(0, sequence, arCount)
        return sequence[0]

class SlimEncoder:
    """Encode and decode in and from real-valued representation of syntax trees.
    
    Parameters
    -----
    unaryNodes :list: unary atomic functions available,
    binaryNodes :list: binary atomic functions available,
    leafNodes :list: leaf nodes available """

    def __init__(self, unaryNodes : list, binaryNodes : list,
            leafNodes : list, train_x : np.ndarray, train_y : np.ndarray) -> None:
        # --- save original catogorization ---
        self.leafNodes = leafNodes
        self.binaryNodes, self.unaryNodes = binaryNodes, unaryNodes
        self.train_x, self.train_y = train_x, train_y
        # --- merge to composition node representation ---
        unaryNodes = [Composition(unaryNodes[i], [Xor(0), Xor(1)][j])
                        for i in range(len(unaryNodes)) for j in range(2)]
        binaryNodes = [Composition(Identity(), bin) for bin in binaryNodes]
        leafNodes = [Composition(Identity(), lf) for lf in leafNodes]
        self.internalNodes = unaryNodes + binaryNodes + leafNodes
    
    
    def _softmax(self, x : np.ndarray) -> np.ndarray:
        """Apply softmax numerically a bit more stabilized as
        large values are subtracted before each exponential call.
        
        Parameters
        -----
        x :np.ndarray: data
        
        Returns
        -----
        softmaxed array of same shape :np.ndarray:"""

        return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()
        
    def _compare(self, ax : np.ndarray, bx : np.ndarray,
                                    epsilon : float = 1) -> float:
        """Metric to determine similarity between atomic functions. The
        epsilon value has a large impact on what makes an adequate step size
        later in traversing the encoded space.
        
        Parameters
        -----
        ax :np.ndarray: left child image, bx :np.ndarray: right child image
        
        Returns
        -----
        similarity :float:"""
        epsilon = epsilon/max(np.abs(self.train_y))
        return np.exp(-np.mean(np.power(ax-bx, 2)) * epsilon)
    
    def encode_leaf(self, tree : Node, x : np.ndarray) -> np.ndarray:
        encoding = np.zeros(len(self.leafNodes))
        comparedNodesCache = []
        for i in range(len(self.leafNodes)):
            compareEval = self.leafNodes[i].eval_indiv(x[0], x[1])
            encoding[i] = self._compare(tree.eval, compareEval)
            comparedNodesCache.append(compareEval)

        return encoding, comparedNodesCache
    
    def encode(self, tree : Node, x : np.ndarray) -> np.ndarray:      
                                                                         
        """Generate encoding of node.

        Parameters
        -----
        tree :Node: root of tree to be encoded,
        x :np.ndarray: x-data 
        
        Returns
        -----
        :np.ndarray: tree representation in encoded space"""

        encoding = np.zeros(len(self.internalNodes))
        comparedNodesCache = []
        for i in range(len(self.internalNodes)):
            compareEval = self.internalNodes[i].eval_indiv(x[0], x[1])
            encoding[i] = self._compare(tree.eval, compareEval)
            comparedNodesCache.append(compareEval)

        return encoding, comparedNodesCache
    
    
    def decode(self, new_index : int):
        return deepcopy(self.internalNodes[new_index])

    def decode_leaf(self, new_index : int):
        return deepcopy(self.leafNodes[new_index])