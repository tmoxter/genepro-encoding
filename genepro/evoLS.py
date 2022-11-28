import numpy as np
import time
from copy import deepcopy
from joblib.parallel import Parallel, delayed
from typing import Tuple, Callable

from genepro.selection import tournament_selection, tournament_selection_vectorized
from genepro.comptree_generators import sample_tree_vectorized, sample_tree
from genepro.local_search import RandomWalkTabu as RandomWalk
from genepro.encoding import SlimEncoder



class EvolutionLocalSearch(RandomWalk):
    """
    Simple evolutionary algorithm similar to Genepro that uses the encoding to make small local variations
    in the form of brief random walks in the endoced space. The new variations yield the offspring.
    Selection step is tournament-selection from https://github.com/marcovirgolin/genepro/blob/main/genepro/selection.py.

    Parameters
    ------
    unaryNodes :list: unary atomic functions available,
    binaryNodes :list: binary atomic functions available,
    leafNodes :list: leaf nodes available,
    train_x :np.ndarray: training data (x),
    train_y :np.ndarray: training data (y),
    step_size :float: size (R2) of the step in the encoding,
    n_attempts :int: number of steps allowed during local search without improvements,
    fitness :Callable: call signature has to be f(tree : Node),
    pop_size :int: size of the population,
    max_evals :int: maximum allowed number of evaluations,
    max_gens :int: maximum allowed number of generations,
    max_time :float: maximum time to run in seconds,
    n_jobs :int: number of jobs to work in prallel (-1) uses all available cpus,
    verbose :bool: per generation statements,
    log :object: torchvision SummaryWriter or None

    Attributes
    -------
    best_of_gen :list: list of best solutions of each generation, pos [-1] for best overall

    """
    
    def __init__(self, unaryNodes: list, binaryNodes: list, leafNodes: list,
        train_x: np.ndarray, train_y: np.ndarray, step_size : float, n_attempts : int = 10,
        restarts : int = 1,
        fitness : Callable = None, pop_size : int = 100,
        max_evals : int=None, max_gens : int=100, max_time : int=None,
        n_jobs : int = -1, verbose : bool=False, log : object = None) -> None:

        super().__init__(unaryNodes, binaryNodes, leafNodes, train_x, train_y)

        self.step_size, self.restarts = step_size, restarts
        self.max_evals, self.max_gens = max_evals, max_gens
        self._fitness = fitness
        self.pop_size, self.pop_size_init = pop_size, pop_size
        self.max_time, self.n_jobs = max_time, n_jobs
        self.verbose, self.log = verbose, log
        self.n_attempts = n_attempts
        self.best_of_gens = list()
        self.time_at_gen, self.eval_at_gen = [], []
        
    def evolve(self) -> None:

        self.start_time = time.time()
        self.num_gens = 0
        self.num_evals = 0

        self._initialize_population()
        
        # --- generational loop ---
        while not self._must_terminate():
            # --- --- perform one generation --- ---
            self._perform_generation()
            # --- --- logging & printing --- ---
            if self.log:
                self.log.add_scalar("Best v. nGen", 
                    self.best_of_gens[-1].fitness, self.num_gens)
                self.log.add_scalar("Pop Fitness v. nGen",
                    self.pop_fitness, self.num_gens)
            
            if self.verbose:
                print("Gen: {}, best of gen fitness: {:.5f}, gen fitness: {:.3f}"\
                    .format(self.num_gens, self.best_of_gens[-1].fitness,
                    np.sum([indiv.fitness for indiv in self.population]))
                    )
    
    def _variation(self, tree):
        var_f, var_t, _ = self.search(self.step_size, n_attempts=self.n_attempts,
                                                fitness=self._fitness, tree=tree)
        var_t.fitness = var_f
        return var_t
    
    def _initialize_population(self):
        self.population = Parallel(self.n_jobs, backend='threading')(
            delayed(sample_tree)
            (self.unaryNodes, self.binaryNodes, self.leafNodes, max_depth=2)
            for _ in range(self.pop_size)
        )
        
        for tree in self.population:
            tree.fitness = self._fitness(tree)
            self._valid_change(self.encode(tree, self.train_x))

    def _perform_generation(self) -> None:
        """Evolve one generation."""

        offspring = Parallel(self.n_jobs, backend='threading')(
            delayed(self._variation) (self.population[i])
            for i in range(self.pop_size)
            )
        self.population = tournament_selection(self.population 
                                                 + offspring, self.pop_size, 2)

        # --- update info ---
        self.num_gens += 1
        self.num_evals += 2*self.pop_size
        self.pop_fitness = np.sum([indiv.fitness for indiv in self.population])
        best = self.population[np.argmax([t.fitness for t in self.population])]
        self.best_of_gens.append(deepcopy(best))
        self.time_at_gen.append(time.time() - self.start_time)
        self.eval_at_gen.append(self.num_evals)
    
    def _must_terminate(self) -> bool:
        """Check if termination must occur."""

        self.elapsed_time = time.time() - self.start_time
        if self.max_time and self.elapsed_time >= self.max_time:
            return True
        elif self.max_evals and self.num_evals >= self.max_evals:
            return True
        elif self.max_gens and self.num_gens >= self.max_gens:
            return True
        return False

class EvolutionLocalSearchSlim(SlimEncoder):
    """
    Simple evolutionary algorithm similar to Genepro that uses the encoding to make small local variations
    in the form of brief random walks in the endoced space. The new variations yield the offspring.
    Selection step is tournament-selection from https://github.com/marcovirgolin/genepro/blob/main/genepro/selection.py.

    Parameters
    ------
    unaryNodes :list: unary atomic functions available,
    binaryNodes :list: binary atomic functions available,
    leafNodes :list: leaf nodes available,
    train_x :np.ndarray: training data (x),
    train_y :np.ndarray: training data (y),
    step_size :float: size (R2) of the step in the encoding,
    n_attempts :int: number of steps allowed during local search without improvements,
    fitness :Callable: call signature has to be f(tree : Node),
    pop_size :int: size of the population,
    max_evals :int: maximum allowed number of evaluations,
    max_gens :int: maximum allowed number of generations,
    max_time :float: maximum time to run in seconds,
    n_jobs :int: number of jobs to work in prallel (-1) uses all available cpus,
    verbose :bool: per generation statements,
    log :object: torchvision SummaryWriter or None

    Attributes
    -------
    best_of_gen :list: list of best solutions of each generation, pos [-1] for best overall

    """
    
    def __init__(self, unaryNodes: list, binaryNodes: list, leafNodes: list,
        train_x: np.ndarray, train_y: np.ndarray, var_size_neighb : int = 5,
        var_size_nodes : int = 1,
        fitness : Callable = None, pop_size : int = 100, max_depth = 2,
        max_evals : int=None, max_gens : int=100, max_time : int=None,
        n_jobs : int = 8, verbose : bool=False, log : object = None) -> None:

        super().__init__(unaryNodes, binaryNodes, leafNodes, train_x, train_y)

        self.max_evals, self.max_gens = max_evals, max_gens
        self.max_depth = max_depth
        self._fitness = fitness
        self.pop_size, self.pop_size_init = pop_size, pop_size
        self.max_time, self.n_jobs = max_time, n_jobs
        self.verbose, self.log = verbose, log
        self.var_size_neighb = var_size_neighb
        self.var_size_nodes = var_size_nodes
        self.best_of_gens = list()
        self.time_at_gen, self.eval_at_gen = [], []
        if not self.log:
            self.save_data = {"best":{"f":[],"t":[],"n_eval":[]},
                                "population":{"f":[],"t":[],"n_eval":[]}}
        
    def evolve(self) -> None:

        self.start_time = time.time()
        self.num_gens = 0
        self.num_evals = 0

        self._initialize_population()
        
        # --- generational loop ---
        while not self._must_terminate():
            # --- --- perform one generation --- ---
            self._perform_generation()
            # --- --- logging & printing --- ---
            if self.log:
                self.log.add_scalar("Best v. nGen", 
                    self.best_of_gens[-1][0], self.num_gens)
                self.log.add_scalar("Pop Fitness v. nGen",
                    self.pop_fitness, self.num_gens)
            else:
                self.save_data["best"]["f"].append(self.best_of_gens[-1][0])
                self.save_data["best"]["t"].append(time.time() - self.start_time)
                self.save_data["best"]["n_eval"].append(self.num_evals)
                self.save_data["population"]["f"].append(self.pop_fitness)
                self.save_data["population"]["t"].append(time.time() - self.start_time)
                self.save_data["population"]["n_eval"].append(self.num_evals)
            if self.verbose:
                print("Gen: {}, best of gen fitness: {:.5f}, gen fitness: {:.3f}"\
                    .format(self.num_gens, self.best_of_gens[-1][0],
                    np.sum([indiv[0] for indiv in self.population]))
                    )
    
    def _variation(self, tree):
        tree = deepcopy(tree)
        
        cIdxs = np.random.choice(range(1, len(tree)),
                                size =self.var_size_nodes)
        # --- change the states of the selected notes ---
        for cidx in cIdxs:
            if cidx > 2**(self.max_depth-1)+1:
                x = (tree[cidx].eval, tree[cidx].eval)
                encoded, cache = self.encode_leaf(tree[cidx], x)
                self.num_evals += len(self.leafNodes) / (2**(self.max_depth+1)-1)
                encoded[np.argmax(encoded)] = -1e10
                stateChange = np.random.choice(range(len(self.leafNodes)),
                                p=self._softmax(encoded))
                tree[cidx] = self.decode_leaf(stateChange)
                tree[cidx].eval = cache[stateChange]
            else:
                x = (tree[2*cidx].eval, tree[2*cidx+1].eval)
                encoded, cache = self.encode(tree[cidx], x)
                self.num_evals += len(self.internalNodes) / (2**(self.max_depth+1)-1)
                nearest = np.argpartition(encoded,
                            -self.var_size_neighb-1)[-self.var_size_neighb-1:]
                nearest = list(set(nearest) - set([np.argmax(encoded)]))
                stateChange = np.random.choice(nearest,
                                            p=self._softmax(encoded[nearest]))
                tree[cidx] = self.decode(stateChange)
                tree[cidx].eval = cache[stateChange]
            
            parentId = (cidx)//2
            while parentId > 0:
                    tree[parentId].eval = tree[parentId].eval_indiv(tree[(parentId)*2].eval,
                                                                    tree[(parentId)*2+1].eval)
                    self.num_evals += 1 / (2**(self.max_depth+1)-1)
                    if parentId == 1:
                        break
                    parentId = (parentId)//2
        
        tree[0] = self._fitness(tree[1].eval)
        return tree
    
    def _initialize_population(self):
        self.population = Parallel(self.n_jobs, backend='threading')(
            delayed(sample_tree_vectorized)
            (self.unaryNodes, self.binaryNodes, self.leafNodes, depth=self.max_depth, xtrain=self.train_x)
            for _ in range(self.pop_size)
        )
        
        for tree in self.population:
            tree[0] = self._fitness(tree[1].eval)
            #self._valid_change(self.encode(tree, self.train_x))

    def _perform_generation(self) -> None:
        """Evolve one generation."""

        offspring = Parallel(self.n_jobs, backend='threading')(
            delayed(self._variation) (self.population[i])
            for i in range(self.pop_size)
            )
        self.population = tournament_selection_vectorized(self.population 
                                                 + offspring, self.pop_size, 8)

        # --- update info ---
        self.num_gens += 1
        self.num_evals = int(self.num_evals)
        self.pop_fitness = np.sum([indiv[0] for indiv in self.population])
        best = self.population[np.argmax([t[0] for t in self.population])]
        self.best_of_gens.append(deepcopy(best))
        self.time_at_gen.append(time.time() - self.start_time)
        self.eval_at_gen.append(self.num_evals)
    
    def _must_terminate(self) -> bool:
        """Check if termination must occur."""

        self.elapsed_time = time.time() - self.start_time
        if self.max_time and self.elapsed_time >= self.max_time:
            return True
        elif self.max_evals and self.num_evals >= self.max_evals:
            return True
        elif self.max_gens and self.num_gens >= self.max_gens:
            return True
        return False