import numpy as np
import time
from copy import deepcopy
from joblib.parallel import Parallel, delayed
from typing import Tuple, Callable
import pandas as pd
#import cProfile as profiler
# --> Remove parallization for insightful profiling

from genepro.selection import tournament_selection_vectorized as tournament_selection
from genepro.variation import node_variations
from genepro.comptree_generators import sample_tree_vectorized as sample_tree
from genepro.encoding import SlimEncoder

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
        train_x: np.ndarray, train_y: np.ndarray,
        variation : dict, selection : dict,
        fitness : Callable = None, pop_size : int = 100, max_depth = 2,
        max_evals : int=None, max_gens : int=100, max_time : int=None,
        n_jobs : int = 8, verbose : bool=False, log : object = None,
        ) -> None:

        super().__init__(unaryNodes, binaryNodes, leafNodes, train_x, train_y)

        self.max_evals, self.max_gens = max_evals, max_gens
        self.max_depth = max_depth
        self._fitness = fitness
        self.variation_config = variation
        self.selection_config = selection
        self.pop_size, self.pop_size_init = pop_size, pop_size
        self.max_time, self.n_jobs = max_time, n_jobs
        self.verbose, self.log = verbose, log
        self.best_of_gens = list()
        self.time_at_gen, self.eval_at_gen = [], []
        self.save_data = []
        
    def evolve(self) -> None:
        """Initialize new generation and evolve until termination criteria are met.
        
        Parameters
        ------
        
        Returns
        ------
        None"""
        self.start_time = time.time()
        self.num_gens = 0
        self.num_evals = 0

        self._initialize_population()
        
        # --- generational loop ---
        while not self._must_terminate():
            # --- --- perform one generation --- ---
            self._perform_generation()
            # --- --- logging & printing --- ---
            self.save_data.append(
                {"n_gen":self.num_gens,
                "f":self.best_of_gens[-1][0],
                "f_fitness":self.pop_fitness,
                "t":time.time() - self.start_time,
                "n_eval": self.num_evals, "tree":self.best_of_gens[-1]}
            )
            if self.log:
                self.log.add_scalar("Best v. nGen", 
                    self.best_of_gens[-1][0], self.num_gens)
                self.log.add_scalar("Pop Fitness v. nGen",
                    self.pop_fitness, self.num_gens)
            if self.verbose:
                print("Gen: {}, best of gen fitness: {:.5f}, gen fitness: {:.3f}"\
                    .format(self.num_gens, self.best_of_gens[-1][0],
                    np.sum([indiv[0] for indiv in self.population]))
                )
    
        self.results = pd.DataFrame.from_dict(self.save_data)
    
    def _initialize_population(self):
        """Initialize the start population and evaluate its fitness"""
        self.population = Parallel(self.n_jobs, backend='threading')(
            delayed(sample_tree)
            (
            self.unaryNodes, self.binaryNodes, self.leafNodes,
            depth=self.max_depth, xtrain=self.train_x
            )
            for _ in range(self.pop_size)
        )
        
        for tree in self.population:
            tree[0] = self._fitness(tree[1].eval)

    def _perform_generation(self) -> None:
        """Evolve one generation."""

        # --- general form of local variation ---
        if self.variation_config["method"] == "nodewise_variation":
            offspring = Parallel(self.n_jobs, backend='threading')(
                delayed(node_variations)
                (self,
                self.population[i],
                self.variation_config["paras"]["n_steps"],
                self.variation_config["paras"]["large_move_rate"],
                self.variation_config["paras"]["selective"])
                for i in range(self.pop_size
                )
            )

        # --- combination of multiple local and and large random moves ---
        elif self.variation_config["method"] == "consequtive_variation":
            # --- --- perform several small selective (exploitative) steps --- ---
            self.population = Parallel(self.n_jobs, backend='threading')(
                delayed(node_variations)
                (self,
                self.population[i],
                self.variation_config["paras"]["n_steps"],
                0, True
                )
                for i in range(self.pop_size)
            )
            # --- --- perform one large (explotative) step --- ---
            offspring = Parallel(self.n_jobs, backend='threading')(
                delayed(node_variations)
                (self,
                self.population[i],
                1, 1, False
                )
                for i in range(self.pop_size)
            )

        # --- selection ---
        if self.selection_config["method"] == "tournament_selection":
            self.population = tournament_selection(
                self.population + offspring,
                self.pop_size, self.selection_config["paras"]["tournament_size"]
                )
        elif self.selection_config["method"] == "no_selection":
            self.population = offspring

        # --- update info ---
        self.num_gens += 1
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