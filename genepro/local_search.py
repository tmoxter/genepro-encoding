import numpy as np
import matplotlib.pyplot as plt
import imageio, os
from genepro.node import Node
from genepro.encoding import Encoder
import genepro.comptree_generators as treegen
from typing import Tuple


class RandomWalk(Encoder):
    """Searches the landscape locally by making small moves in the encoded space, which
    are expected to resemble small changes with reagard to tree fitness. Moves that
    result in worse fitness are always discarded. Inherits from the Encoder to perform
    local moves in the encoded space.
        
    Parameters
    -----
    unaryNodes :list: unary atomic functions available, binaryNodes :list: binary atomic functions
    available,leafNodes :list: leaf nodes available, train_x :np.ndarray: training data (x),
    train_y :np.ndarray: training data (y). """

    def __init__(self, unaryNodes: list, binaryNodes: list, leafNodes: list,
        train_x : np.ndarray, train_y : np.ndarray) -> None:
        super().__init__(unaryNodes, binaryNodes, leafNodes)
        self.train_x = train_x
        self.train_y = train_y
        self.plot = False
    
    def local_variation(self, tree : Node, step_size : float) -> Node:
        """Get new tree that is a local (in ecoding) variation of original tree.
        ----
        Return :Node: New tree after variation"""
        
        encoding = self.encode(tree, self.train_x)
        shpx, shpy, shpz = encoding.shape
        rand_angle = np.random.randn(shpx, shpy, shpz)
        rand_angle = rand_angle / np.linalg.norm(rand_angle) * step_size
        variation = encoding + rand_angle
        tree = self.decode(variation)
        
        return tree
    
    def search(self, step_size : float, n_attempts : int, restarts : int,
                                        fitness : callable = None) -> Tuple:
        """Perform search in the embedded space: perform local variation, discard
        if less fit, keep if more. Step size is uniformally sampled between 0
        and given step size. If not fitness function is provided, MSE is used.
        
        Parameters
        -----
        step_size :float: upper limit for the r2 distance moved in ecoded space,
        n_attempts :int: number of consecutive steps performed without an improvement,
        restarts :int: number of restarts, fitness :callable: if provided, needs to 
        have call singniture fitness(tree : Node).
        
        Returns
        -----
        best_of_runs_f :float:, best_of_runs_t :Node:, track :list: accepted variations 
        of shape (f, t)
        """

        if not fitness:
            fitness = lambda tree: np.mean(
                np.power(tree(self.train_x)-self.train_y, 2)
                )
        # --- initialize ---
        t0 = treegen.sample_tree(self.unaryNodes, self.binaryNodes,
                                                self.leafNodes, 2)
        best_of_runs_f = fitness(t0)
        best_of_runs_t = t0
        track = []
        # --- for plotting ---
        lower = np.min(self.train_y) - (np.max(self.train_y) 
                                     -  np.min(self.train_y))/ 2 * .5
        upper = np.min(self.train_y) + (np.max(self.train_y) 
                                     -  np.min(self.train_y))/ 2 * 2.5
        # --- main loop ---
        for i in range(restarts):
            print("(Re)start: %s..." %i)
            # --- --- (re)initialize --- ---
            sample = []
            j = 0
            best_of_var_f = fitness(t0)
            best_of_var_t = t0
            vc = 0
            # --- --- loop for 1 walk --- ---
            while j < n_attempts:
                vc += 1
                var_t = self.local_variation(best_of_var_t,
                        step_size=np.random.random()*step_size)
                var_f = fitness(var_t)

                # --- --- plotting --- ---
                if self.plot:
                    sample.append(var_t)
                    fig, ax = plt.subplots()
                    plt.ylim(lower, upper)
                    plt.grid()
                    self._plot_functions(self.train_x, ax, sample, best_of_var_t)
                    ax.plot(self.train_x, self.train_y, 'g^')
                    plt.title("Restart : %s, step: %s, stepsize: %s" % \
                                        (i, j+1, np.round(step_size, 5)))
                    fig.savefig("temp/"+str(i*n_attempts + vc)+".png")
                    plt.close()
                # --- ---

                if var_f < best_of_var_f:
                    best_of_var_t = var_t
                    best_of_var_f = var_f
                    j = 0
                j+=1
            # --- ---

            track.append((best_of_var_f, best_of_var_t))
            if best_of_var_f < best_of_runs_f:
                best_of_runs_t = best_of_var_t
                best_of_runs_f = best_of_var_f
                if best_of_runs_f < 1e-6:
                    print("Terminating sucessfully...")
                    break
            t0 = treegen.sample_tree(self.unaryNodes, self.binaryNodes,
                                                    self.leafNodes, 2)
        # ---
        
        if self.plot:
            self._create_gif()
        print("Done")
        return best_of_runs_f, best_of_runs_t, track

    def _plot_functions(self, x, ax, sample, best):
        """Create a plot which serves as a frame in the video."""
        for t in sample[-10:]:
            o = t.get_output(x)
            ax.plot(x, o, '.', color="black", alpha=0.1)

        # plot also the current best
        o = best.get_output(x)
        ax.plot(x, o, 'g.-',linewidth=3)
    
    def _create_gif(self):
        """Create a video of the search process."""
        os.makedirs("temp", exist_ok=True)
        images = list()
        pngs = sorted([x for x in os.listdir("temp/") if x.endswith(".png")],
                                    key=lambda x: int(x.replace(".png","")))
        for filename in pngs:
            images.append(imageio.imread("temp/"+filename))
        # pause on last one
        for _ in range(int(0.5*len(images))):
            images.append(images[-1])
        imageio.mimsave('temp/movie.gif', images)

class SimulatedAnnealing(Encoder):
    """Simulated annealing procedure for a single solution candidate. Local changes are made,
    of which 'good' moves are always accepted, while 'bad' moves have a chance of being accepted
    depending both on the current temperature and the loss of fitness during the move.
    Inherits from the Encoder to perform local moves in the encoded space.
        
    Parameters
    -----
    unaryNodes :list: unary atomic functions available, binaryNodes :list: binary atomic functions
    available, leafNodes :list: leaf nodes available, """

    def __init__(self, unaryNodes: list, binaryNodes: list, leafNodes: list,
                train_x : np.ndarray, train_y : np.ndarray) -> None:
        super().__init__(unaryNodes, binaryNodes, leafNodes)
        self.train_x = train_x
        self.train_y = train_y
        self.plot = False
    
    def local_variation(self, tree : Node, step_size : float) -> Node:
        """Get new tree that is a local (in ecoding) variation of original tree.
        ----
        Return :Node: New tree after variation"""
        
        encoding = self.encode(tree, self.train_x)
        shpx, shpy, shpz = encoding.shape
        rand_angle = np.random.randn(shpx, shpy, shpz)
        rand_angle = rand_angle / np.linalg.norm(rand_angle) * step_size
        variation = encoding + rand_angle
        tree = self.decode(variation)
        
        return tree
    

    def search(self, step_size : float, restarts : int, kmax : int = 50, 
                temp0 : float = None, temp_terminate : float = 0.5,
                fitness : callable = None, t0 = None) -> Tuple:

        """Perform search in the embedded space according to the simulated annealing procedure.
        If no fitness function is provided, MSE is used.
        
        Parameters
        -----
        step_size :float: upper limit for the r2 distance moved in ecoded space,
        restarts :int: number of restarts, kmax :int: number of steps performed
        at every temprerature, temp0 :float: initial temperature, if not provided it is
        set to temp0 = (-U)∕ ln(Racc), where U is the median fitness increase during uphill moves in
        a random walk with large step size to explore the landscape, and Racc is the desired initial
        acceptance rate set here to 0.9 [Rozenberg et al., 'Handbook of Natural Computing',
        DOI 10.1007/978-3-540-92910-9, pp.1684], temp_terminate :float: final temperature at termination,
        fitness :callable: if provided, needs to have call singniture fitness(tree : Node).
        
        Returns
        -----
        best_of_runs_f :float:, best_of_runs_t :Node:, track :list: accepted variations 
        of shape (f, t)
        """
        
        if not fitness:
            fitness = lambda tree: np.mean(
                np.power(tree(self.train_x)-self.train_y, 2)
                )
        
        if not t0:
            t0 = treegen.sample_tree(self.unaryNodes, self.binaryNodes,
                                                    self.leafNodes, 2)

        # --- find initial temperature ---
        # -> Should be problem-specific and is therefore kept for all runs
        #    in reality it might be reasonable to recalculate temp0 for each new t0 in every run
        if not temp0:
            track = []
            best_of_var_f = fitness(t0)
            best_of_var_t = t0
            # --- --- random walk loop --- ---
            for _ in range(100):
                var_t = self.local_variation(best_of_var_t,
                        step_size=np.random.random()*step_size)
                var_f = fitness(var_t)
                if var_f > best_of_var_f:
                    track.append(var_f - best_of_var_f)
                best_of_var_t = var_t
                best_of_var_f = var_f
            u = np.median(track)
            temp0 = -u / np.log(0.9)
            print("Initial T: {:.4f}".format(temp0))
        # --- 

        best_of_runs_f = fitness(t0)
        best_of_runs_t = t0
        track = []

        # --- for plotting ---
        lower = np.min(self.train_y) - (np.max(self.train_y) 
                                     -  np.min(self.train_y))/ 2 * .5
        upper = np.min(self.train_y) + (np.max(self.train_y) 
                                     -  np.min(self.train_y))/ 2 * 2.5
        # --- main loop ---
        for i in range(restarts):
            print("(Re)start: %s..." %i)
            sample = []

            # --- Initialize ---
            temp = temp0
            state_t = t0
            state_f = fitness(t0)
            best_of_var_f = state_f
            best_of_var_t = state_t
            tcnt = 0
            while temp > temp_terminate:
                tcnt += 1
                print("New temperature: {:.4f}".format(temp))
                print("Current best: {:.4f}".format(best_of_var_f))
                nrep = kmax
                fdev = []
                for _ in range(nrep):
                    var_t = self.local_variation(state_t, step_size)
                    var_f = fitness(var_t)
                    fdev.append(var_f)
                    sample.append(var_t)

                    if var_f < state_f:
                        state_f = var_f
                        state_t = var_t
                    elif np.random.random() < np.exp(-(state_f - var_f)/temp):
                        state_f = var_f
                        state_t = var_t
                    
                    # --- save best state visited in this run ---
                    if state_f < best_of_var_f:
                        best_of_var_f = state_f
                        best_of_var_t = state_t
                    # ---
                # --- --- plotting --- ---
                if self.plot:
                    sample.append(var_t)
                    fig, ax = plt.subplots()
                    plt.ylim(lower, upper)
                    plt.grid()
                    self._plot_functions(self.train_x, ax, sample, best_of_var_t)
                    ax.plot(self.train_x, self.train_y, 'g^')
                    plt.title("Restart : %s, T: %s" % (i, 
                                        np.round(temp, 5)))
                    fig.savefig("temp/"+str(tcnt)+".png")
                    plt.close()
                # --- ---
                if best_of_var_f < 0.005 * (np.max(np.abs(self.train_y))
                                          - np.min(np.abs(self.train_y))):
                    print("Terminating sucessfully...")
                    break
                # ---- --- temperature lowering schedule, alpha = 0.98 --- ---
                temp = temp * 0.98 #np.exp(-0.75 * temp / np.std(fdev)) needs testing
    
            track.append((best_of_var_f, best_of_var_t))
            if best_of_var_f < best_of_runs_f:
                best_of_runs_t = best_of_var_t
                best_of_runs_f = best_of_var_f

            t0 = treegen.sample_tree(self.unaryNodes, self.binaryNodes, self.leafNodes, 2)
        # ---
        
        if self.plot:
            self._create_gif()
        print("Done")
        return best_of_runs_f, best_of_runs_t, track

    def _plot_functions(self, x, ax, sample, best):
        """Create a plot which serves as a frame in the video."""
        for t in sample[-10:]:
            o = t.get_output(x)
            ax.plot(x, o, '.', color="black", alpha=0.1)

        # plot also the current best
        o = best.get_output(x)
        ax.plot(x, o, 'g.-',linewidth=3)
    
    def _create_gif(self):
        """Create a video of the search process."""
        os.makedirs("temp", exist_ok=True)
        images = list()
        pngs = sorted([x for x in os.listdir("temp/") if x.endswith(".png")],
                                    key=lambda x: int(x.replace(".png","")))
        for filename in pngs:
            images.append(imageio.imread("temp/"+filename))
        # pause on last one
        for _ in range(int(0.5*len(images))):
            images.append(images[-1])
        imageio.mimsave('temp/movie.gif', images)