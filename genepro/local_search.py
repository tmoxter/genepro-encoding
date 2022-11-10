import numpy as np
import matplotlib.pyplot as plt
import imageio, os
from genepro.node import Node
from genepro.encoding import Encoder
import genepro.comptree_generators as treegen


class RandomWalk(Encoder):

    def __init__(self, unaryNodes: list, binaryNodes: list, leafNodes: list,
        train_x : np.ndarray, train_y : np.ndarray) -> None:
        super().__init__(unaryNodes, binaryNodes, leafNodes)
        self.train_x = train_x
        self.train_y = train_y
        self.plot = False
    
    def local_variation(self, tree : Node, step_size : float) -> Node:
        """Get new tree that is a local variation of original tree in its encoded
        representation.
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
                                                fitness : callable = None):
        """Perform search in the embedded space: perform local variation, discard
        if less fit, keep if more. Step size is uniformally sampled between 0 
        and given step size. """

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

        for t in sample[-10:]:
            o = t.get_output(x)
            ax.plot(x, o, '.', color="black", alpha=0.1)

        # plot also the current best
        o = best.get_output(x)
        ax.plot(x, o, 'g.-',linewidth=3)
    
    def _create_gif(self):
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

    def __init__(self, unaryNodes: list, binaryNodes: list, leafNodes: list) -> None:
        super().__init__(unaryNodes, binaryNodes, leafNodes)
    pass
    
