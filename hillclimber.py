import functools
import time
import pickle
from collections import Counter

import numpy as np

from afpo import Solution, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, DEAD, ALIVE, ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_TANH
from simulation_cpu import simulate, get_layer_mask, DEAD, ALIVE, WORLD_SIZE, NUM_STEPS, NUM_LAYERS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, ACTIVATION_SIGMOID, ACTIVATION_RELU, ACTIVATION_TANH
from util import create_hollow_circle, create_square, create_diamond

activation2int = {
    'sigmoid': ACTIVATION_SIGMOID,
    'tanh': ACTIVATION_TANH,
    'relu': ACTIVATION_RELU
}

class HillClimber:
    def __init__(self, experiment_constants):
        self.max_generations = experiment_constants['max_generations']
        self.target_population_size = experiment_constants['target_population_size']
        self.layers = experiment_constants['layers']
        self.use_growth = experiment_constants['use_growth']
        self.activation = experiment_constants['activation']
        self.shape = experiment_constants['shape']
        self.mutate_layers = experiment_constants['mutate_layers']
        self.state_or_growth = experiment_constants['state_or_growth']
        self.neighbor_map_type = experiment_constants['neighbor_map_type']

        self.n_layers = len(self.layers)
        self.base_layer = next((i for i, d in enumerate(self.layers) if d.get('base', False)), None)
        self.parent_population = {}
        self.children_population = {}
        self.current_generation = 1
        self.num_ids = 0

        self.below_map = self.initialize_below_map()
        self.above_map = self.initialize_above_map()

        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.parent_child_distance_history = []
        self.n_neutral_over_generations = []
        self.mutation_data_over_generations = []
        self.mutation_layers_over_generations = []

    def evolve(self):
        self.initialize_population()
        while self.current_generation <= self.max_generations:
            print(f'Generation {self.current_generation}')
            self.evolve_one_generation()
            self.current_generation += 1

        return self.best_solution()

    def evolve_one_generation(self):
        # Actually run the simulations, and time how long it takes.
        start = time.perf_counter()

        unsimulated_growth_genotypes, unsimulated_state_genotypes, unsimulated_ids = self.get_unsimulated_genotypes()
        init_phenotypes = self.make_seed_phenotypes(self.target_population_size)

        rand_id = list(self.children_population.keys())[0]

        ##### SIMULATE ON GPUs #####
        print(f'Starting {self.target_population_size} simulations...')
        phenotypes = simulate(
            unsimulated_growth_genotypes, 
            unsimulated_state_genotypes, 
            self.n_layers, 
            self.base_layer,  
            self.children_population[rand_id].around_start, 
            self.children_population[rand_id].above_start, 
            self.use_growth, 
            init_phenotypes, 
            activation2int[self.activation],
            self.below_map,
            self.above_map)

        elapsed = time.perf_counter() - start
        lps = self.target_population_size / elapsed
        print(f'Finished in {elapsed:0.2f} seconds ({lps:0.2f} lifetimes per second).')

        fitness_scores = self.evaluate_phenotypes(phenotypes)
        parent_child_distances = []
        # Set the fitness and simulated flag for each of the just-evaluated solutions
        for i, id in enumerate(unsimulated_ids):
            self.children_population[id].set_fitness(fitness_scores[i])
            self.children_population[id].set_simulated(True)
            self.children_population[id].set_phenotype(phenotypes[i][self.base_layer][-1] > 0) # phenotype is now binarized last step of base layer
            # Get actual parent Solution object from population using parent_id
            parent_id = self.children_population[id].parent_id
            parent = self.parent_population[parent_id] if parent_id is not None else None
            if parent is not None:
                parent_child_distances.append(self.children_population[id].get_distance_from_parent(parent))

        # Reduce the population by selecting parent or child to remove
        n_neutral_children = self.select()
        # Extend the population using tournament selection
        mutation_data, mutation_layers = self.mutate_population()

        print(mutation_data)
        print('Average fitness:',
              np.mean([sol.fitness for id, sol in self.parent_population.items()]),
              ', Min fitness: ',
              min([sol.fitness for id, sol in self.parent_population.items()]))
        print('Average age:',
              np.mean([sol.age for id, sol in self.parent_population.items()]))
        print('Proportion of neutral mutations: ', n_neutral_children / self.target_population_size)
        
        # if self.children_population[rand_id].parent_id is not None:
        #     print(self.children_population[rand_id].mutation_info)
        #     print(self.children_population[rand_id].state_genotype)
        #     print(self.parent_population[self.children_population[rand_id].parent_id].state_genotype)
        #     print(Counter([solution.mutation_info['layer'] for i, solution in self.children_population.items()]))
        
        self.mutation_data_over_generations.append(mutation_data)
        self.n_neutral_over_generations.append(n_neutral_children)
        self.best_fitness_history.append(self.best_solution())
        self.mean_fitness_history.append(np.mean([sol.fitness for id, sol in self.parent_population.items()]))
        self.parent_child_distance_history.append(parent_child_distances)
        self.mutation_layers_over_generations.append(mutation_layers)


    def initialize_population(self):
        # Initialize target_population_size random solutions
        for _ in range(self.target_population_size):
            new_id = self.get_available_id()
            self.children_population[new_id] = Solution(layers=self.layers, id=new_id)

    def select(self):
        """
        Look through children, compare fitness with parents, keep the best of the two.
        """
        n_neutral_children = 0
        for child_id, child in self.children_population.items():
            # Get actual parent Solution object from population using parent_id
            parent_id = child.parent_id
            parent = self.parent_population[parent_id] if parent_id is not None else None
            if parent is not None:
                if child.fitness == parent.fitness:
                    n_neutral_children += 1
                if child.fitness <= parent.fitness:
                    del self.parent_population[parent_id]
                    self.parent_population[child_id] = child
            else:
                self.parent_population[child_id] = child

        return n_neutral_children


    def mutate_population(self):
        """
        Mutate each individual in the population  
        """
        mutation_data = []
        self.children_population = {}
        # Make a new child from every parent
        for id, solution in self.parent_population.items():
            child = solution.make_offspring(id, mutate_layers=self.mutate_layers, state_or_growth=self.state_or_growth)
            mutation_data.append(child.mutation_info)
            self.children_population[id] = child

        aggregate_mutation_data = {
            'type': dict(Counter([mutation_info['type'] for mutation_info in mutation_data])),
            'kind': dict(Counter([mutation_info['kind'] for mutation_info in mutation_data])),
            'layer': dict(Counter([mutation_info['layer'] for mutation_info in mutation_data])),
        }
        mutation_layers = [mutation_info['layer'] for mutation_info in mutation_data]

        return aggregate_mutation_data, mutation_layers


    def get_unsimulated_genotypes(self):
        # Filter out just the genotypes that haven't been simulated yet.
        unsimulated_growth_genotypes = [
            sol.growth_genotype for _, sol in self.children_population.items() if not sol.been_simulated
        ]
        unsimulated_state_genotypes = [
            sol.state_genotype for _, sol in self.children_population.items() if not sol.been_simulated
        ]
        unsimulated_ids = [
            id for id, sol in self.children_population.items() if not sol.been_simulated
        ]
        # Aggregate the genotypes into a single matrix for simulation
        return np.array(unsimulated_growth_genotypes, dtype=np.float32), np.array(unsimulated_state_genotypes, dtype=np.float32), unsimulated_ids

    def get_target_shape(self):
        """
        Returns the target shape using self.shape and the resolution of self.base_layer.
        Always returns 64x64 numpy array. 
        """

        resolution = self.layers[self.base_layer]['res']
        target_size = WORLD_SIZE // resolution

        assert resolution in [2**n for n in range(int(np.log2(WORLD_SIZE)) + 1)]

        if self.shape == 'square':
            target = create_square(target_size)
        elif self.shape == 'diamond':
            target = create_diamond(target_size)
        elif self.shape == 'circle':
            target = create_hollow_circle(target_size)

        # Upsize back to 64x64 because that's how we're comparing 
        # while target.shape[0] < 64: 
        #     target = np.repeat(np.repeat(target, 2, axis=0), 2, axis=1)
        # assert target.shape[0] == 64

        return target
        

    def evaluate_phenotypes(self, phenotypes):
        """Score a set of phenotypes generated by the simulate function."""
        target = self.get_target_shape()

        assert target.shape == (phenotypes[0][self.base_layer][-1]).shape

        # Infer pop_size from phenotypes
        pop_size = len(phenotypes)

        # Allocate space for results.
        fitness_scores = np.zeros(pop_size, dtype=np.uint32)

        # For each individual in the population...
        for i in range(pop_size):
            # Look at just the final state of the layer0 part of the phenotype.
            # Compare it to the target image, and sum up the deltas to get the
            # final fitness score (lower is better, 0 is a perfect score).
            fitness_scores[i] = np.sum(np.abs(target - (phenotypes[i][self.base_layer][-1] > 0)))

        return fitness_scores


    def make_seed_phenotypes(self, n):
        """Starting phenotypes to use by default (one ALIVE cell in middle)."""
        # For each inidividual, capture phenotype development over NUM_STEPS. Each
        # phenotype has NUM_LAYERS layers which are all WORLD_SIZE x WORLD_SIZE
        # squares. Layer0 is the "ground truth" of the CA while layers 1 and 2
        # represent a sort of hierarchical internal state for the organism. Layers
        # 1 and 2 are conceptually smaller than layer0 (1/4 and 1/8 respectively),
        # but are represented using arrays of the same size for simplicity.
        phenotypes = []
        for _ in range(n):
            phenotype = []
            g = WORLD_SIZE 
            for l in range(self.n_layers):
                layer = np.zeros((NUM_STEPS,g,g))
                phenotype.append(layer)
                if l == self.base_layer:
                    phenotype[self.base_layer][0,g//2,g//2] = ALIVE
                g = g // 2
            phenotypes.append(phenotype)

        return phenotypes
    
    def initialize_below_map(self):
        below_map = np.zeros((self.n_layers, 4, 3)).astype(int)
        if self.neighbor_map_type == 'random':
            for l in range(self.n_layers):
                rand_l = np.random.randint(self.n_layers)
                rand_r_offset = np.random.randint(WORLD_SIZE)
                rand_c_offset = np.random.randint(WORLD_SIZE)
                below_map[l, 0] = [rand_l, rand_r_offset, rand_c_offset]
        else:
            for l in range(self.n_layers):
                below_map[l, 0] = [l-1, 0, 0]
                below_map[l, 1] = [l-1, 0, 1]
                below_map[l, 2] = [l-1, 1, 0]
                below_map[l, 3] = [l-1, 1, 1]

        return below_map
    
    def initialize_above_map(self):
        above_map = np.zeros((self.n_layers, 3)).astype(int)
        if self.neighbor_map_type == 'random':
            for l in range(self.n_layers):
                rand_l = np.random.randint(self.n_layers)
                rand_r_offset = np.random.randint(WORLD_SIZE)
                rand_c_offset = np.random.randint(WORLD_SIZE)
                above_map[l] = [rand_l, rand_r_offset, rand_c_offset]
        else:
            for l in range(self.n_layers):
                above_map[l] = [l+1, 0, 0]

        return above_map


    def pickle_hc(self, pickle_file_name):
        with open(pickle_file_name, 'wb') as pf:
            pickle.dump(self, pf, protocol=pickle.HIGHEST_PROTOCOL)

    def get_available_id(self):
        self.num_ids += 1
        return self.num_ids

    def best_solution(self):
        return min([sol for id, sol in self.parent_population.items() if sol.fitness is not None])
