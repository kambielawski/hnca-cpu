import os
import time
from afpo import AgeFitnessPareto, activation2int
import numpy as np

from simulation_cpu import simulate, simulate_parallel

print('num cores: ', os.cpu_count())

exp_params = {
    'optimizer': 'afpo',
    'num_trials': 20,
    'target_population_size': 100,
    'max_generations': 2000,
    'state_or_growth': None,
    'neighbor_map_type': 'random',
    'mutate_layers': None,
    'sim_steps': 100,
    'shape': 'square',
    'layers': [
        {'res': 1, 'base': True},
        {'res': 2},
        {'res': 4},
        {'res': 8},
    ],
    'use_growth': True,
    'activation': 'sigmoid'
  }

afpo = AgeFitnessPareto(exp_params)
afpo.initialize_population()

growth_genotypes = np.array([solution.growth_genotype for solution in afpo.population])
state_genotypes = np.array([solution.state_genotype for solution in afpo.population])
base_layer = 1
around_start = afpo.population[0].around_start
above_start = afpo.population[0].above_start
use_growth = True
activation = activation2int['sigmoid']
below_map = afpo.initialize_below_map()
above_map = afpo.initialize_above_map()

def make_seed_phenotypes(n, n_layers, n_timesteps=100):
    phenotypes = []
    for _ in range(n):
        phenotype = []
        g = 64
        for l in range(n_layers):
            layer = np.zeros((n_timesteps,g,g))
            phenotype.append(layer)
            g = g // 2

        phenotype[0][0,32,32] = 1
        phenotypes.append(phenotype)

    return phenotypes

pop_size = exp_params['target_population_size']
n_layers = len(exp_params['layers'])
init_phenotypes = make_seed_phenotypes(pop_size, n_layers)

print(len(init_phenotypes))
print([l.shape for l in init_phenotypes[0]])


start_time = time.time()
first_phenotypes = simulate(
    growth_genotypes,
    state_genotypes,
    n_layers,
    base_layer,
    around_start,
    above_start,
    use_growth,
    init_phenotypes,
    activation,
    below_map,
    above_map
)
print(f'Sequential time: {time.time() - start_time}')

start_time = time.time()
second_phenotypes = simulate(
    growth_genotypes,
    state_genotypes,
    n_layers,
    base_layer,
    around_start,
    above_start,
    use_growth,
    init_phenotypes,
    activation,
    below_map,
    above_map
)
print(f'Parallel time: {time.time() - start_time}')

n_same = 0
for i in range(pop_size):
    print([(first_phenotypes[i][l] == second_phenotypes[i][l]).all() for l in range(n_layers)])
    if all([(first_phenotypes[i][l] == second_phenotypes[i][l]).all() for l in range(n_layers)]):
        n_same += 1
    else:
        print(first_phenotypes[0][0][1])
        print(second_phenotypes[0][0][1])

print('n same: ', n_same)

print(first_phenotypes[0][0][1])
print((first_phenotypes[0][0][-1] > 0).astype(int))



