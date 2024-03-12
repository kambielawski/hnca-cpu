
import argparse
import pickle
import time
import numpy as np
from simulation_cpu import simulate
from afpo import AgeFitnessPareto, activation2int, Solution

# Parse command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('params_file', type=str, help='Parameters file')
# args = parser.parse_args()

# # Read the experiment file into exp_arms variable
# params_file = open(args.params_file)
# params_string = params_file.read()
# params = eval(params_string)
# params_file.close()

params = {
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
        {'res': 1},
        {'res': 2},
        {'res': 4},
        {'res': 8, 'base': True},
    ],
    'use_growth': True,
    'activation': 'sigmoid'
  }

base_layer = next((i for i, d in enumerate(params['layers']) if d.get('base', False)), None)
n_layers = len(params['layers'])

N_TOTAL = 100000
BATCH_SIZE = 100
N_BATCHES = N_TOTAL // BATCH_SIZE

parent_child_distance = np.zeros(N_TOTAL)
parent_child_fitness_pairs = np.zeros((N_TOTAL, 2))

for batch_idx in range(N_BATCHES):

    start = time.time()

    afpo = AgeFitnessPareto(params)

    # Make parents
    parent_population = [Solution(layers=params['layers'], id=afpo.get_available_id()) for _ in range(BATCH_SIZE)]
    parent_unsimulated_growth_genotypes = np.array([sol.growth_genotype for sol in parent_population])
    parent_unsimulated_state_genotypes = np.array([sol.state_genotype for sol in parent_population])
    seed_phenotypes = afpo.make_seed_phenotypes(BATCH_SIZE)

    # Simulate parents
    phenotypes = simulate(
        parent_unsimulated_growth_genotypes, 
        parent_unsimulated_state_genotypes, 
        n_layers, 
        base_layer,  
        parent_population[0].around_start, 
        parent_population[0].above_start, 
        params['use_growth'], 
        seed_phenotypes, 
        activation2int[params['activation']],
        afpo.below_map,
        afpo.above_map)
    
    parents_binarized_phenotypes = [solution[base_layer][-1] > 0 for solution in phenotypes]
    parents_fitness = afpo.evaluate_phenotypes(phenotypes)

    # Mutate and get BATCH_SIZE children
    children_population = [solution.make_offspring(new_id=afpo.get_available_id()) for solution in parent_population]
    children_unsimulated_growth_genotypes = np.array([sol.growth_genotype for sol in children_population])
    children_unsimulated_state_genotypes = np.array([sol.state_genotype for sol in children_population])

    # Simulate children 
    phenotypes = simulate(
        children_unsimulated_growth_genotypes, 
        children_unsimulated_state_genotypes, 
        n_layers, 
        base_layer,  
        children_population[0].around_start, 
        children_population[0].above_start, 
        params['use_growth'], 
        seed_phenotypes, 
        activation2int[params['activation']],
        afpo.below_map,
        afpo.above_map)
    
    children_binarized_phenotypes = [solution[base_layer][-1] > 0 for solution in phenotypes]
    children_fitness = afpo.evaluate_phenotypes(phenotypes)

    # assert children_fitness.shape == (BATCH_SIZE, )

    # Compare binarized phenotypes of children and parents
    for i in range(BATCH_SIZE):
        parent_child_fitness_pairs[batch_idx*BATCH_SIZE + i] = (parents_fitness[i], children_fitness[i])
        parent_child_distance[batch_idx*BATCH_SIZE + i] = np.count_nonzero(parents_binarized_phenotypes[i] == children_binarized_phenotypes[i])
        
    batch_exec_time = time.time() - start

    print(f'Batch {batch_idx+1}/{N_BATCHES} took {batch_exec_time:.2f} seconds')

i=10
print(parents_fitness[i], children_fitness[i])
print(parents_binarized_phenotypes[i].shape)
print(parents_binarized_phenotypes[i])
print(children_binarized_phenotypes[i])
print((parents_binarized_phenotypes[i] == children_binarized_phenotypes[i]).all())

print(sum([pair[0] == pair[1] for pair in parent_child_fitness_pairs]))

print(parent_unsimulated_state_genotypes[i])
print(children_unsimulated_state_genotypes[i])

date_and_time = time.strftime('%Y-%m-%d_%H-%M-%S')

# Save results to pickle file
with open(f'./experiments/parent_child_phenotypes/parent_child_same_phenotype_n{N_TOTAL}_{date_and_time}.pkl', 'wb') as f:
    results = {
        'parent_child_distance': parent_child_distance,
        'parent_child_fitness_pairs': parent_child_fitness_pairs
    }
    pickle.dump(results, f)
    