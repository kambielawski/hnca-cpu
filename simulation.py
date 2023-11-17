import time

from numba import cuda
import numpy as np
from PIL import Image

# Phenotype development unfolds over NUM_STEPS time steps.
NUM_STEPS = 100
# This code models hierarchical CAs with 1, 2, or 3 layers.
NUM_LAYERS = 3
# Phenotypes are all 64x64 squares.
WORLD_SIZE = 64

# All cells in all layers have one of these two states.
ALIVE = 1
DEAD = 0

# Max number of threads per block on a 8.6cc GPU
MAX_THREADS_PER_BLOCK = 1024
# Break up each row into COL_BATCH_SIZE groups of COLS_PER_THREAD cells each.
COL_BATCH_SIZE = MAX_THREADS_PER_BLOCK // WORLD_SIZE
COLS_PER_THREAD = WORLD_SIZE // COL_BATCH_SIZE

# Each cell can have neighbors on the layer below, the same layer, or the layer
# above, and each of those neighbors gets a single weight value.
NUM_DOWN_WEIGHTS = 4
NUM_AROUND_WEIGHTS = 9
NUM_UP_WEIGHTS = 1
NUM_INPUT_NEURONS = NUM_DOWN_WEIGHTS + NUM_AROUND_WEIGHTS + NUM_UP_WEIGHTS
# Each cell at the bottom level has 5 output neurons
# 4 binary outputs for spreading up/left/right/down, 1 for self signal state
NUM_OUTPUT_NEURONS = 5

# The genome consists of weights for all layers and an activation function
# which is represented with four scalar values.
LAYER_GENOME_SHAPE = (NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS) 

# Genome indices for thersholds of a non-linear activation function:
# LONELINESS_THRESHOLD = 0
# SPAWN_THRESHOLD_LOW = LONELINESS_THRESHOLD + 1
# SPAWN_THRESHOLD_HIGH = SPAWN_THRESHOLD_LOW + 1
# OVERPOPULATION_THRESHOLD = SPAWN_THRESHOLD_HIGH + 1
# Genome indices for neighbor weights:
DOWN_WEIGHTS_START = 0
AROUND_WEIGHTS_START = DOWN_WEIGHTS_START + NUM_DOWN_WEIGHTS
UP_WEIGHTS_START = AROUND_WEIGHTS_START + NUM_AROUND_WEIGHTS


# @cuda.jit
# def activate_binary(layer, genotypes, pop_idx, prev_state, weighted_sum):
#     """Compute a cell's new value given the weighted sum of its neighbors."""
#     if weighted_sum <= genotypes[pop_idx][layer][LONELINESS_THRESHOLD]:
#         return DEAD
#     if (weighted_sum >= genotypes[pop_idx][layer][SPAWN_THRESHOLD_LOW] and
#         weighted_sum <= genotypes[pop_idx][layer][SPAWN_THRESHOLD_HIGH]):
#         return ALIVE
#     if weighted_sum >= genotypes[pop_idx][layer][OVERPOPULATION_THRESHOLD]:
#         return DEAD
#     return prev_state

@cuda.jit
def activate_sigmoid(weighted_sum):
    return 1 / (1 + np.e ** (-weighted_sum))


@cuda.jit
def look_down(layer, phenotypes, genotypes, pop_idx, step, row, col):
    """Compute the weighted sum of this cell's neighbors in the layer below."""
    if layer == 0:
        return 0

    # The "granularity" of this layer. Layer0 == 1, layer1 == 2, layer2 == 4.
    g = 1 << layer

    weight_index = DOWN_WEIGHTS_START
    result = 0
    # Look at all the cells that occupy "the same position" as this cell but
    # one layer down.
    for r in range((row // g)*g, (row // g)*g + g):
        for c in range((col // g)*g, (col // g)*g + g):
            # Do wrap-around bounds checking. This may be inefficient, since
            # we're doing extra modulus operations and working on
            # non-contiguous memory may prevent coallesced reads. However, it's
            # simple, and avoids complications when working with different
            # granularities.
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            neighbor_state = phenotypes[pop_idx][step-1][layer-1][r][c]
            weight = genotypes[pop_idx, layer][weight_index, 0]
            result += neighbor_state * weight
            weight_index += 1
    return result


@cuda.jit
def look_around(layer, phenotypes, genotypes, pop_idx, step, row, col):
    """Compute the weighted sum of this cell's neighbors in this layer."""
    # The "granularity" of this layer. Layer0 == 1, layer1 == 2, layer2 == 4.
    g = 1 << layer

    weight_index = AROUND_WEIGHTS_START
    result = 0
    # Look at a Moore neighborhood around (row, col) in the current layer.
    for r in range(row-g, row+g+1, g):
        for c in range(col-g, col+g+1, g):
            # Do wrap-around bounds checking. This may be inefficient, since
            # we're doing extra modulus operations and working on
            # non-contiguous memory may prevent coallesced reads. However, it's
            # simple, and avoids complications when working with different
            # granularities.
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            neighbor_state = phenotypes[pop_idx][step-1][layer][r][c]
            weight = genotypes[pop_idx, layer][weight_index, 0]
            result += neighbor_state * weight
            weight_index += 1
    return result


@cuda.jit
def look_up(layer, phenotypes, genotypes, pop_idx, step, row, col):
    """Compute the weighted sum of this cell's neighbors in the layer above."""
    if layer == 2:
        return 0
    # Look at just the single neighbor in the next layer up.
    neighbor_state = phenotypes[pop_idx][step-1][layer+1][row][col]
    weight = genotypes[pop_idx, layer][UP_WEIGHTS_START, 0]
    return neighbor_state * weight

@cuda.jit
def get_spread_update(phenotypes, genotypes, pop_idx, step, row, col): # L=0
    up_neighbor = phenotypes[pop_idx][step-1][1][row][col]
    around_neighbors = np.zeros((3,3))
    for i,r in enumerate(range(row-1, row+2, 1)):
        for j,c in enumerate(range(col-1, col+2, 1)):
            r = r % WORLD_SIZE
            c = c % WORLD_SIZE
            around_neighbors[i, j] = phenotypes[pop_idx][step-1][0][r][c]

    # Up, down, left, right spreading



@cuda.jit
def update_cell(layer, phenotypes, genotypes, pop_idx, step, row, col):
    """Compute the next state for a single cell in layer0 from prev states."""

    # Calculate the weighted sum of all neighbors.
    down_signal_sum = look_down(layer, phenotypes, genotypes, pop_idx, step, row, col) # Should return 0 for L=0
    around_signal_sum = look_around(layer, phenotypes, genotypes, pop_idx, step, row, col) 
    up_signal_sum = look_up(layer, phenotypes, genotypes, pop_idx, step, row, col)

    signal_sum = down_signal_sum + around_signal_sum + up_signal_sum

    # Actually update the phenotype state for step on layer1 at (row, col).
    phenotypes[pop_idx][step][layer][row][col] = activate_sigmoid(signal_sum)

    # Update cells to be alive if on L=0 (only if current cell is actually alive)
    if layer == 0 and phenotypes[pop_idx][step][layer][row][col] != 0:
        # Spread to nearby cells... is this necessary?
        # get_spread_update(phenotypes, genotypes, pop_idx, step, row, col)
        pass


# Max registers can be tuned per device. 64 is the most my laptop can handle.
@cuda.jit(max_registers=64)
def simulation_kernel(genotypes, layers, phenotypes):
    """Compute and record the full development process of a population."""
    # Compute indices for this thread.
    pop_idx = cuda.blockIdx.x
    row = cuda.threadIdx.x
    start_col = cuda.threadIdx.y * COLS_PER_THREAD

    # For each step in the simulation (state at step 0 is pre-populated by the
    # caller)...
    for step in range(1, NUM_STEPS):
        # This thread will compute COLS_PER_THREAD contiguous cells from row,
        # starting at start_col.
        for col in range(start_col, start_col + COLS_PER_THREAD):
            # Update the state in every layer this individual uses.
            for layer in range(0, layers[pop_idx] + 1):
                update_cell(layer, phenotypes, genotypes, pop_idx, step, row, col)
        # Make sure all threads have finished computing this step before going
        # on to the next one.
        cuda.syncthreads()

def get_layer_mask(l):
    """Mask the genome (the NN weights) for a given layer"""
    mask = np.zeros(LAYER_GENOME_SHAPE)
    if l == 0:   # L=0: All
        mask[AROUND_WEIGHTS_START:][:] = 1
    elif l == 1: # L=1: All inputs, single output
        mask[:, 0] = 1
    elif l == 2: # L=2: All inputs except above
        mask[DOWN_WEIGHTS_START:UP_WEIGHTS_START, 0] = 1

    return mask


def check_granularity(g, image):
    """Returns True iff image is tiled with g x g squares of a single value."""
    # Scale down by sampling every gth cell in both dimensions.
    scaled_down = image[0::g, 0::g]
    # Scale back up by repeating every cell g times in both dimensions.
    scaled_up = np.repeat(np.repeat(scaled_down, g, 0), g, 1)
    # Check whether the original image matches the resampled version.
    return np.array_equal(image, scaled_up)


def simulate(genotypes, layers, phenotypes):
    """Simulate genotypes and return phenotype videos."""
    # Infer population size from genotypes
    pop_size = genotypes.shape[0]

    # Each individual has a genotype that consists of an activation function
    # and a set of neighbor weights for each layer in the hierarchical CA.
    assert genotypes.shape == (pop_size, 3, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS)
    # assert genotypes.dtype == np.uint8

    # Each individual is configured to have 0, 1, or 2 extra layers. This way,
    # we can simulate individuals from control and experiment in the same
    # batch. Individuals with fewer layers should complete faster. That could
    # hurt performance, but shouldn't because each individual is handled by
    # MAX_THREADS_PER_BLOCK threads, which should mean that whole warps fall
    # out of the computation together.
    assert layers.shape == (pop_size,)
    assert layers.dtype == np.uint8
    assert all(layer in range(NUM_LAYERS) for layer in layers)

    # Copy input data from host memory to device memory.
    d_phenotypes = cuda.to_device(phenotypes)
    d_genotypes = cuda.to_device(genotypes)
    d_layers = cuda.to_device(layers)

    # Actually run the simulation for all individuals in parallel on the GPU.
    simulation_kernel[
        # Each grid contains one block per organism.
        (pop_size,),
        # Organize threads into two dimensions. Each thread computes
        # COLS_PER_THREAD cells in the CA. The X dimension is the row within
        # the CA world to compute, and the Y dimension is multiplied by
        # COLS_PER_THREAD to find the first column to start from.
        (WORLD_SIZE, COL_BATCH_SIZE)
    ](d_genotypes, d_layers, d_phenotypes)

    # Copy output data from device memory to host memory.
    phenotypes = d_phenotypes.copy_to_host()

    # Layer1 in all phenotypes from all steps of the simulation has a
    # granularity of 2x2.
    assert all(check_granularity(2, p)
               for p in np.reshape(
                   phenotypes[:, :, 1], (-1, WORLD_SIZE, WORLD_SIZE)))

    # Layer2 in all phenotypes from all steps of the simulation has a
    # granularity of 4x4.
    assert all(check_granularity(4, p)
               for p in np.reshape(
                   phenotypes[:, :, 2], (-1, WORLD_SIZE, WORLD_SIZE)))

    return phenotypes


def make_seed_phenotypes(pop_size):
    """Starting phenotypes to use by default (one ALIVE cell in middle)."""
    # For each inidividual, capture phenotype development over NUM_STEPS. Each
    # phenotype has NUM_LAYERS layers which are all WORLD_SIZE x WORLD_SIZE
    # squares. Layer0 is the "ground truth" of the CA while layers 1 and 2
    # represent a sort of hierarchical internal state for the organism. Layers
    # 1 and 2 are conceptually smaller than layer0 (1/4 and 1/8 respectively),
    # but are represented using arrays of the same size for simplicity.
    phenotypes = np.full(
        (pop_size, NUM_STEPS, NUM_LAYERS, WORLD_SIZE, WORLD_SIZE),
        DEAD, dtype=np.float32)

    # Use a single ALIVE pixel in the middle of the CA world as the initial
    # phenotype state for all individuals in the population.
    for i in range(pop_size):
        middle = WORLD_SIZE // 2
        phenotypes[i][0][0][middle][middle] = ALIVE

    return phenotypes

def make_seed_genotypes(pop_size):
    """Starting genotypes: random initialization"""
    # Randomly initialize the NN weights
    genotypes = np.random.random((pop_size, 3, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS)).astype(np.float32)
    
    # Mask out the weights of layers 1 and 2
    for l in range(NUM_LAYERS):
        genotypes[:, l] = genotypes[:, l] * get_layer_mask(l)

    return genotypes


def compute_fitness(phenotypes, target):
    """Score a set of phenotypes generated by the simulate function."""
    # Infer pop_size from phenotypes
    pop_size = phenotypes.shape[0]
    # All phenotypes and the target image are WORLD_SIZE x WORLD_SIZE squares.
    assert phenotypes.shape == (
        pop_size, NUM_STEPS, NUM_LAYERS, WORLD_SIZE, WORLD_SIZE)
    assert target.shape == (WORLD_SIZE, WORLD_SIZE)

    # Allocate space for results.
    fitness_scores = np.zeros(pop_size, dtype=np.uint32)

    # For each individual in the population...
    for i in range(pop_size):
        # Look at just the final state of the layer0 part of the phenotype.
        # Compare it to the target image, and sum up the deltas to get the
        # final fitness score (lower is better, 0 is a perfect score).
        fitness_scores[i] = np.sum(np.abs(target - phenotypes[i][-1][0]))

    return fitness_scores


def visualize(phenotype, filename):
    def make_frame(frame_data):
        # Scale up the image 4x to make it more visible.
        frame_data = frame_data.repeat(4, 1).repeat(4, 2)
        layer0, layer1, layer2 = frame_data
        l, w = layer0.shape
        # print(layer0[l//2-1:l//2+5, w//2-1:w//2+5])
        # Make a composite of layers 1 and 2 to overlay on layer0.
        # print(layer1 * 0xffff0000, (layer1 * 0xffff0000).shape)
        # print(layer2 * 0xff00ff00, (layer2 * 0xff00ff00).shape)
        # overlay = np.array(
        #     np.bitwise_or(
        #         layer1 * 0xffff0000,  # layer1 in blue
        #         layer2 * 0xff00ff00), # layer2 in green
        #     dtype=np.uint32)
        # Render layer0 as a black and white image.
        # base = np.array(
        #     np.bitwise_or(
        #         (layer0 == DEAD) * 0xffffffff,   # DEAD cells are black
        #         (layer0 != DEAD) * 0xff000000), # ALIVE cells are white
        #     dtype=np.uint32)
        # Merge the base and overlay images.
        return Image.fromarray(layer0, mode='RGBA')
        # return Image.blend(
        #     Image.fromarray(base, mode='RGBA'),
        #     Image.fromarray(overlay, mode='RGBA'),
        #     0.3)

    frames = [make_frame(frame_data) for frame_data in phenotype]
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0)


def demo():
    """Run a simple demo to sanity check the code above."""
    # The number of simulations to run in one batch. This is limited by the
    # available GPU memory, but the bigger this number the more efficient the
    # simulation will be.
    pop_size = 100

    # Give every individual in the population the same genotype.
    genotypes = make_seed_genotypes(pop_size)

    # Generate a mix of simulations with 0, 1, or 2 layers.
    # Pattern is: 0, 1, 2, 0, 1, 2, 0, 1, 2, ...
    layers = np.zeros(pop_size, dtype=np.uint8)
    layers[1:3] = 1
    layers[2:3] = 2

    phenotypes = make_seed_phenotypes(pop_size)



    # Actually run the simulations, and time how long it takes.
    print(f'Starting {pop_size} simulations...')
    start = time.perf_counter()
    phenotypes = simulate(genotypes, layers, phenotypes)
    elapsed = time.perf_counter() - start
    lps = pop_size / elapsed
    print(f'Finished in {elapsed:0.2f} seconds '
          f'({lps:0.2f} lifetimes per second).')

    # Compute fitness just to run the function through its paces. The target is
    # random, so the result is meaningless.
    target = np.random.choice((DEAD, ALIVE), (WORLD_SIZE, WORLD_SIZE))
    compute_fitness(phenotypes, target)

    # Output a gif video of the demo simulation with 0, 1, and 2 layers.
    visualize(phenotypes[0], 'demo0.gif')
    visualize(phenotypes[1], 'demo1.gif')
    visualize(phenotypes[2], 'demo2.gif')


if __name__ == '__main__':
    demo()
