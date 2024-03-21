import pickle
import time
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from afpo import AgeFitnessPareto, Solution, activation2int
from simulation_cpu import simulate, make_seed_phenotypes, DEAD, ALIVE, WORLD_SIZE


def simulate_one_individual(afpo : AgeFitnessPareto, solution : Solution):
    init_phenotypes = make_seed_phenotypes(1, n_layers=solution.n_layers)
    print(solution.n_layers)

    phenotypes = simulate(
            np.array([solution.state_genotype]),
            solution.n_layers,  
            solution.around_start, 
            solution.above_start, 
            phenotypes=init_phenotypes,
            below_map=afpo.below_map,
            above_map=afpo.above_map)
    
    return phenotypes[0]

def visualize_all_layers(phenotype, filename, base_layer_idx=0):
    # def make_frame(frame_data):
    #     n_layers, l, w = frame_data.shape
        
    #     base_layer = frame_data[base_layer_idx]

    #     base = np.array(
    #         np.bitwise_or(
    #             (base_layer != DEAD) * 0xffffffff,   # DEAD cells are black
    #             (base_layer == DEAD) * 0xff000000), # ALIVE cells are white
    #         dtype=np.uint32)

    #     # Calculate the total width for the new image
    #     total_width = w * 4

    #     # Create a new image with the calculated total width
    #     combined_image = Image.new('RGBA', (total_width, l))

    #     # Base layer first
    #     base_img = Image.fromarray(base, mode='RGBA')
    #     combined_image.paste(base_img, (0, 0))

    #     # Convert layers to images
    #     for layer in range(n_layers):
    #         img = Image.fromarray(frame_data[layer], mode='RGBA')
    #         combined_image.paste(img, ((layer+1) * w, 0))

    #     return combined_image

    def make_frame(frame_data):
        n_layers, l, w = frame_data.shape

        # Normalize float values to 0-255 range
        normalized_data = (frame_data * 255).astype(np.uint8)

        # Initialize a new image for combined visualization
        total_width = w * (n_layers + 1) # +1 for the base layer visualization
        combined_image = Image.new('RGBA', (total_width, l))

        base_layer = frame_data[base_layer_idx]

        base = np.array(
            np.bitwise_or(
                (base_layer != DEAD) * 0xffffffff,   # DEAD cells are black
                (base_layer == DEAD) * 0xff000000), # ALIVE cells are white
            dtype=np.uint32)

        # Calculate the total width for the new image
        total_width = w * 5

        # Create a new image with the calculated total width
        combined_image = Image.new('RGBA', (total_width, l))

        # Base layer first
        base_img = Image.fromarray(base, mode='RGBA')
        combined_image.paste(base_img, (0, 0))

        # Process and visualize each layer
        for layer_idx in range(n_layers):
            layer = normalized_data[layer_idx]

            # Create an RGBA image for the layer
            rgba_image = np.zeros((l, w, 4), dtype=np.uint8)
            rgba_image[..., :3] = layer[:, :, None]  # Set R, G, and B channels
            rgba_image[..., 3] = 255  # Set alpha channel

            layer_img = Image.fromarray(rgba_image, mode='RGBA')

            combined_image.paste(layer_img, ((layer_idx+1) * w, 0))

            # # Position for this layer in the combined image
            # if layer_idx == 0:
            #     # Base layer visualization
            #     combined_image.paste(layer_img, (0, 0))
            # else:
            #     # Other layers offset by their index
            #     combined_image.paste(layer_img, (layer_idx * w, 0))

        return combined_image

    n_layers = len(phenotype)
    frames = []
    n_timesteps = phenotype[0].shape[0]
    # print(n_layers)

    for timestep in range(n_timesteps):
        frame_data = []
        g = 1
        for layer in range(n_layers):
            rescaled_layer = np.repeat(np.repeat(phenotype[layer][timestep], g, 0), g, 1)
            # print(rescaled_layer.shape)
            frame_data.append(rescaled_layer)
            # if timestep == 21 and layer == 3:
            #     print(rescaled_layer[:9, :9])
            #     plt.imshow(rescaled_layer)
            #     plt.show()
            #     # exit(1)
            g = g * 2

        frame_data = np.array(frame_data)
        frame = make_frame(frame_data)
        frames.append(frame)
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=10)


def visualize_frames(phenotype, filename, n_frames=4, layer=0):
    def make_image(frames, n_frames, layer):
        l, w = frames[0][0].shape

        # Calculate the total width for the new image
        total_width = w * n_frames

        # Create a new image with the calculated total width
        combined_image = Image.new('RGBA', (total_width, l))

        for f in range(n_frames):
            layer0, layer1, layer2 = frames[f]
            l, w = layer0.shape

            if layer == 'base':
                base = np.array(
                np.bitwise_or(
                    (layer0 != DEAD) * 0xffffffff,   # DEAD cells are black
                    (layer0 == DEAD) * 0xff000000), # ALIVE cells are white
                dtype=np.uint32)
                img = Image.fromarray(base, mode='RGBA')
            else:
                img = Image.fromarray(frames[f][layer], mode='RGBA')

            combined_image.paste(img, (w*f, 0))

        return combined_image

    img = make_image(phenotype, n_frames, layer)
    
    img.save(filename)



def visualize_one_layer(phenotype, filename, layer=0):
    def make_frame(frame_data):
        # Scale up the image 4x to make it more visible.
        # frame_data = frame_data.repeat(4, 1).repeat(4, 2)
        layer0, layer1, layer2 = frame_data
        l, w = layer0.shape
        # Render layer0 as a black and white image.
        base = np.array(
            np.bitwise_or(
                (layer0 != DEAD) * 0xffffffff,   # DEAD cells are black
                (layer0 == DEAD) * 0xff000000), # ALIVE cells are white
            dtype=np.uint32)
        # Merge the base and overlay images.
        # return layer1
        if layer == 0:
            return Image.fromarray(layer0, mode='RGBA')
        elif layer == 1:
            return Image.fromarray(layer1, mode='RGBA')
        elif layer == 2:
            return Image.fromarray(layer2, mode='RGBA') 
        elif layer == 'base':
            return Image.fromarray(base, mode='RGBA')


    frames = [make_frame(frame_data) for frame_data in phenotype]
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=10)


def visualize_all_layers_last_timestep(phenotype, filename):
    def make_frame(frame_data):
        layer0, layer1, layer2 = frame_data
        l, w = layer0.shape
        base = np.array(
            np.bitwise_or(
                (layer0 != DEAD) * 0xffffffff,   # DEAD cells are black
                (layer0 == DEAD) * 0xff000000), # ALIVE cells are white
            dtype=np.uint32)

        # Calculate the total width for the new image
        total_width = w * 4

        # Create a new image with the calculated total width
        combined_image = Image.new('RGBA', (total_width, l))

        # Convert layers to images
        image0 = Image.fromarray(base, mode='RGBA')
        image1 = Image.fromarray(layer0, mode='RGBA')
        image2 = Image.fromarray(layer1, mode='RGBA')
        image3 = Image.fromarray(layer2, mode='RGBA')
        
        # Paste each image side by side in the combined image
        combined_image.paste(image0, (0, 0))
        combined_image.paste(image1, (w, 0))
        combined_image.paste(image2, (w * 2, 0))
        combined_image.paste(image3, (w * 3, 0))

        return combined_image

    frame = make_frame(phenotype[-1]) 
    
    frame.save(filename)



if __name__ == '__main__':
    with open('./experiments/exp_square_4layer_b1/square_l4_b1/square_l4_b1_t3.pkl', 'rb') as pf:
        afpo = pickle.load(pf)

    # print(exp.shape)
    # print(exp.get_target_shape())

    # exp_best_phenotype = simulate_one_individual(exp.best_solution())
    # visualize_all_layers(exp_best_phenotype, 'control_best_all_layers_1.gif')
    fitnesses = [sol.fitness for sol in afpo.population]
    # plt.hist(fitnesses)
    solution = afpo.best_solution()
    # solution = [sol for sol in afpo.population if sol.fitness != None][6]
    exp_best_phenotype = simulate_one_individual(afpo, solution)
    
    # plt.imshow(afpo.get_target_shape())

    target_shape = afpo.get_target_shape()
    print(np.sum(np.abs(target_shape - (exp_best_phenotype[afpo.base_layer][-1] > 0))))
    print(solution.fitness)

    print(exp_best_phenotype[0][0][30:35, 30:35])
    print(exp_best_phenotype[1][0][30:35, 30:35])

    print(exp_best_phenotype[0][1][30:35, 30:35])
    print(exp_best_phenotype[1][1][30:35, 30:35])

    # save_folder = '/'.join(args.exp.split('/')[:-2]) + '/vis'
    # file_name = args.exp.split('/')[-1]
    # os.makedirs(f'{save_folder}', exist_ok=True)

    visualize_all_layers(exp_best_phenotype, './experiments/afpo_test/afpo_test.gif', base_layer_idx=afpo.base_layer)
