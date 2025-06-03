import sys
import os
import time
from array2gif import write_gif

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from PIL import Image
from utils.simple_environment import SimpleEnvironment, SimpleObstacle  # Assuming utils is the correct path

def simple_environment(width, height, occlusion_rate):
    environment = SimpleEnvironment(width, height, occlusion_rate)

    #Original simple environment
    environment.add_agent((50, 30 ), 15, shape='circle')  # obstacle
    environment.add_agent((90, 30 ), 7, shape='circle', color=(1.0, 0.0, 0.0))  # objective point

    return environment

def environment_multiple_obstacles(width, height, occlusion_rate):

    environment = SimpleEnvironment(100, 60, occlusion_rate=0.0)

    obs_shape = 'circle'
    obstacles = [
          SimpleObstacle(coord=(14, 16),  diameter=11, shape=obs_shape, color=(0.0, 0.75, 1.0))
        , SimpleObstacle(coord=(50, 20), diameter=9, shape=obs_shape, color=(0.6, 0.6, 0.5))
        , SimpleObstacle(coord=(17, 30),  diameter=15, shape=obs_shape, color=(1.0, 1.0, 0.0))
        , SimpleObstacle(coord=(36, 50),   diameter=13, shape=obs_shape, color=(1.0, 0.0, 1.0))
        , SimpleObstacle(coord=(66, 48), diameter=11, shape=obs_shape, color=(1.0, 0.5, 0.5))
        , SimpleObstacle(coord=(84, 4),   diameter=9, shape=obs_shape, color=(1.0, 1.0, 0.5))
        , SimpleObstacle(coord=(23, 6),   diameter=11, shape=obs_shape, color=(0.7, 0.4, 0.0))
        , SimpleObstacle(coord=(66, 16),    diameter=13, shape=obs_shape, color=(0.0, 0.3, 0.8))
        , SimpleObstacle(coord=(34, 36),  diameter=17, shape=obs_shape, color=(0.3, 0.5, 0.5))
        , SimpleObstacle(coord=(88, 50),   diameter=13, shape=obs_shape, color=(0.2, 0.1, 0.5))
        , SimpleObstacle(coord=(80, 4),   diameter=15, shape=obs_shape, color=(0.4, 0.2, 0.6))
        , SimpleObstacle(coord=(70, 34),  diameter=9, shape=obs_shape, color=(0.5, 0.1, 0.5))
    ]

    for obstacle in obstacles:
        environment.add_agent_object(obstacle)
        

    environment.add_agent((90, 30 ), 7, shape='star', color=(1.0, 0.0, 0.0))  # Goal point

    return environment



def generate_data(generation_params):

    width, height = generation_params['width'], generation_params['height']
    agent_radius = generation_params['agent_diameter']
    num_samples = generation_params['num_samples']
    occlusion_rate = generation_params['occlusion_rate']
    data_name = generation_params['dataset_name']

    ############################### RANDOM POSITION FOR ALL ELEMENTS #####################################################
    # Save data to Excel with images' labels
    if not os.path.exists(data_name):
        os.makedirs(data_name)

    # entries = []
    # for i in range(num_samples):
    #     # Initialize Environment and Add agent
    #     environment = simple_environment(width, height, occlusion_rate)

    #     # Generate random coordinates for obstacle and objective point
    #     obstacle_x = np.random.randint(0, width)
    #     obstacle_y = np.random.randint(0, height)
    #     objective_x = np.random.randint(0, width)
    #     objective_y = np.random.randint(0, height)

    #     # Ensure the coordinates are within the bounds and avoid placing them on the same spot
    #     while (obstacle_x, obstacle_y) == (objective_x, objective_y):
    #         objective_x = np.random.randint(0, width)
    #         objective_y = np.random.randint(0, height)

    #     environment.add_agent((obstacle_x, obstacle_y), 16, shape='circle')  # obstacle
    #     environment.add_agent((objective_x, objective_y), 8, shape='star', color=(1.0, 0.0, 0.0))  # objective point

    #     # Create random trajectory
    #     samples_x = np.random.randint(low=0, high=width)
    #     samples_y = np.random.randint(low=0, high=height)
    #     trajectory = [(samples_x, samples_y)]

    #     # Draw trajectory images
    #     trajectory_images = environment.draw_trajectory(trajectory, agent_radius)

    #     img_name = f'{data_name}/img{i}.png'
    #     im = Image.fromarray((trajectory_images[0] * 255).astype(np.uint8))
    #     im.save(img_name)

    #     entries.append([img_name, (samples_x, samples_y), (obstacle_x, obstacle_y), (objective_x, objective_y)])

    # df = pd.DataFrame(entries, columns=['Path', 'Agent Coordinate', 'Obstacle Coordinate', 'Objective Coordinate'])
    # df.to_excel(f'{data_name}.xlsx')

    ######################################################################################################################
    ############################# ORIGINAL DATASET #######################################################################
    # Initialize Environment and Add agent
    environment = simple_environment(width, height, occlusion_rate)
    # environment = environment_multiple_obstacles(width, height, occlusion_rate)

    
    # Create random trajectory
    samples_x = np.random.randint(low=3, high=width-3, size=num_samples)
    samples_y = np.random.randint(low=3, high=height-3, size=num_samples)
    # print(samples_x)
    # print(samples_y)
    trajectory = list(zip(samples_x, samples_y))

    # Format coordinates with decimal places and save as string
    # entries.append([img_name, formatted_coord])

    # Draw trajectory images
    trajectory_images = environment.draw_trajectory(trajectory, agent_radius)

    # Save data to Excel with images' labels
    if not os.path.exists(data_name):
       os.makedirs(data_name)

    entries = []
    for i, (img, coord) in enumerate(zip(trajectory_images, trajectory)):
       # formatted_coord = f"({coord[0]:.4f}, {coord[1]:.4f})"
       img_name = f'{data_name}/img{i}.png'
       im = Image.fromarray((img * 255).astype(np.uint8))
       im.save(img_name)
       entries.append([img_name, coord])

    df = pd.DataFrame(entries, columns=['Path', 'Agent Coordinate'])
    df.to_excel(f'{data_name}.xlsx')

    #######################################################################################################################

def build_gif(sources, name, fps=6):
    """
    Build a gif.
    :param sources: List of images used to create gif.
    :param name: Gif file name.
    :param fps: Frames-per-second of the gif animation.
    :return: None
    """

    joined = sources
    joined = 255.0 * np.array(joined).squeeze()

    write_gif(joined, name, fps=fps)


if __name__ == '__main__':
    for i in range(1):
        gen_params = dict(
            width=100,
            height=60,
            agent_diameter=5,
            occlusion_rate=0,
            num_samples=1000,
            dataset_name='train0'
        )


generate_data(gen_params)
