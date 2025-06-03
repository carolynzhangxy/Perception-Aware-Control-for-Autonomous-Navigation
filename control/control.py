from copy import deepcopy

from utils.simple_environment import SimpleEnvironment

# from tensorflow import keras, make_tensor_proto, make_ndarray

import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache

def main(params):
    """
    Perform function calls to execute simulation.
    :param params: Dictionary with the simulation parameters.
    :return: None
    """

    trajectory, init_x, init_y, diff = waypoint_tracking(params)

    # trajectory, init_x, init_y, diff = pursue_objective(params)
    # if params['use_learning']:
    #     drawing = trajectory
    #     print('no need for new drawing')
    # else:
    #     environment = create_environment(params)
    #     drawing = environment.draw_trajectory(trajectory)


def pursue_objective(params):
    """
    Test function.
    :param params:
    :return:
    """

    # Parameter retrieval
    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    obj_pos = params['objective_coord']
    stop_difference = params['stop_difference']
    use_learning = params['use_learning']
    preprocessing = params['preprocessing']

    # Constants
    obs_safe_distance = obs_diameter // 2

    # Convenience Functions
    def calculate_difference(pos):
        return np.sqrt((pos[0] - obj_pos[0]) ** 2 + (pos[1] - obj_pos[1]) ** 2)

    def get_prediction(pred):
        return list(make_ndarray(make_tensor_proto(pred))[0])

    # Set agent's initial position (coordinates)
    init_x = -40  # -28  # np.random.randint(low=-width//2, high=width//2)
    init_y = 0  # -11  # np.random.randint(low=-height//2, high=height//2)

    # Initialize the state at the agent's initial position
    state = []
    if use_learning:
        environment = create_environment(params)
        # model = load_model(params)
        model = load_cnn_model()

        drawing = environment.draw_trajectory([[init_x, init_y]])
        state.append(drawing[0])

        # plt.imshow(state[-1])
        # plt.title('INIT: ({}, {})'.format(init_x, init_y))
        # plt.show()
    else:
        state.append([init_x, init_y])

    # Controller parameters
    i = 0
    a = 0.95
    b = 1 - 0.95
    dt = 0.3

    x, y = init_x, init_y
    while calculate_difference([x, y]) > stop_difference:
        i += 1

        # Get Previous state
        if use_learning:
            prev_state = preprocessing(state[-1])
            predicted_state = model(prev_state)
            print('iteration', i)
            print('pred state', predicted_state)
            x_t, y_t = get_prediction(predicted_state)
        else:
            x_t, y_t = state[-1]

        # Controller
        mx = 1 if x_t > 1 else -1
        my = 1 if y_t > 1 else -1

        obj_x = (x_t - obj_pos[0])
        obs_x = (obs_diameter + obs_safe_distance - mx * (x_t - obs_pos[0]))

        obj_y = (y_t - obj_pos[1])
        obs_y = (obs_diameter + obs_safe_distance - my * (y_t - obs_pos[1]))

        ref = 1e-1
        if len(state) > 3:
            if use_learning:
                if np.allclose(state[-1], state[-3], rtol=ref):
                    a = a * 1.05
                    b = 1 - a
                    print('inside img state')
            else:
                if abs(state[-1][0] - state[-2][0]) < ref and abs(state[-1][1] == state[-2][1]) < ref:
                    a = a * 1.05
                    b = 1 - a
                    print('inside num state')

        ux = -a * obj_x + b * obs_x
        uy = -a * obj_y + b * obs_y

        x = x_t + dt * ux
        y = y_t + dt * uy

        if use_learning:
            print('new state ({}, {})'.format(x, y))
            drawing = environment.draw_trajectory([[x, y]])
            state.append(drawing[0])

            plt.imshow(state[-1])
            plt.title('(x, y) = ({:.2f}, {:.2f}) | pred = ({:.2f}, {:.2f})'.format(x, y, x_t, y_t))
            plt.show()
        else:
            state.append([x, y])

    print('Stopped in iteration {}'.format(i))
    diff = calculate_difference(state[-1])

    return state, init_x, init_y, diff


def waypoint_tracking(params):
    """
    Simulate a follower agent tracking a reference agent using the Vision-Based controller.
    :param params: Dictionary with simulation parameters.
    :return: Trajectory of the agents, initial positions, and Euclidean distance to goal when simulation is stopped.
    """

    # Parameter retrieval
    width, height = params['env_dimension']
    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    objective_pos = params['objective_coord']
    stop_difference = params['stop_difference']
    preprocessing = params['preprocessing']

    # Convenience Functions
    def calculate_difference(params):

        pos = params['state'][-1]
        obj = params['obj_pos']

        return np.sqrt((pos[0] - obj[0]) ** 2 + (pos[1] - obj[1]) ** 2)

    def get_prediction(pred):
        return list(make_ndarray(make_tensor_proto(pred))[0])

    # Set agent's initial position (coordinates)
    init_x = -10    # np.random.randint(low=-width//2, high=width//2)
    init_y = 26      # np.random.randint(low=-height//2, high=height//2)

    # Initialize the state at the agent's initial position
    environment = create_environment(params)
    # model = load_model(params)
    model = load_cnn_model()

    # Controller parameters
    i = 0
    reference_params = dict(
        a=0.95, b=1 -0.95, dt=0.1,
        state=[[init_x, init_y]],
        obj_pos=objective_pos,
        obstacle_coord=obs_pos,
        obstacle_diameter=obs_diameter,
        color=(1.0, 1.0, 1.0)
    )
    tracker_params = deepcopy(reference_params)
    tracker_params['a'] = 1.0  #0.45
    tracker_params['b'] = 1 - tracker_params['a']
    tracker_params['color'] = (0.0, 1.0, 0.0)

    reference_params['obj_pos'] = objective_pos

    trajectory = []
    while calculate_difference(reference_params) > stop_difference:
        i += 1
        if i > 300:
            break

        d = tracker_params['state'][-1]
        drawing = environment.draw_trajectory([d], color=tracker_params['color'])[0]
        d2 = reference_params['state'][-1]
        drawing2 = environment.draw_trajectory([d2])[0]

        d_comb = np.clip(drawing + drawing2, 0.0, 1.0)
        trajectory.append(d_comb)

        # Get Previous state
        x_t, y_t = reference_params['state'][-1]

        ##################
        drawing = environment.draw_trajectory([[x_t, y_t]])  # Take picture of environment to predict objective's state.
        prev_state = preprocessing(drawing[0])
        predicted_state = model(prev_state)
        predicted_state = get_prediction(predicted_state)
        tracker_params['obj_pos'] = predicted_state
        ##################

        # Move Tracker
        tracker_params = controller(tracker_params)

        # Move Reference
        reference_params = controller(reference_params)

    print('Waypoint Tracking stopped in iteration {}'.format(i))
    diff = calculate_difference(reference_params)
    plot_trajectory_normal_controller(reference_params, tracker_params)

    return trajectory, init_x, init_y, diff


def controller(params):
    """
    Implementation of the gradient-based controller.
    :param params: Dictionary of simulation parameters.
    :return: The updated parameters dictionary.
    """

    state = params['state']
    current_pos = state[-1]
    obj_pos = params['obj_pos']
    if isinstance(obj_pos[0], list):
        obj_pos = obj_pos[-1]

    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    obs_safe_distance = obs_diameter // 2

    # Controller
    mx = 1 if current_pos[0] > 1 else -1
    my = 1 if current_pos[1] > 1 else -1

    obj_x = (current_pos[0] - obj_pos[0])
    obs_x = (obs_diameter + obs_safe_distance - mx * (current_pos[0] - obs_pos[0]))

    obj_y = (current_pos[1] - obj_pos[1])
    obs_y = (obs_diameter + obs_safe_distance - my * (current_pos[1] - obs_pos[1]))

    ref = 1e-1
    if len(state) > 2:
        if abs(state[-1][0] - state[-2][0]) < ref and abs(state[-1][1] == state[-2][1]) < ref:
            params['a'] = params['a'] * 1.05
            params['b'] = 1 - params['a']

    a = params['a']
    b = params['b']
    dt = params['dt']

    ux = -a * obj_x + b * obs_x
    uy = -a * obj_y + b * obs_y

    x = current_pos[0] + dt * ux
    y = current_pos[1] + dt * uy

    params['state'].append([x, y])

    return params


def plot_trajectory_normal_controller(reference_params, tracker_params):
    """
    Plots the agents' trajectories.
    :param reference_params: Dictionary of simulation parameters for the reference agent.
    :param tracker_params: Dictionary of simulation parameters for the follower agent.
    :return: None
    """

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Static elements
    obstacle = plt.Circle((0, 0), radius=7.5, ls='-', color='w')
    objective = plt.Circle((40, 0), radius=3.5, color='r')
    ax.add_patch(obstacle)
    ax.add_patch(objective)

    # Labels
    plt.suptitle('Agents\' trajectory')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=0, top=60)
    plt.xticks(ticks=np.arange(0, 100, 10))
    plt.yticks(ticks=np.arange(0, 60, 10))

    x = [xi for xi, _ in reference_params['state']]
    y = [yi for _, yi in reference_params['state']]
    plt.plot(x, y, c='w', label='Leader')

    x = [xi for xi, _ in tracker_params['state']]
    y = [yi for _, yi in tracker_params['state']]
    plt.plot(x, y, c='g', label='Follower', ls='--')

    x0, y0 = reference_params['state'][0]
    ax.text(x0+1, y0-3, r'$({}, {})$'.format(x0, y0), fontsize=8, c='w',
            horizontalalignment='center', verticalalignment='center')
    start = plt.Rectangle((x0-1, y0-1), 2, 2, color='w')
    ax.add_patch(start)

    # Text annotations
    ax.text(0, 0, r'$\mathcal{N}$', fontsize=10,
            horizontalalignment='center', verticalalignment='center')
    ax.text(40, 0, r'$\mathcal{G}$', fontsize=10,
            horizontalalignment='center', verticalalignment='center')

    plt.legend()

    plt.subplots_adjust(
        top=0.94,
        bottom=0.095,
        left=0.08,
        right=0.975,
        hspace=0.4,
        wspace=0.2
    )

    plt.show()
    # plt.savefig('trajectory_init({}, {}).pdf'.format(x0, y0))


def create_environment(params):
    """
    Visual representation of the simulation's environment (using the class SimpleEnvironment).
    :param params: Dictionary of simulation parameters.
    :return: The generated environment.
    """

    width, height = params['env_dimension']
    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    obj_pos = params['objective_coord']
    obj_radius = params['objective_diameter']
    occlusion_rate = params['occlusion_rate']

    environment = SimpleEnvironment(width, height, occlusion_rate)
    environment.add_agent(obs_pos, obs_diameter, shape='circle')
    environment.add_agent(obj_pos, obj_radius, shape='circle', color=(1.0, 0.0, 0.0))

    return environment


def load_model(params):
    """
    Helper to load the perception map used as the noisy sensor giving feedback to the controller.
    :param params: Dictionary of simulation parameters.
    :return: Initialized and loaded vision model, ready to do inference.
    # """


    from ultralytics import YOLO as YOLOv10
    # model_path = "/home/perception/yolov10/runs/detect/train2/weights/best.pt"
    # model_path = "/home/perception/neural_network/yolov10/runs/detect/train59/weights/best.pt"
    model_path = "C:/Users/Perception/Desktop/Obstacle_Avoidance/Obstacle_Avoidance_2024/weights/best.pt"
    model = YOLOv10(model_path)

    # from ultralytics import RTDETR
    # # model_path = "/home/perception/yolov10/runs/detect/train33/weights/rtdetr_best.pt"
    # model_path = "/home/perception/yolov10/runs/detect/train60/weights/rtdetr_best.pt"
    # model = RTDETR(model_path)

    model = getattr(model, 'predict')
    return model

def load_cnn_model(params):
    path = "object_detector_model.pth"
    from detect.object_detect_cli import load_model,preprocess_image_array,detect_object,preprocess_image

    model = load_model(path)
    # return model
    def predict(*args,**kwargs):
        image_path = kwargs["source"]
        # plt.imsave("image.png", image)
        # image = preprocess_image("image.png")
        image = preprocess_image(image_path)

        return [detect_object(model, image)]
    return predict

def load_rtdetr_model(params):
    """
    Helper to load the perception map used as the noisy sensor giving feedback to the controller.
    :param params: Dictionary of simulation parameters.
    :return: Initialized and loaded vision model, ready to do inference.
    # """


    from ultralytics import RTDETR
    # model_path = "/home/perception/yolov10/runs/detect/train33/weights/rtdetr_best.pt"
    # model_path = "/home/perception/neural_network/yolov10/runs/detect/train60/weights/rtdetr_best.pt"
    model_path = "C:/Users/Perception/Desktop/Obstacle_Avoidance/Obstacle_Avoidance_2024/weights/rtdetr_best.pt"
    model = RTDETR(model_path)

    model = getattr(model, 'predict')
    return model



@lru_cache()
def load_occlusion_predictor(name: str):
    from clf.inference import Predector
    pred = Predector(name)
    return pred

def get_best_model(image_path: str):
    pred = load_occlusion_predictor("new")
    res = int(pred.predict(image_path))
    if res < 4:
        return "yolo"
    return "rtdetr"

# def load_model(params):
#     """
#     Helper to load the perception map used as the noisy sensor giving feedback to the controller.
#     :param params: Dictionary of simulation parameters.
#     :return: Initialized and loaded vision model, ready to do inference.
#     """

#     # model_path = params['model_path']
#     model_path = "/home/perception/neural_networkyolov10/runs/detect/train2/weights/best.pt"
#     import torch


#     from ultralytics import YOLOv10
#     # load pytorch model
#     # model = torch.hub.load('THU-MIG/yolov10', 'custom', path=model_path, source='local')  # local repo
#     model = YOLOv10()
#     mode = "predict"
#     model = getattr(model, mode)

    # return modelg


if __name__ == '__main__':

    # MODEL
    # x(t + 1) = x(t) + s * ux
    # y(t + 1) = y(t) + s * uy
    # ux = -a(x - xt) + b(x - 1);
    # uy = -a(y - yt) + b(y - 2);

    def img_preprocessing(img):
        """
        Helper function to preprocess images following the perception map's preproccessing for training data.
        :param img: Matrix (image) to be preprocessed.
        :return: Preprocessed image.
        """

        img = img.astype(np.uint8).astype('float32')
        img = np.expand_dims(img, axis=0)

        return img

    # Define simulation parameters.
    sim_params = dict(
        env_dimension=(100, 60),               # Environment dimensions (width, height).
        obstacle_coord=(50, 30),                 # (x, y)-coordinate where to place the obstacle.
        obstacle_diameter=15,                  # Diameter of the ball representing the obstacle.
        objective_coord=(90, 30),               # (x, y)-coordinate where the agent's goal is located.
        objective_diameter=7,                  # Diameter of the ball representing the goal.
        stop_difference=0.5,                   # Minimum difference between agent and goal to stop simulation.
        occlusion_rate=0.9,                    # Rate (0 <= r <= 1) at which random occlusions are added to the image.
        preprocessing=img_preprocessing,       # Function to preprocess images before feeding them to the vision model.
        model_path='/home/perception/neural_network/yolov10/runs/detect/train2/weights/best.pt',    # 'cnn_100kTrain_100%_OcclusionRate'),
        # use_learning=False
    )

    # Simulate the controller
    for i in range(1):
        main(sim_params)