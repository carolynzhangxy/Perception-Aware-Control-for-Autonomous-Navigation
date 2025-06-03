from copy import deepcopy

from control import load_model, create_environment  # , controller

# from tensorflow import make_tensor_proto, make_ndarray

import matplotlib.pyplot as plt
import numpy as np


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
    #
    # build_gif(trajectory, 'no_pred_init({:.1f}, {:.1f})_diff_{:.2f}.gif'.format(init_x, init_y, diff))


def cost_function(current_pos, obj_pos=(40, 0), obs_pos=(0, 0), obs_diameter=16):
    """
    Objective function (to minimize) defining the cost to reach the goal while doing obstacle avoidance.
    :param current_pos: (x, y)-coordinate of the agent's current position.
    :param obj_pos: (x, y)-coordinate of the goal.
    :param obs_pos: (x, y)-coordinate of the obstacle.
    :param obs_diameter: Diameter of the obstacle.
    :return: Cost to reach the goal (scalar).
    """

    obs_radius = 0.5 * obs_diameter
    x = -((current_pos[0] - obj_pos[0]) ** 2)
    y = -((current_pos[1] - obj_pos[1]) ** 2)

    if (current_pos[0] - obs_pos[0])**2 + (current_pos[1] - obs_pos[1])**2 > obs_radius**2:
        dist = np.sqrt((current_pos[0] - obs_pos[0]) ** 2 + (current_pos[1] - obs_pos[1]) ** 2) - obs_radius
    else:
        dist = 0

    if dist <= obs_radius:
        b = (np.log(obs_radius) - np.log(dist)) * ((dist - obs_radius) ** 2)
    else:
        b = 0

    return abs(x + y - b)


def waypoint_tracking(params):
    """
    Simulate a follower agent tracking a reference agent using the Vision-Based controller.
    :param params: Dictionary with simulation parameters.
    :return: Trajectory of the agents, initial positions, and Euclidean distance to goal when simulation is stopped.
    """

    # Parameter retrieval
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
    init_x = -14  # -9.53   # -8.00000001    # np.random.randint(low=-width//2, high=width//2)
    init_y = 1  # 0.1      # np.random.randint(low=-height//2, high=height//2)

    # Initialize the state at the agent's initial position
    environment = create_environment(params)
    model = load_model(params)

    # Controller parameters
    i = 0
    reference_params = dict(
        dt=0.1,
        state=[[init_x, init_y]],
        obj_pos=objective_pos,
        obstacle_coord=obs_pos,
        obstacle_diameter=obs_diameter,
        color=(1.0, 1.0, 1.0)
    )
    tracker_params = deepcopy(reference_params)
    tracker_params['color'] = (0.0, 1.0, 0.0)

    trajectory = []
    while calculate_difference(reference_params) > stop_difference:
        i += 1
        if i > 500:
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
        # tracker_params = controller2(tracker_params)

        # Move Reference
        reference_params = controller2(reference_params)

    print('Waypoint Tracking stopped in iteration {}'.format(i))
    diff = calculate_difference(reference_params)
    plot_trajectory_normal_controller(reference_params, tracker_params)

    return trajectory, init_x, init_y, diff


def controller2(params):
    """
    Implementation of the gradient-based controller.
    :param params: Dictionary of simulation parameters.
    :return: The updated parameters dictionary.
    """

    state = params['state']
    current_pos = state[-1]
    obj_pos = params['obj_pos']
    # if isinstance(obj_pos, list):
    #     obj_pos = obj_pos[-1]

    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']

    def get_jacobian():

        h = 0.01
        cost = cost_function(current_pos.copy(), obj_pos, obs_pos, obs_diameter)

        jacobian = []
        for i in range(len(current_pos)):

            mod_params = current_pos.copy()
            mod_params[i] = mod_params[i] + h

            cost_h = cost_function(mod_params, obj_pos, obs_pos, obs_diameter)

            partial_derivative = (cost_h - cost) / h

            jacobian.append(partial_derivative)

        jacobian = np.array(jacobian)
        jacobian = np.clip(jacobian, -10, 10)  # Gradient clipping

        return jacobian

    dt = params['dt']
    u = -get_jacobian()

    # ########################## ADVERSARIAL NOISE #####################################################################
    # #####  NOTE: dt = 0.1
    # saddle_point = [-13.8, 0.8]
    # delta_x = 1.5
    # delta_y = 1.5
    # delta_x = delta_x if abs(current_pos[0] - saddle_point[0]) <= delta_x else 0.0
    # delta_y = delta_y if abs(current_pos[1] - saddle_point[1]) <= delta_y else 0.0
    #
    # eps = 4.2
    # # print(delta_x, delta_y, u, eps * delta_x * (current_pos[0] - saddle_point[0]), eps * delta_y * (current_pos[1] - saddle_point[1]))
    #
    # x = current_pos[0] + dt * (u[0] - eps * delta_x * (current_pos[0] - saddle_point[0]))
    # y = current_pos[1] + dt * (u[1] - eps * delta_y * (current_pos[1] - saddle_point[1]))
    # ##################################################################################################################

    x = current_pos[0] + dt * u[0]        # Comment this out when simulating adversarial noise
    y = current_pos[1] + dt * u[1]        # Comment this out when simulating adversarial noise

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
    plt.suptitle('Agents\' trajectory', fontsize=26)
    plt.xlabel('x-coordinate', fontsize=16)
    plt.ylabel('y-coordinate', fontsize=16)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=-50, right=50)
    ax.set_ylim(bottom=-30, top=30)
    plt.xticks(ticks=np.arange(-50, 50, 10))
    plt.yticks(ticks=np.arange(-30, 30, 10))

    x = [xi for xi, _ in reference_params['state']]
    y = [yi for _, yi in reference_params['state']]
    plt.plot(x, y, ls='-', lw=2, c='w', label='Leader')

    x = [xi for xi, _ in tracker_params['state']]
    y = [yi for _, yi in tracker_params['state']]
    plt.plot(x, y, c='lime', label='Follower', ls='--', lw=2)

    x0, y0 = reference_params['state'][0]
    ax.text(x0+1, y0-3, r'$({}, {})$'.format(x0, y0), fontsize=8, c='w',
            horizontalalignment='center', verticalalignment='center')
    start = plt.Rectangle((x0-1, y0-1), 2, 2, color='w')
    ax.add_patch(start)

    # Text annotations
    ax.text(0, 0, r'$\mathcal{N}$', fontsize=20,
            horizontalalignment='center', verticalalignment='center')
    ax.text(40, 0, r'$\mathcal{G}$', fontsize=20,
            horizontalalignment='center', verticalalignment='center')

    plt.legend()

    plt.subplots_adjust(
        top=0.91,
        bottom=0.1,
        left=0.085,
        right=0.975,
        hspace=0.395,
        wspace=0.2
    )

    plt.show()
    # plt.savefig('trajectory_init({}, {}).pdf'.format(x0, y0))


def draw_contour_plot(cost_point=(40, 0), agents=None, save_path=None):
    """
    Draw the level sets of the cost function on the state space.
    :param cost_point: (x, y)-coordinate where the goal is located.
    :param agents: List of (coord, color, radius, label)-tuples describing agents in the environment.
    :param save_path: Path to save the generated figure.
    :return: None
    """

    if save_path is None:
        fig = plt.figure(figsize=(60, 100))
    else:
        fig = plt.figure(figsize=(25, 13))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    sampling = 5
    x = np.linspace(-50, 50, sampling*100)
    y = np.linspace(30, -30, sampling*60)

    mesh_shape = (len(y), len(x))
    mesh = np.zeros(mesh_shape)

    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            mesh[j, i] = cost_function(current_pos=(x_i, y_j), obj_pos=cost_point)

    plt.contour(x, y, mesh, cmap='Wistia', levels=48)

    # Static elements
    obstacle = plt.Circle((0, 0), radius=7.5, ls='-', color='w')
    objective = plt.Circle((40, 0), radius=3.5, color='r')
    ax.add_patch(obstacle)
    ax.add_patch(objective)

    if agents is not None and isinstance(agents, list):
        added_labels = set()
        for coord, color, radius, label in agents:
            x = coord[0] - (radius / 2)
            y = coord[1] - (radius / 2)
            if label not in added_labels:
                ag = plt.Rectangle((x,y), radius, radius, color=color, label=label)
            else:
                ag = plt.Rectangle((x, y), radius, radius, color=color)
            added_labels.add(label)
            ax.add_patch(ag)
        plt.legend(fontsize=16)

    # Text annotations
    ax.text(0, 0, r'$\mathcal{N}$', fontsize=26,
            horizontalalignment='center', verticalalignment='center')
    ax.text(40, 0, r'$\mathcal{G}$', fontsize=26,
            horizontalalignment='center', verticalalignment='center')

    # Labels
    plt.title('Level sets', fontsize=20)
    plt.xlabel('x-coordinate', fontsize=16)
    plt.ylabel('y-coordinate', fontsize=16)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=-50, right=50)
    ax.set_ylim(bottom=-30, top=30)
    plt.xticks(ticks=np.arange(-50, 50, 10))
    plt.yticks(ticks=np.arange(-30, 30, 10))
    fig.subplots_adjust(
        top=0.95,
        bottom=0.05,
        left=0.035,
        right=1.0,
        hspace=0.2,
        wspace=0.2
    )
    cb = plt.colorbar()
    cb.lines[0].set_linewidth(10)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, orientation='landscape')


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
        obstacle_coord=(0, 0),                 # (x, y)-coordinate where to place the obstacle.
        obstacle_diameter=16,                  # Diameter of the ball representing the obstacle.
        objective_coord=(40, 0),               # (x, y)-coordinate where the agent's goal is located.
        objective_diameter=7,                  # Diameter of the ball representing the goal.
        stop_difference=1.0,                   # Minimum difference between agent and goal to stop simulation.
        occlusion_rate=0.0,                    # Rate (0 <= r <= 1) at which random occlusions are added to the image.
        preprocessing=img_preprocessing,       # Function to preprocess images before feeding them to the vision model.
        model_path='../saved_models/cnn_100kTrain',           # cnn_100kTrain_100%_OcclusionRate'
        # use_learning=False
    )

    # Simulate the controller
    for i in range(1):
        main(sim_params)

    # Plot the level sets of the cost function.
    # draw_contour_plot()