import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import moviepy.video.io.ImageSequenceClip

from PIL import Image
from tensorflow import make_tensor_proto, make_ndarray

from control import load_model
from hybrid_control import Fringe, hybrid_controller, cost_function
from utils.simple_environment import SimpleEnvironment, SimpleObstacle


def waypoint_tracking(params, coord=None):
    """
    Simulate a follower agent tracking a reference agent using the Vision-Based hybrid controller.
    :param params: Dictionary with simulation parameters.
    :param coord: Optional. Coordinate where the robots are initialized at.
    :return: Trajectory of the agents, initial positions, and Euclidean distance to goal when simulation is stopped.
    """

    # Parameter retrieval
    objective_pos = params['objective_coord']
    stop_difference = params['stop_difference']
    preprocessing = params['preprocessing']

    # Convenience Functions
    def calculate_difference(params):

        pos = params['state'][-1]
        obj = params['obj_pos']
        if isinstance(obj, list):
            obj = obj[-1]

        return np.sqrt((pos[0] - obj[0]) ** 2 + (pos[1] - obj[1]) ** 2)

    def get_prediction(pred):
        x, y = list(make_ndarray(make_tensor_proto(pred))[0])
        x, y = np.clip(x, -100, 100), np.clip(y, -50, 50)
        return x, y

    # Set agent's initial position (coordinates)
    if coord is None:
        init_x = -46  # np.random.randint(low=-width//2, high=0)  #width//2)
        init_y = 25  # np.random.randint(low=-height//2, high=height//2)
    else:
        init_x, init_y = coord

    # Initialize the state at the agent's initial position
    environment = environment_multiple_obstacles()
    model = load_model(params)

    # Controller parameters
    reference_params = dict(
        dt=0.05, chi=2.5, lamb=0.5,
        current_controller=1,
        controller_history=[1],
        state=[[init_x, init_y]],
        obj_pos=[objective_pos],
        color=(1.0, 1.0, 1.0),
        id='reference'
    )

    tracker_params = deepcopy(reference_params)
    tracker_params['dt'] = 0.1
    tracker_params['color'] = (0.0, 1.0, 0.0)
    tracker_params['obj_pos'] = [reference_params['state'][0]]
    tracker_params['id'] = 'tracker'

    i = 0
    trajectory = []
    from pathlib import Path
    import shutil
    gif_folder = Path('frames-v2')
    shutil.rmtree(gif_folder, ignore_errors=True)
    gif_folder.mkdir(exist_ok=True)
    while calculate_difference(reference_params) > stop_difference:
        i += 1
        if i > 2000:
            break

        # Draw simulation
        d = tracker_params['state'][-1]    # reference_params['state'][-1]
        drawing = environment.draw_trajectory([d], color=tracker_params['color'], ignore_occlusion=True)[0]
        d2 = reference_params['state'][-1]
        drawing2 = environment.draw_trajectory([d2])[0]

        d_comb = np.clip(drawing + drawing2, 0.0, 1.0)
        trajectory.append(d_comb)

        # Get Previous state
        x_t, y_t = reference_params['state'][-1]

        # Predict State of Reference robot
        drawing = environment.draw_trajectory([[x_t, y_t]])  # Take picture of environment to predict objective's state.
        prev_state = preprocessing(drawing[0])
        predicted_state = model(prev_state)
        # predicted_state = get_prediction(predicted_state)
        img = preprocessing(drawing[0])
        img_path = gif_folder / f'img{i}.png'
        plt.imsave(img_path, img)
        overrides = {
            "conf": 0.25,
            # "source": "/home/perception/yolov10/datasets/test/images/img862.png"
            "source"  : img_path,
            # 关闭日志输出
            #"verbose": False
        } 
        # predicted_state = model(prev_state, **overrides)  
        next_res = model( **overrides)[0]  
        print(next_res.boxes)
        if next_res.boxes.xywh.shape[0] > 0:
            predicted_state = next_res.boxes.xywh[0][:2]

        # Move Tracker
        tracker_params['fringe_O1'], tracker_params['fringe_O2'], obs = get_fringe(predicted_state)
        tracker_params['obstacle_diameter'] = obs.diameter
        tracker_params['obj_pos'].append(predicted_state)
        tracker_params = hybrid_controller(tracker_params)

        # Move Reference
        reference_params['fringe_O1'], reference_params['fringe_O2'], obs = get_fringe(reference_params['state'][-1])
        reference_params['obstacle_diameter'] = obs.diameter
        reference_params = hybrid_controller(reference_params)

    diff = calculate_difference(reference_params)
    print('Waypoint Tracking stopped in iteration {}. Difference = {}'.format(i, diff))

    plot_trajectory_hc(reference_params, tracker_params)
    # video_contour_plot(reference_params, tracker_params)
    # plot_prediction_vs_label(reference_params, tracker_params)

    return trajectory, init_x, init_y, diff


def get_fringe(state):

    xt, yt = state
    obstacles = get_obstacles()

    if xt <= -70:
        obstacle = obstacles[0]
    elif xt <= -55:
        if yt <= 0:
            obstacle = obstacles[1]
        else:
            obstacle = obstacles[2]
    elif xt <= -25:
        if yt <= 15:
            obstacle = obstacles[3]
        else:
            obstacle = obstacles[2]
    elif xt <= 2:
        if yt <= -15:
            obstacle = obstacles[4]
        else:
            obstacle = obstacles[5]
    elif xt <= 25:
        obstacle = obstacles[6]
    elif xt <= 55:
        if yt <= 0:
            obstacle = obstacles[8]
        else:
            obstacle = obstacles[7]
    else:
        if yt <= -25:
            obstacle = obstacles[11]
        elif yt <= 10:
            obstacle = obstacles[10]
        else:
            obstacle = obstacles[9]

    obs_pos = obstacle.coord
    obs_radius = (obstacle.diameter // 2) + (obstacle.diameter // 4) + 1
    p0 = (obs_pos[0] - obs_radius, obs_pos[1])
    p2 = (obs_pos[0] + obs_radius, obs_pos[1])
    f1 = Fringe(p0=p0, p1=(obs_pos[0], obs_pos[1] + obs_radius), p2=p2)
    f2 = Fringe(p0=p0, p1=(obs_pos[0], obs_pos[1] - obs_radius), p2=p2)

    return f1, f2, obstacle


def get_obstacles():

    obs_shape = 'circle'
    obstacles = [
          SimpleObstacle(coord=(-84, 16),  diameter=16, shape=obs_shape, color=(0.0, 0.75, 1.0))
        , SimpleObstacle(coord=(-64, -20), diameter=10, shape=obs_shape, color=(0.6, 0.6, 0.5))
        , SimpleObstacle(coord=(-50, 34),  diameter=18, shape=obs_shape, color=(1.0, 1.0, 0.0))
        , SimpleObstacle(coord=(-36, 0),   diameter=12, shape=obs_shape, color=(1.0, 0.0, 1.0))
        , SimpleObstacle(coord=(-16, -28), diameter=12, shape=obs_shape, color=(1.0, 0.5, 0.5))
        , SimpleObstacle(coord=(-4, 24),   diameter=10, shape=obs_shape, color=(1.0, 1.0, 0.5))
        , SimpleObstacle(coord=(14, -6),   diameter=18, shape=obs_shape, color=(0.7, 0.4, 0.0))
        , SimpleObstacle(coord=(36, 16),    diameter=8, shape=obs_shape, color=(0.0, 0.3, 0.8))
        , SimpleObstacle(coord=(34, -36),  diameter=16, shape=obs_shape, color=(0.3, 0.5, 0.5))
        , SimpleObstacle(coord=(74, 30),   diameter=14, shape=obs_shape, color=(0.2, 0.1, 0.5))
        , SimpleObstacle(coord=(70, -4),   diameter=16, shape=obs_shape, color=(0.4, 0.2, 0.6))
        , SimpleObstacle(coord=(70, -34),  diameter=10, shape=obs_shape, color=(0.5, 0.1, 0.5))
    ]

    return obstacles


def environment_multiple_obstacles():

    environment = SimpleEnvironment(200, 100, occlusion_rate=0.0)

    for obstacle in get_obstacles():
        environment.add_agent_object(obstacle)

    environment.add_agent((90, 40), 12, color=(1.0, 0.0, 0.0))  # Goal point

    # plt.imshow(environment.get_field_copy())
    # plt.show()
    return environment


def plot_trajectory_hc(reference_params, tracker_params):
    """
    Plot the agents' trajectories and the switching of the controllers.
    :param reference_params: Dictionary of simulation parameters for the reference agent.
    :param tracker_params: Dictionary of simulation parameters for the follower agent.
    :return: None
    """

    plt.subplots(2, 1, figsize=(7.5, 7), gridspec_kw={'height_ratios': [2.5, 1]})
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax1.set_axisbelow(True)

    # Static elements
    for i, obstacle in enumerate(get_obstacles()):
        if obstacle.shape == 'circle':
            obs = plt.Circle(obstacle.coord, radius=obstacle.diameter/2, ls='-', color=obstacle.color)
        else:
            side = obstacle.diameter/2
            obs = plt.Rectangle(obstacle.coord, side, side, color=obstacle.color)
        ax1.add_patch(obs)

        ax1.text(obstacle.coord[0], obstacle.coord[1], r'$\mathcal{N}_{' + f'{i+1}' + r'}$', fontsize=12, c='gray',
                 horizontalalignment='center', verticalalignment='center')

    objective = plt.Rectangle((86, 36), 8, 8, color='r')
    ax1.add_patch(objective)

    # Labels
    plt.title('Agents\' trajectory', fontsize=26)
    plt.xlabel('x-coordinate', fontsize=16)
    plt.ylabel('y-coordinate', fontsize=16)

    # Plot config.
    ax1.set_facecolor('k')
    ax1.set_xlim(left=-100, right=100)
    ax1.set_ylim(bottom=-50, top=50)
    plt.xticks(ticks=np.arange(-100, 100, 20))
    plt.yticks(ticks=np.arange(-50, 50, 20))

    x = [xi for xi, _ in reference_params['state']]
    y = [yi for _, yi in reference_params['state']]

    plt.plot(x, y, lw=2, ls='-', c='w', label='Leader')

    x = [xi for xi, _ in tracker_params['state']]
    y = [yi for _, yi in tracker_params['state']]
    plt.plot(x, y, c='lime', label='Follower', ls='--', lw=2)

    x0, y0 = reference_params['state'][0]
    ax1.text(x0+1, y0-3, r'$({}, {})$'.format(x0, y0), fontsize=8, c='w',
            horizontalalignment='center', verticalalignment='center')
    start = plt.Rectangle((x0-1, y0-1), 2, 2, color='w')
    ax1.add_patch(start)

    # Text annotations
    ax1.text(90, 40, r'$\mathcal{G}$', fontsize=20, c='w',
            horizontalalignment='center', verticalalignment='center')

    plt.legend(fontsize=16)

    ###########################
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor('k')
    ax2.grid(True, ls=':', lw=0.5, alpha=0.5)

    ax2.plot(reference_params['controller_history'], c='w', lw=2)
    ax2.plot(tracker_params['controller_history'], c='lime', ls='--', lw=2)

    plt.title('Agents\' controller state', fontsize=26)
    plt.xlabel('Timestep', fontsize=16)
    plt.ylabel(r'Logic State $q_i$', fontsize=16)
    plt.yticks([1, 2])

    plt.subplots_adjust(
        top=0.94,
        bottom=0.095,
        left=0.125,
        right=0.955,
        hspace=0.4,
        wspace=0.2
    )

    plt.show()
    # plt.savefig('trajectory_init({}, {}).pdf'.format(x0, y0))


def video_contour_plot(reference_params, tracker_params):
    """
    Generate a video of the change in the level sets of the hybrid controller as the follower agent moves.
    :param reference_params: Dictionary of parameters of the reference agent.
    :param tracker_params: Dictionary of parameters of the follower agent.
    :return: None
    """

    controllers_indices = [1, 2]
    frame_rates = [5, 24]

    ref_state = reference_params['state']
    tracker_state = tracker_params['state']
    tracker_obj = tracker_params['obj_pos']

    folder_name_base = 'frames_FollowerLevelSets_q{}_{}'
    image_folders = [folder_name_base.format(q, ref_state[0]) for q in controllers_indices]
    for folder in image_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    state_range = min(len(ref_state), len(tracker_state))
    for i in range(state_range):

        reference_agent = (ref_state[i], 'w', 2, 'Leader')
        follower_agent = (tracker_state[i], 'lime', 2, 'Follower')
        agents = [reference_agent, follower_agent]

        for q in controllers_indices:
            folder = image_folders[q-1]

            cost_point = tracker_obj[i] if isinstance(tracker_obj, list) else tracker_obj
            if q == 1:
                draw_contour_plot_q1(cost_point, agents, save_path='{}/{}.png'.format(folder, i))
            elif q == 2:
                draw_contour_plot_q2(cost_point, agents, save_path='{}/{}.png'.format(folder, i))

    video_name_base = 'FollowerLevelSets_q{}_{}_{}fps.mp4'
    for q in controllers_indices:
        folder = image_folders[q-1]
        image_files = [folder + '/' + img for img in os.listdir(folder) if img.endswith(".png")]
        image_files = sorted(image_files, key=lambda name: int(name.split('/')[-1].split('.')[0]))

        for fps in frame_rates:
            video_name = video_name_base.format(q, ref_state[0], fps)
            print('Saving:', video_name)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(video_name)


def draw_contour_plot_q1(cost_point=(90, 40), agents=None, save_path=None):
    """
    Generate level sets when q = 1.
    :param cost_point: (x, y)-coordinate of the agent's goal.
    :param agents: List of (coord, color, radius, label)-tuples describing agents in the environment.
    :param save_path: Path to save the generated figure.
    :return: None
    """

    if save_path is None:
        fig = plt.figure(figsize=(100, 200))
    else:
        fig = plt.figure(figsize=(25, 13))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    sampling = 5
    x = np.linspace(-100, 100, sampling*100)
    y = np.linspace(-50, 50, sampling*60)

    mesh_shape = (len(y), len(x))
    mesh1 = np.zeros(mesh_shape)
    mesh2 = np.zeros(mesh_shape)

    fringe1_borders = []
    flow_set = []  # C
    jump_set = []  # D

    chi, lamb = 2.5, 0.5
    fringe1, fringe2, _ = get_fringe(agents[0][0])
    for i, x_i in enumerate(x):
        c, d = None, None
        for j, y_j in enumerate(y):
            mesh1[j, i] = cost_function((x_i, y_j), cost_point, fringe1, 8)
            mesh2[j, i] = cost_function((x_i, y_j), cost_point, fringe2, 8)

            ############# FRINGE 1 IS VALLEY ######################################################################
            # Fill Flow Set (C)
            if c is None and mesh1[j, i] != np.inf and mesh1[j, i] <= chi * mesh2[j, i]:
                c = y_j

            # Fill Jump Set (D)
            if d is None and mesh1[j, i] < (chi - lamb) * mesh2[j, i]:
                d = y_j
            ########################################################################################################

        fringe1_borders.append(fringe1(x_i))
        flow_set.append(c)
        jump_set.append(d)

    plt.contour(x, y, mesh1, cmap='Wistia', levels=16)
    plt.plot(x, fringe1_borders, ls='--', color='silver')

    flow_set_color = 'fuchsia'
    jump_set_color = 'cyan'
    plt.plot(x, flow_set, c=flow_set_color, lw=4)
    plt.plot(x, jump_set, c=jump_set_color, lw=4)

    flow_set_arrows = [70, 110, 150]
    for arrow in flow_set_arrows:
        plt.arrow(x[arrow], flow_set[arrow], -1, -1, color=flow_set_color, head_width=1)
        plt.arrow(x[-arrow], flow_set[-arrow], 1, -1, color=flow_set_color, head_width=1)

    arrow = 60  # 150
    ax.text(x[arrow] + 0.0, jump_set[arrow] - 2.5, r'$\mathit{C}_1$', fontsize=40, c=flow_set_color,
                horizontalalignment='center', verticalalignment='center')
    ax.text(x[-arrow] - 0.0, jump_set[-arrow] - 2.5, r'$\mathit{C}_1$', fontsize=40, c=flow_set_color,
            horizontalalignment='center', verticalalignment='center')

    jump_set_arrows = [50, 90, 130]
    for arrow in jump_set_arrows:
        plt.arrow(x[arrow], jump_set[arrow], 1, 1, color=jump_set_color, head_width=1)
        plt.arrow(x[-arrow], jump_set[-arrow], -1, 1, color=jump_set_color, head_width=1)

    arrow = 90
    ax.text(x[arrow] + 0.6, jump_set[arrow] + 4, r'$\mathit{D}_1$', fontsize=40, c=jump_set_color,
                horizontalalignment='center', verticalalignment='center')
    ax.text(x[-arrow] - 0.6, jump_set[-arrow] + 4, r'$\mathit{D}_1$', fontsize=40, c=jump_set_color,
            horizontalalignment='center', verticalalignment='center')

    # Static elements
    for i, obstacle in enumerate(get_obstacles()):
        if obstacle.shape == 'circle':
            obs = plt.Circle(obstacle.coord, radius=obstacle.diameter/2, ls='-', color=obstacle.color)
        else:
            side = obstacle.diameter/2
            obs = plt.Rectangle(obstacle.coord, side, side, color=obstacle.color)
        ax.add_patch(obs)

        ax.text(obstacle.coord[0], obstacle.coord[1], r'$\mathcal{N}_{' + f'{i+1}' + r'}$', fontsize=30,
                 horizontalalignment='center', verticalalignment='center')

    objective = plt.Rectangle((86, 36), 8, 8, color='r')
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
    ax.text(90, 40, r'$\mathcal{G}$', fontsize=40,
            horizontalalignment='center', verticalalignment='center')
    # ax.text(0, -20, r'$V_2 (p) = \mathcal{\infty}$', fontsize=48, color='w',
    #         horizontalalignment='center', verticalalignment='center')
    # ax.text(0, 25, r'$\mathcal{O}_2$', fontsize=48, color='w',
    #         horizontalalignment='center', verticalalignment='center')

    # Labels
    plt.title(r'Level sets of the localization function when $q = 1$', fontsize=34)
    plt.xlabel('x-coordinate', fontsize=24)
    plt.ylabel('y-coordinate', fontsize=24)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=-100, right=100)
    ax.set_ylim(bottom=-50, top=50)
    plt.xticks(ticks=np.arange(-100, 100, 10))
    plt.yticks(ticks=np.arange(-50, 50, 10))
    fig.subplots_adjust(
        top=0.92,
        bottom=0.14,
        left=0.04,
        right=1.0,
        hspace=0.0,
        wspace=0.0
    )
    cb = plt.colorbar()
    cb.lines[0].set_linewidth(10)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, orientation='landscape')


def draw_contour_plot_q2(cost_point=(90, 40), agents=None, save_path=None):
    """
    Generate level sets when q = 2.
    :param cost_point: (x, y)-coordinate of the agent's goal.
    :param agents: List of (coord, color, radius, label)-tuples describing agents in the environment.
    :param save_path: Path to save the generated figure.
    :return: None
    """

    if save_path is None:
        fig = plt.figure(figsize=(100, 200))
    else:
        fig = plt.figure(figsize=(25, 13))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    sampling = 5
    x = np.linspace(-100, 100, sampling*100)
    y = np.linspace(-50, 50, sampling*60)

    mesh_shape = (len(y), len(x))
    mesh1 = np.zeros(mesh_shape)
    mesh2 = np.zeros(mesh_shape)

    fringe1_borders = []
    flow_set = []  # C
    jump_set = []  # D

    chi, lamb = 2.5, 0.5
    fringe2, fringe1, _ = get_fringe(agents[0][0])
    for i, x_i in enumerate(x):
        c, d = None, None
        for j, y_j in enumerate(y):
            mesh1[j, i] = cost_function((x_i, y_j), cost_point, fringe1, 6)
            mesh2[j, i] = cost_function((x_i, y_j), cost_point, fringe2, 6)

            ############# FRINGE 1 IS MOUNTAIN #####################################################################
            # Fill Flow Set (C)
            if mesh1[j, i] != np.inf and mesh1[j, i] <= chi * mesh2[j, i]:
                c = y_j

            # Fill Jump Set (D)
            if mesh1[j, i] < (chi - lamb) * mesh2[j, i]:
                d = y_j
            ########################################################################################################

        fringe1_borders.append(fringe1(x_i))
        flow_set.append(c)
        jump_set.append(d)

    plt.contour(x, y, mesh1, cmap='Wistia', levels=16)
    plt.plot(x, fringe1_borders, ls='--', color='silver')

    flow_set_color = 'fuchsia'
    jump_set_color = 'cyan'
    plt.plot(x, flow_set, c=flow_set_color, lw=4)
    plt.plot(x, jump_set, c=jump_set_color, lw=4)

    flow_set_arrows = [70, 110, 150]
    for arrow in flow_set_arrows:
        plt.arrow(x[arrow], flow_set[arrow], -1, 1, color=flow_set_color, head_width=1)
        plt.arrow(x[-arrow], flow_set[-arrow], 1, 1, color=flow_set_color, head_width=1)

    arrow = 60  # 150
    ax.text(x[arrow] - 0.75, jump_set[arrow] + 2, r'$\mathit{C}_2$', fontsize=40, c=flow_set_color,
                horizontalalignment='center', verticalalignment='center')
    ax.text(x[-arrow] + 0.75, jump_set[-arrow] + 2, r'$\mathit{C}_2$', fontsize=40, c=flow_set_color,
            horizontalalignment='center', verticalalignment='center')

    jump_set_arrows = [50, 90, 130]
    for arrow in jump_set_arrows:
        plt.arrow(x[arrow], jump_set[arrow], 1, -1, color=jump_set_color, head_width=1)
        plt.arrow(x[-arrow], jump_set[-arrow], -1, -1, color=jump_set_color, head_width=1)

    arrow = 90
    ax.text(x[arrow] + 0.5, jump_set[arrow] - 4, r'$\mathit{D}_2$', fontsize=40, c=jump_set_color,
                horizontalalignment='center', verticalalignment='center')
    ax.text(x[-arrow] - 0.5, jump_set[-arrow] - 4, r'$\mathit{D}_2$', fontsize=40, c=jump_set_color,
            horizontalalignment='center', verticalalignment='center')

    # Static elements
    for i, obstacle in enumerate(get_obstacles()):
        if obstacle.shape == 'circle':
            obs = plt.Circle(obstacle.coord, radius=obstacle.diameter/2, ls='-', color=obstacle.color)
        else:
            side = obstacle.diameter/2
            obs = plt.Rectangle(obstacle.coord, side, side, color=obstacle.color)
        ax.add_patch(obs)

        ax.text(obstacle.coord[0], obstacle.coord[1], r'$\mathcal{N}_{' + f'{i+1}' + r'}$', fontsize=30,
                 horizontalalignment='center', verticalalignment='center')

    objective = plt.Rectangle((86, 36), 8, 8, color='r')
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
    ax.text(90, 40, r'$\mathcal{G}$', fontsize=40,
            horizontalalignment='center', verticalalignment='center')
    # ax.text(0, -20, r'$V_2 (p) = \mathcal{\infty}$', fontsize=48, color='w',
    #         horizontalalignment='center', verticalalignment='center')
    # ax.text(0, 25, r'$\mathcal{O}_2$', fontsize=48, color='w',
    #         horizontalalignment='center', verticalalignment='center')

    # Labels
    plt.title(r'Level sets of the localization function when $q = 2$', fontsize=34)
    plt.xlabel('x-coordinate', fontsize=24)
    plt.ylabel('y-coordinate', fontsize=24)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=-100, right=100)
    ax.set_ylim(bottom=-50, top=50)
    plt.xticks(ticks=np.arange(-100, 100, 10))
    plt.yticks(ticks=np.arange(-50, 50, 10))
    fig.subplots_adjust(
        top=0.92,
        bottom=0.14,
        left=0.04,
        right=1.0,
        hspace=0.0,
        wspace=0.0
    )
    cb = plt.colorbar()
    cb.lines[0].set_linewidth(10)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, orientation='landscape')


if __name__ == '__main__':
    # draw_contour_plot_q2()   # save_path='/home/alejandro/Downloads/test.png')
    #
    # environment_multiple_obstacles()


    def img_preprocessing(img):
        """
        Helper function to preprocess images following the perception map's preproccessing for training data.
        :param img: Matrix (image) to be preprocessed.
        :return: Preprocessed image.
        """

        # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        #     img_path = f.name
        #     cv2.imwrite(img_path, img)
        img_path = "test.png"
        import matplotlib.pyplot as plt

        plt.imshow(img) 
        return img

    def get_model_path(model_name):
        base = '../saved_models/'
        return base + model_name

    # Define simulation parameters.
    sim_params = dict(
        env_dimension=(200, 100),               # Environment dimensions (width, height).
        objective_coord=(90, 40),               # (x, y)-coordinate where the agent's goal is located.
        objective_diameter=7,                  # Diameter of the ball representing the goal.
        stop_difference=0.5,                    # Minimum difference between agent and goal to stop simulation.
        occlusion_rate=0.0,                     # Rate (0 <= r <= 1) at which random occlusions are added to the image.
        preprocessing=img_preprocessing,        # Function to preprocess images before feeding them to the vision model.
        model_path=get_model_path('cnn_200kTrain_MultipleObstacles'),
    )

    trajectory, init_x, init_y, diff = waypoint_tracking(params=sim_params, coord=(-90, 0))
