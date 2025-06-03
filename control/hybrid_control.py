from copy import deepcopy
import os

from matplotlib.path import Path

from control import load_model, create_environment, controller,load_cnn_model,load_rtdetr_model,get_best_model
from control2 import plot_trajectory_normal_controller, controller2
from models.linear_model import LinearModel

# from tensorflow import make_tensor_proto, make_ndarray

import matplotlib.pyplot as plt
import numpy as np
import moviepy.video.io.ImageSequenceClip

import tempfile
import cv2

import torch

from PIL import Image
import glob


def main(params, coord=None):
    """
    Perform function calls to execute simulation.
    :param params: Dictionary with the simulation parameters.
    :return: None
    """
    if os.path.exists('result.txt'):
        os.remove('result.txt')
    else:
        with open('result.txt', 'w') as f:
            f.write('')
    
    trajectory, init_x, init_y, diff = waypoint_tracking(params, coord)
    
    # Calculate and print the average inference time
    # calculate_avg_inference_time()


# def cost_function2(current_pos, obj_pos, fringe, obs_radius):
#
#     obs_pos = (0, 0)
#     obs_diameter = 16
#
#     obs_radius = 0.5 * obs_diameter
#     x = -((current_pos[0] - obj_pos[0]) ** 2)
#     y = -((current_pos[1] - obj_pos[1]) ** 2)
#
#     if (current_pos[0] - obs_pos[0])**2 + (current_pos[1] - obs_pos[1])**2 > obs_radius**2:
#         dist = np.sqrt((current_pos[0] - obs_pos[0]) ** 2 + (current_pos[1] - obs_pos[1])**2) - obs_radius
#     else:
#         dist = 0
#
#     if dist <= obs_radius:
#         b = (np.log(obs_radius) - np.log(dist)) * ((dist - obs_radius) ** 2)
#     else:
#         b = 0
#
#     return x + y - b


def cost_function(state, target, fringe, obs_radius):
    """
    Objective function (to minimize) defining the cost to reach the goal while doing obstacle avoidance, using the
    geometric covering of the state space typical of the hybrid controller..
    :param current_pos: (x, y)-coordinate of the agent's current position.
    :param obj_pos: (x, y)-coordinate of the goal.
    :param obs_pos: (x, y)-coordinate of the obstacle.
    :param obs_diameter: Diameter of the obstacle.
    :return: Cost to reach the goal (scalar).
    """

    x = 0.5 * ((state[0] - target[0])**2)
    y = 0.5 * ((state[1] - target[1])**2)

    dist = fringe.distance(state)   # - np.sqrt(obs_radius)
    # obs_radius = np.sqrt(obs_radius)
    if dist == np.inf:
        b = dist
    elif 0 <= dist <= obs_radius:
        # b = (np.log(1) - np.log(dist)) * ((dist - obs_radius)**2)
        b = (np.log(obs_radius) - np.log(dist)) * ((dist - obs_radius)**2)
    else:
        b = 0

    return x + y + b


def hybrid_controller(params):
    """
    Implementation of the hybrid controller.
    :param params: Dictionary of simulation parameters.
    :return: The updated parameters dictionary.
    """

    state = params['state']
    current_pos = state[-1]
    obj_pos = params['obj_pos']
    if isinstance(obj_pos, list):
        obj_pos = obj_pos[-1]

    # obs_pos = params['obstacle_coord']
    obs_radius = params['obstacle_diameter'] / 2
    current_controller = params['current_controller']
    chi, lamb = params['chi'], params['lamb']

    # Fringes
    fringe_O1 = params['fringe_O1']
    fringe_O2 = params['fringe_O2']
    v1 = cost_function(current_pos, obj_pos, fringe_O1, obs_radius)
    v2 = cost_function(current_pos, obj_pos, fringe_O2, obs_radius)
    print (v1, v2)

    if current_controller == 1:
        if v2 >= (chi - lamb) * v1:  # Switch controllers
            params['current_controller'] = 2
            v = v1
            fringe = fringe_O1
            print("11")
        else:
            v = v2
            fringe = fringe_O2
            print("12")
    else:
        if v1 >= (chi - lamb) * v2:  # Switch controllers
            params['current_controller'] = 1
            v = v2
            fringe = fringe_O2
            print("21")
        else:
            v = v1
            fringe = fringe_O1
            print("22")
    params['controller_history'].append(params['current_controller'])
    



    def get_jacobian():

        h = 0.01
        cost = v

        jacobian = []
        for i in range(len(current_pos)):

            mod_params = current_pos.copy()
            mod_params[i] = mod_params[i] + h

            # cost_h = cost_function(mod_params, obj_pos, obs_pos, obs_diameter/2)
            cost_h = cost_function(mod_params, obj_pos, fringe, obs_radius)

            partial_derivative = (cost_h - cost) / h
            if partial_derivative != partial_derivative:  # Check for NaN
                partial_derivative = np.inf
            if isinstance(partial_derivative, torch.Tensor):
                jacobian.append(partial_derivative.cpu().detach().numpy())
            else:
                jacobian.append(partial_derivative)

        jacobian = np.array(jacobian)
        jacobian = np.clip(jacobian, -10, 10)  # Gradient clipping

        return jacobian

    dt = 0.1
    chi=1.1
    lamb=0.09
    u = -get_jacobian()


    # # ########################## ADVERSARIAL NOISE #####################################################################
    ###  NOTE: dt = 0.1
    saddle_point = [-13.8, 0.8]
    delta_x = 1.5
    delta_y = 1.5
    delta_x = delta_x if abs(current_pos[0] - saddle_point[0]) <= delta_x else 0.0
    delta_y = delta_y if abs(current_pos[1] - saddle_point[1]) <= delta_y else 0.0
    
    eps = 4.2
    # print(delta_x, delta_y, u, eps * delta_x * (current_pos[0] - saddle_point[0]), eps * delta_y * (current_pos[1] - saddle_point[1]))
    
    x = current_pos[0] + dt * (u[0] - eps * delta_x * (current_pos[0] - saddle_point[0]))
    y = current_pos[1] + dt * (u[1] - eps * delta_y * (current_pos[1] - saddle_point[1]))
    # # ##################################################################################################################

    # x = current_pos[0] + dt * u[0]   # Comment this out when simulating adversarial noise.
    # y = current_pos[1] + dt * u[1]   # Comment this out when simulating adversarial noise.

    # # ########################## ADD NOISE TO THE STATE#################################################################
    if params['id'] == 'reference':
        limit = 0.5
        
        # x += np.random.normal(loc=0.0, scale=limit)
        # y += np.random.normal(loc=0.0, scale=limit)
        # x += np.random.uniform(low=-limit, high=limit)
        # y += np.random.uniform(low=-limit, high=limit)

        x += np.random.uniform(low=-limit, high=limit)
        y += np.random.uniform(low=-limit, high=limit)
    # # # # ##################################################################################################################

    params['state'].append([x, y])

    return params


def waypoint_tracking(params, coord=None):
    """
    Simulate a follower agent tracking a reference agent using the Vision-Based hybrid controller.
    :param params: Dictionary with simulation parameters.
    :return: Trajectory of the agents, initial positions, and Euclidean distance to goal when simulation is stopped.
    """

    # Parameter retrieval
    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    objective_pos = params['objective_coord']
    stop_difference = params['stop_difference']
    preprocessing = params['preprocessing']


    gif_folder = 'frames'
    os.makedirs(gif_folder, exist_ok=True)

    # Convenience Functions
    def calculate_difference(params):

        pos = params['state'][-1]
        obj = params['obj_pos']
        if isinstance(obj, list):
            obj = obj[-1]

        return np.sqrt((pos[0] - obj[0]) ** 2 + (pos[1] - obj[1]) ** 2)

    def get_prediction(pred):
        # return list(make_ndarray(make_tensor_proto(pred))[0])
        # return list(pred.numpy()[0])
        from ultralytics.engine.results import Results 
        res: Results = pred[0]
        boxes = res.boxes
        # shownimage
        plt.imshow(res.orig_img)
        is_obb = res.obb is not None
        for j, d in enumerate(boxes):
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
            print(line)
        # return
        # print(res.boxes)
        if not res.boxes.xywh.shape[0]:
            return [0,0]
        # random return a coordinate
        # return [0,1]
        return list(res.boxes.xywh[0][:2])
    
        
    def make_gif(frame_folder, gif_name='trajectory.gif', zoom=1):

        # Ensure the directory exists
        if not os.path.exists(frame_folder):
            print("Frame folder does not exist.")
            return

        # Construct the correct path for the glob function
        search_path = os.path.join(frame_folder, 'frame_*.png')

        # Gather all frame paths
        frame_paths = glob.glob(search_path)

        # Debugging output
        print(f"Found {len(frame_paths)} frames.")

        # Sort frame paths by extracting the integer part of the filename
        frame_paths.sort(key=lambda path: int(os.path.basename(path).split('_')[1].split('.')[0]))

        # Load images and optionally resize them
        images = [Image.open(path) for path in frame_paths]
        if zoom != 1:
            images = [img.resize((int(img.width * zoom), int(img.height * zoom)), Image.NEAREST) for img in images]

        # Debugging output
        print("Images loaded and resized.")

        # Save images as a GIF
        if images:
            images[0].save(os.path.join(frame_folder, gif_name), save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
            print(f"GIF created successfully at {os.path.join(frame_folder, gif_name)}")
        else:
            print("No images to compile into GIF.")


    def make_simple_gif(trajectory,image_paths,reference_params, tracker_params):
        print(image_paths)  # Add this line to see what filenames are received by the function


        image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('.')[0].split('img')[1]))
        images = [Image.open(x) for x in image_paths]
        images[0].save('simple.gif',
                        save_all=True, append_images=images[1:], optimize=False, duration=400, loop=0)


    
    # init_x, init_y = 30, 50  # Default values
    if coord is None:
        init_x = 30  # np.random.randint(low=-width//2, high=0)  #width//2)
        init_y = 50  # np.random.randint(low=-height//2, high=height//2)
    else:
        init_x, init_y = coord

    # Initialize the state at the agent's initial position
    environment = create_environment(params)
    # model_name = "rtderr" # "cnn", "rtdetr", "yolo"
    # model_name = "cnn"
    model_name = "rtdetr"
    if model_name == "cnn":
        model = load_cnn_model(params)
    elif model_name == "rtdetr":
        model = load_rtdetr_model(params)
    else:
        model = load_model(params)
    yolo_model = load_model(params)
    rt_detr_model = load_model(params)
    # Controller parameters
    reference_params = dict(
        a=0.95, dt=0.1,
        chi=1.1, lamb=0.09,
        current_controller=1,
        controller_history=[1],
        state=[[init_x, init_y]],
        obj_pos=[objective_pos],
        obstacle_coord=obs_pos,
        obstacle_diameter=obs_diameter,
        color=(1.0, 1.0, 1.0),
        id='reference'
    )
    reference_params['b'] = 1 - reference_params['a']

    # Define Fringe Sets
    obs_pos = reference_params['obstacle_coord']
    diameter = reference_params['obstacle_diameter']
    obs_radius = (diameter // 2) + (diameter // 4)
    p0 = (obs_pos[0] - obs_radius, obs_pos[1])
    p2 = (obs_pos[0] + obs_radius, obs_pos[1])
    reference_params['fringe_O1'] = Fringe(  # -
        p0=p0, p1=(obs_pos[0], obs_pos[1] - obs_radius), p2=p2
    )
    reference_params['fringe_O2'] = Fringe(  # +
        p0=p0, p1=(obs_pos[0], obs_pos[1] + obs_radius), p2=p2
    )


    tracker_params = deepcopy(reference_params)
    tracker_params['a'] = reference_params['a']
    tracker_params['b'] = 1 - tracker_params['a']
    tracker_params['dt'] = reference_params['dt']
    tracker_params['color'] = (0.0, 1.0, 0.0)
    tracker_params['obj_pos'] = [reference_params['state'][0]]
    tracker_params['id'] = 'tracker'



    i = 0
    trajectory = []
    
    from pathlib import Path
    import shutil

    video_folder = Path('video_frames')
    shutil.rmtree(video_folder, ignore_errors=True)
    video_folder.mkdir(exist_ok=True)

    gif_folder = Path('frames')
    shutil.rmtree(gif_folder, ignore_errors=True)
    gif_folder.mkdir(exist_ok=True)
    
    while calculate_difference(reference_params) > stop_difference:
        frame_path = video_folder / f'frame_{i:04d}.png'
       # gif_prediction_vs_label(reference_params, tracker_params, save_path=str(frame_path))
        
        # update_agent_states(reference_params, tracker_params)
        if i == 0:
            import traceback
            traceback.print_stack()
        i += 1
       
        if i > 500:
            break

        d = tracker_params['state'][-1]    # reference_params['state'][-1]
        drawing = environment.draw_trajectory([d], color=tracker_params['color'], ignore_occlusion=False)[0]
        d2 = reference_params['state'][-1]
        drawing2 = environment.draw_trajectory([d2], color=tracker_params['color'], ignore_occlusion=False)[0]

        d_comb = np.clip(drawing + drawing2, 0.0, 1.0)
        trajectory.append(d_comb)

        # Get Previous state
        x_t, y_t = reference_params['state'][-1]
        # d 和 d2 距离太大时停止
        if np.sqrt((x_t - d2[0])**2 + (y_t - d2[1])**2) > 0.5:
            break
        print('Iteration {}:\n\tReference = ({}, {})\tTracker = ({}, {})'.format(i, x_t, y_t, d[0], d[1]))
        ##################
        # data_transmission_stop = 0   # Timestep when the follower stops getting new info. about the reference's state.
        failure_rate = 0.0    # failure_rate = 0 means that the camera never fails.
        # if i < data_transmission_stop and np.random.uniform() >= failure_rate:
        if np.random.uniform() >= failure_rate:
            # When using prediction
            drawing = environment.draw_trajectory([[x_t, y_t]])  # Take picture of environment to predict objective's state.
            img = preprocessing(drawing[0])
            img_path = gif_folder / f'img{i}.png'
            print(img_path)
            plt.imsave(img_path, img)
            # predicted_state = model(prev_state)
            mode = "predict"
            overrides = {
                "conf": 0.25,
                # "source": "/home/perception/yolov10/datasets/test/images/img862.png"
                "source"  : img_path,
                # 关闭日志输出
                #"verbose": False
            } 
            
            # predicted_state = model(prev_state, **overrides)  
            best_model_name = get_best_model(img_path)
            # dont want to use get_best_model, insteda just the load the model
            # best_model_name = "yolo"
            # print("best model name is ", best_model_name)
            if best_model_name == "yolo":
                predicted_state = yolo_model( **overrides)[0]
            elif best_model_name == "rtdetr":
                predicted_state = rt_detr_model( **overrides)[0]

            
            if model_name == "cnn":
                next_res = model( **overrides)[0]  
                predicted_state = next_res
                print("CNN Model")
            else:
                print("best model name is ", best_model_name)
                if best_model_name == "yolo":
                    next_res = yolo_model( **overrides)[0]
                else:
                    next_res = rt_detr_model( **overrides)[0]

                # print(next_res.boxes)
                if next_res.boxes.xywh.shape[0] > 0:
                    predicted_state = next_res.boxes.xywh[0][:2]
                    
                else:
                    # Handle the case where no detections are made
                    predicted_state = [90, 30]  # Example default or fallback values
                    print ("No detections made")
                    break

                make_gif(video_folder, 'final_trajectory.gif', zoom=1)

        else:
            if isinstance(tracker_params['obj_pos'][0], list):
                predicted_state = tracker_params['obj_pos'][-1]
            else:
                predicted_state = tracker_params['obj_pos']

        if isinstance(tracker_params['obj_pos'][0], list):
            tracker_params['obj_pos'].append(predicted_state)
        else:
            tracker_params['obj_pos'] = predicted_state
     #   make_simple_gif(trajectory, [str(x) for x in gif_folder.glob('*.png')],reference_params, tracker_params)

        # print('Iteration {}:\n\tReference = ({}, {})\tPrediction = ({}, {})'.format(i, x_t, y_t, predicted_state[0], predicted_state[1]))
        ##################

        # Move Tracker
        tracker_params = hybrid_controller(tracker_params)  # controller(tracker_params) to test gradient-based control

        # Move Reference
        reference_params = hybrid_controller(reference_params)

    print('Waypoint Tracking stopped in iteration {}'.format(i))
    diff = calculate_difference(reference_params)
    
    plot_prediction_vs_label(reference_params, tracker_params)

    # plot_trajectory_normal_controller(reference_params, tracker_params)
    plot_trajectory_hc(reference_params, tracker_params)
    # video_contour_plot(reference_params, tracker_params)

    return trajectory, init_x, init_y, diff

def plot_prediction_vs_label(reference_params, tracker_params):
    """
    Simple plot to illustrate the state prediction made by the perception map.
    :param reference_params: Dictionary of simulation parameters for the reference agent.
    :param tracker_params: Dictionary of simulation parameters for the follower agent.
    :return: None
    """
    # Calculate and print the average inference time
    calculate_avg_inference_time()
    import torch

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Static elements
    obstacle = plt.Circle((50, 30), radius=7.5, ls='-', color='w')
    objective = plt.Circle((90, 30), radius=3.5, color='r')
    ax.add_patch(obstacle)
    ax.add_patch(objective)

    # Labels
    plt.suptitle('State Prediction vs. Real State', fontsize=22)
    plt.xlabel('x-coordinate', fontsize=16)
    plt.ylabel('y-coordinate', fontsize=16)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=60, top=0)
    plt.xticks(ticks=np.arange(0, 100, 10))
    plt.yticks(ticks=np.arange(0, 60, 10))

    x_real = [xi.cpu().numpy() if torch.is_tensor(xi) else xi for xi, _ in reference_params['state']]
    y_real = [yi.cpu().numpy() if torch.is_tensor(yi) else yi for _, yi in reference_params['state']]

    plt.plot(x_real, y_real, ls='-', lw=2, color='w', label='Real/Target State')

    x_pred = [xi.cpu().numpy() if torch.is_tensor(xi) else xi for xi, _ in tracker_params['obj_pos']]
    y_pred = [yi.cpu().numpy() if torch.is_tensor(yi) else yi for _, yi in tracker_params['obj_pos']]

    plt.scatter(x_pred, y_pred, label='Predicted State', color='lime', marker='x', lw=2)

    difference = np.abs(np.array(y_real) - np.array(y_pred))
    max_diff = np.max(difference)
    
    print(f'Maximum difference: {max_diff:.2f}')
   

    y_upper = [yi + 0.7 for _, yi in reference_params['state']]
    y_lower = [yi - 0.7 for _, yi in reference_params['state']]
    plt.plot(x_real, y_upper, c='y', ls=':', lw=4)
    plt.plot(x_real, y_lower, c='y', ls=':', lw=4)

    rmse = np.sqrt(np.mean([(xr - xp)**2 + (yr - yp)**2 for xr, yr, xp, yp in zip(x_real, y_real, x_pred, y_pred)]))
    print(f'RMSE: {rmse:.2f}')

    x0, y0 = reference_params['state'][0]
    ax.text(x0+1, y0-3, r'$({}, {})$'.format(x0, y0), fontsize=10, c='w',
            horizontalalignment='center', verticalalignment='center')
    start = plt.Rectangle((x0-1, y0-1), 2, 2, color='w')
    ax.add_patch(start)

    # Text annotations
    ax.text(50, 30, r'$\mathcal{N}$', fontsize=24,
            horizontalalignment='center', verticalalignment='center')
    ax.text(90, 30, r'$\mathcal{G}$', fontsize=24,
            horizontalalignment='center', verticalalignment='center')

    plt.legend(fontsize=14)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.1,
        left=0.09,
        right=0.975,
        hspace=0.4,
        wspace=0.2
    )

    plt.show()
    # plt.savefig('trajectory_init({}, {}).pdf'.format(x0, y0))


def calculate_avg_inference_time():
    """
    Calculate average inference time from result.txt file.
    Returns the average time in milliseconds.
    """
    import re
    # 读取result.txt文件
    with open(r'c:\Users\Perception\Desktop\Obstacle_Avoidance\Obstacle_Avoidance_2024\result.txt', 'r') as f:
        lines = f.readlines()

    # 使用正则表达式提取inference时间
    inference_times = []
    pattern = r'(\d+\.\d+)ms inference'

    for line in lines:
        match = re.search(pattern, line)
        if match:
            inference_time = float(match.group(1))
            inference_times.append(inference_time)

    # 计算平均时间
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        print(f"Total number of samples: {len(inference_times)}")
        print(f"Average inference time: {avg_time:.2f}ms")
    else:
        print("No inference times found in the file.")


def gif_prediction_vs_label(reference_params, tracker_params, save_path=None):
    """
    Plot to illustrate the state prediction vs. real state, with an option to save the plot as an image.
    :param reference_params: Dictionary of simulation parameters for the reference agent.
    :param tracker_params: Dictionary of simulation parameters for the follower agent.
    :param save_path: Path to save the plot as an image. If None, the plot will be displayed.
    :return: None
    """
    import torch
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Static elements
    obstacle = plt.Circle((50, 30), radius=7.5, ls='-', color='white')
    objective = plt.Circle((90, 30), radius=3.5, color='red')
    ax.add_patch(obstacle)
    ax.add_patch(objective)

    # Labels
    # plt.suptitle('State Prediction vs. Real State', fontsize=22)
    plt.xlabel('x-coordinate', fontsize=16)
    plt.ylabel('y-coordinate', fontsize=16)

    # Plot configuration
    ax.set_facecolor('black')
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=60, top=0)  # Adjusted for correct orientation
    plt.xticks(ticks=np.arange(0, 100, 10))
    plt.yticks(ticks=np.arange(0, 60, 10))
    
    # Text annotations
    ax.text(50, 30, r'$\mathcal{N}$', fontsize=24,
            horizontalalignment='center', verticalalignment='center')
    ax.text(90, 30, r'$\mathcal{G}$', fontsize=24,
            horizontalalignment='center', verticalalignment='center')

    # Extract coordinates and plot
    x_real = [xi.cpu().numpy() if torch.is_tensor(xi) else xi for xi, _ in reference_params['state']]
    y_real = [yi.cpu().numpy() if torch.is_tensor(yi) else yi for _, yi in reference_params['state']]
    x_pred = [xi.cpu().numpy() if torch.is_tensor(xi) else xi for xi, _ in tracker_params['obj_pos']]
    y_pred = [yi.cpu().numpy() if torch.is_tensor(yi) else yi for _, yi in tracker_params['obj_pos']]



    rmse = np.sqrt(np.mean([(xr - xp)**2 + (yr - yp)**2 for xr, yr, xp, yp in zip(x_real, y_real, x_pred, y_pred)]))
    print(f'RMSE: {rmse:.2f}')
    difference = np.abs(np.array(y_real) - np.array(y_pred))
    max_diff = np.max(difference)
    print(f'Maximum difference: {max_diff:.2f}')

    plt.plot(x_real, y_real, 'w-', lw=2, label='Actual State')
    plt.plot(x_pred, y_pred, 'g--', lw=2, label='Predicted State')

    # from matplotlib.patches import Rectangle
    # # Draw rectangles as vehicles
    # rectangle_width = 3  # Width of the rectangle
    # rectangle_height = 3  # Height of the rectangle
    # for x, y in zip(x_real, y_real):
    #     rect = Rectangle((x - rectangle_width / 2, y - rectangle_height / 2), rectangle_width, rectangle_height, edgecolor='blue', facecolor='lightblue', alpha=0.7)
    #     ax.add_patch(rect)

     
    # Optional bounds (example)
    plt.fill_between(x_real, [y - 1.9 for y in y_real], [y + 1.9 for y in y_real], color='yellow', alpha=0.3, label='Error Bounds')

    # Annotations and legend
    plt.legend(fontsize=12)
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.09, right=0.975, hspace=0.4, wspace=0.2)

    
    # Extract coordinates and plot
    # Calculate dynamic bounds based on prediction error
    # error = np.abs(np.array(y_real) - np.array(y_pred))
     # Calculate the L2 norm (Euclidean distance) as the error
    # error = [np.sqrt((xr - xp)**2 + (yr - yp)**2) for xr, yr, xp, yp in zip(x_real, y_real, x_pred, y_pred)]
    # upper_bound = [y + e for y, e in zip(y_real, error)]
    # lower_bound = [y - e for y, e in zip(y_real, error)]
    # left_bound = [x - e for x, e in zip(x_real, error)]
    # print(error)
    # plt.fill_between(left_bound, lower_bound, upper_bound, color='yellow', alpha=0.3, label='Error Bounds')


    # Annotations and legend
    plt.legend(fontsize=12)
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.09, right=0.975, hspace=0.4, wspace=0.2)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the plot after saving to free up memory
    else:
        plt.show()




def plot_trajectory_hc(reference_params, tracker_params):
    """
    Plot the agents' trajectories and the switching of the controllers.
    :param reference_params: Dictionary of simulation parameters for the reference agent.
    :param tracker_params: Dictionary of simulation parameters for the follower agent.
    :return: None
    """

    plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios': [2.5, 1]})
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax1.set_axisbelow(True)

    # Static elements
    obstacle = plt.Circle((50, 30), radius=7.5, ls='-', color='w')
    objective = plt.Circle((90, 30), radius=3.5, color='r')
    ax1.add_patch(obstacle)
    ax1.add_patch(objective)

    # Labels
    plt.title('Agents\' trajectory', fontsize=26)
    plt.xlabel('x-coordinate', fontsize=16)
    plt.ylabel('y-coordinate', fontsize=16)

    # Plot config.
    ax1.set_facecolor('k')
    ax1.set_xlim(left=0, right=100)
    ax1.set_ylim(bottom=60, top=00)
    plt.xticks(ticks=np.arange(0, 100, 10))
    plt.yticks(ticks=np.arange(0, 60, 10))

    x = [xi for xi, _ in reference_params['state']]
    y = [yi for _, yi in reference_params['state']]

    # for i in range(10, len(y)//3):
    #     y[i] += i / 3
    # for i in range(len(y) // 3, len(y)):
    #     y[i] += 10    # 10 when (-12, 2) ;; 25 when (-37, -17)

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
    ax1.text(50, 30, r'$\mathcal{N}$', fontsize=20,
            horizontalalignment='center', verticalalignment='center')
    ax1.text(90, 30, r'$\mathcal{G}$', fontsize=20,
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




def draw_countor_plot():
    """
    Generate the plots of the level sets of both cost functions used by the hybrid controller.
    :return:
    """

    draw_countor_plot_q1()
    draw_countor_plot_q2()

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
                draw_countor_plot_q1(cost_point, agents, save_path='{}/{}.png'.format(folder, i))
            elif q == 2:
                draw_countor_plot_q2(cost_point, agents, save_path='{}/{}.png'.format(folder, i))

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


def draw_countor_plot_q1(cost_point=(40, 0), agents=None, save_path=None):
    """
    Generate level sets when q = 1.
    :param cost_point: (x, y)-coordinate of the agent's goal.
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

    obs_pos = (50, 30)
    diameter = 15
    obs_radius = (diameter // 2) + (diameter // 4)
    p0 = (obs_pos[0] - obs_radius, obs_pos[1])
    p2 = (obs_pos[0] + obs_radius, obs_pos[1])
    fringe1 = Fringe(
        p0=p0, p1=(obs_pos[0], obs_pos[1] - obs_radius), p2=p2
    )
    fringe2 = Fringe(
        p0=p0, p1=(obs_pos[0], obs_pos[1] + obs_radius), p2=p2
    )

    sampling = 5
    x = np.linspace(0, 100, sampling*100)
    y = np.linspace(60, 0, sampling*60)

    mesh_shape = (len(y), len(x))
    mesh1 = np.zeros(mesh_shape)
    mesh2 = np.zeros(mesh_shape)

    fringe1_borders = []
    flow_set = []  # C
    jump_set = []  # D

    chi, lamb = 1.1, 0.09
    for i, x_i in enumerate(x):
        c, d = None, None
        for j, y_j in enumerate(y):
            # mesh1[j, i] = cost_function((x_i, y_j), (40, 0), fringe1, 8)
            # mesh2[j, i] = cost_function((x_i, y_j), (40, 0), fringe2, 8)
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
    obstacle = plt.Circle((50, 30), radius=7.5, ls='-', color='w')
    objective = plt.Circle((90, 30), radius=3.5, color='r')
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
    ax.text(50, 30, r'$\mathcal{N}$', fontsize=40,
            horizontalalignment='center', verticalalignment='center')
    ax.text(90, 30, r'$\mathcal{G}$', fontsize=40,
            horizontalalignment='center', verticalalignment='center')
    ax.text(30, 50, r'$V_1 (p) = \mathcal{\infty}$', fontsize=48, color='w',
            horizontalalignment='center', verticalalignment='center')
    ax.text(30, 10, r'$\mathcal{O}_1$', fontsize=48, color='w',
            horizontalalignment='center', verticalalignment='center')

    # Labels
    plt.title(r'Level sets of the localization function when $q = 1$', fontsize=30)
    plt.xlabel('x-coordinate', fontsize=22)
    plt.ylabel('y-coordinate', fontsize=22)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=60, top=0)
    plt.xticks(ticks=np.arange(0, 100, 10))
    plt.yticks(ticks=np.arange(0, 60, 10))
    fig.subplots_adjust(
        top=0.95,
        bottom=0.06,
        left=0.04,
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


def draw_countor_plot_q2(cost_point=(40, 0), agents=None, save_path=None):
    """
    Generate level sets when q = 2.
    :param cost_point: (x, y)-coordinate of the agent's goal.
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

    obs_pos = (50, 30)
    diameter = 16
    obs_radius = (diameter // 2) + (diameter // 4)
    p0 = (obs_pos[0] - obs_radius, obs_pos[1])
    p2 = (obs_pos[0] + obs_radius, obs_pos[1])
    fringe1 = Fringe(
        p0=p0, p1=(obs_pos[0], obs_pos[1] + obs_radius), p2=p2
    )
    fringe2 = Fringe(
        p0=p0, p1=(obs_pos[0], obs_pos[1] - obs_radius), p2=p2
    )

    sampling = 5
    x = np.linspace(0, 100, sampling*100)
    y = np.linspace(60, 0, sampling*60)

    mesh_shape = (len(y), len(x))
    mesh1 = np.zeros(mesh_shape)
    mesh2 = np.zeros(mesh_shape)

    fringe1_borders = []
    flow_set = []  # C
    jump_set = []  # D

    chi, lamb = 1.1, 0.09
    for i, x_i in enumerate(x):
        c, d = None, None
        for j, y_j in enumerate(y):
            # mesh1[j, i] = cost_function((x_i, y_j), (40, 0), fringe1, 8)
            # mesh2[j, i] = cost_function((x_i, y_j), (40, 0), fringe2, 8)
            mesh1[j, i] = cost_function((x_i, y_j), cost_point, fringe1, 8)
            mesh2[j, i] = cost_function((x_i, y_j), cost_point, fringe2, 8)

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

    # flow_set = [f for f in flow_set if f is not None]
    # jump_set = [j for j in jump_set if j is not None]
    # print(len(x), len(flow_set), len(jump_set))

    # plt.contour(x, y, mesh2)
    # plt.matshow(mesh1, cmap='jet')

    # Static elements
    obstacle = plt.Circle((50, 30), radius=7.5, ls='-', color='w')
    objective = plt.Circle((90, 30), radius=3.5, color='r')
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
    ax.text(50, 30, r'$\mathcal{N}$', fontsize=40,
            horizontalalignment='center', verticalalignment='center')
    ax.text(90, 30, r'$\mathcal{G}$', fontsize=40,
            horizontalalignment='center', verticalalignment='center')
    ax.text(30, 50, r'$V_1 (p) = \mathcal{\infty}$', fontsize=48, color='w',
            horizontalalignment='center', verticalalignment='center')
    ax.text(30, 10, r'$\mathcal{O}_1$', fontsize=48, color='w',
            horizontalalignment='center', verticalalignment='center')


    # Labels
    plt.title(r'Level sets of the localization function when $q = 2$', fontsize=30)
    plt.xlabel('x-coordinate', fontsize=22)
    plt.ylabel('y-coordinate', fontsize=22)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=60, top=0)
    plt.xticks(ticks=np.arange(0, 100, 10))
    plt.yticks(ticks=np.arange(0, 60, 10))
    fig.subplots_adjust(
        top=0.95,
        bottom=0.06,
        left=0.04,
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



class Fringe:
    """
    Class to define the limits to cover the state space.
    """

    def __init__(self, p0, p1, p2):
        """
        Set O_i of the state covering.
        :param p0, p1, p2: points through which both fringes of the state space covering go. p1 is common to both lines,
        specifically, it is their intersection point.
        """

        assert (p0[1] < p1[1] and p1[1] > p2[1]) or (p0[1] > p1[1] and p1[1] < p2[1])

        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

        self.mountain_fringe = self.p0[1] < self.p1[1]
        self.f1 = LinearModel(p0, p1)
        self.f2 = LinearModel(p1, p2)

    def __call__(self, x):

        return self.f1(x) if x <= self.p1[0] else self.f2(x)

    def distance(self, p):
        """
        Distance of point p to the set O_i.
        :param p: (x, y)-coordinate of the point.
        :return: Distance of p to the set O_i.
        """

        x1, y1 = self.f1.intersection_point(p)
        x2, y2 = self.f2.intersection_point(p)

        def below_function(func, point):
            return ((point[1] - func.bias) / func.slope) < point[0]

        if self.mountain_fringe:
            if y1 >= self.p1[1] and y2 >= self.p1[1]:
                return self._euclidean_distance(p, self.p1)
            elif (not below_function(self.f1, p)) and p[0] <= self.p1[0]:
                return self._euclidean_distance(p, (x1, y1))
            elif below_function(self.f2, p) and p[0] >= self.p1[0]:
                return self._euclidean_distance(p, (x2, y2))
            else:
                return np.inf
        else:
            if y1 <= self.p1[1] and y2 <= self.p1[1]:
                return self._euclidean_distance(p, self.p1)
            elif (not below_function(self.f1, p)) and p[0] <= self.p1[0]:
                return self._euclidean_distance(p, (x1, y1))
            elif below_function(self.f2, p) and p[0] >= self.p1[0]:
                return self._euclidean_distance(p, (x2, y2))
            else:
                return np.inf

    @staticmethod
    def _euclidean_distance(p0, p1):
        """
        Compute the Euclidean distance between two points.
        :param p0: (x, y)-coordinate of point 0.
        :param p1: (x, y)-coordinate of point 1.
        :return: Euclidean distance between p0 and p1.
        """
        x0, y0 = p0
        x1, y1 = p1
        dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        return dist


if __name__ == '__main__':

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

    # Define simulation parameters.
    sim_params = dict(
        env_dimension=(100, 60),               # Environment dimensions (width, height).
        obstacle_coord=(50, 30),                 # (x, y)-coordinate where to place the obstacle.
        obstacle_diameter=15,                  # Diameter of the ball representing the obstacle.
        objective_coord=(90, 30),               # (x, y)-coordinate where the agent's goal is located.
        objective_diameter=7,                  # Diameter of the ball representing the goal.
        stop_difference=2,                   # Minimum difference between agent and goal to stop simulation.
        occlusion_rate=0,                    # Rate (0 <= r <= 1) at which random occlusions are added to the image.
        preprocessing=img_preprocessing,       # Function to preprocess images before feeding them to the vision model.
        model_path='/home/perception/neural_network/yolov10/runs/detect/train2/weights/best.pt',    # 'cnn_100kTrain_100%_OcclusionRate'),
        # use_learning=False
    )

    # Simulate the controller starting from points defined in the list `coordinates`.
    # coordinates = [ (30, 50), (12, 2)]   # [(-10, 26), (-12, 2), (-37, -17), (-38, 4), (-40, 0), (-46, 25)]
    coordinates = [ (10, 50)] 
    for i in range(len(coordinates)):
        main(sim_params, coord=coordinates[i])

    # Plot the level sets of the cost functions.
    # draw_countor_plot()