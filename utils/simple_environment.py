import numpy as np

import matplotlib.pyplot as plt


class SimpleEnvironment:
    """
    Class to represent the agent's environment.
    """

    def __init__(self, width, height, occlusion_rate):
        assert width > 0 and height > 0, 'Width and Height must be greater than zero.'

        self._width = width
        self._height = height
        self._field = np.zeros((self._height, self._width, 3))
        self._obstacles = []
        self._occlusion_rate = occlusion_rate

    def get_field_copy(self):

        return self._field.copy()

    def add_agent_object(self, agent):
        if isinstance(agent, SimpleObstacle):
            self.add_agent(agent.coord, agent.diameter, agent.shape, agent.color)

    def add_agent(self, coord, diameter, shape='top_car', color=(1.0, 1.0, 1.0)):

        diameter = abs(diameter)
        # if diameter % 2 != 0:
        #     diameter += 1

        num_channels = 3
        agent = np.zeros((diameter, diameter, num_channels))

        if shape == 'circle':
            # Set Color
            inside = lambda x, y: (x - diameter // 2) ** 2 + (y - diameter // 2) ** 2 < (diameter // 2) ** 2
            for i in range(diameter):
                for j in range(diameter):
                    if inside(i, j):
                        agent[j, i, :] = color
        elif shape == 'top_car':
           # Draw the main body of the car
            agent[:, :, :] = color

            # Draw tires at the four corners
            tire_size = max(1, diameter // 8)  # Ensure at least a size of 1
               # Draw tires at the four corners
            tire_color = (0.0, 0.9, 0.0)  # Black tires
            # Positions for tires to be rectangles of size 1x2 pixels
            tire_positions = [
                (0, 0), (0, diameter - 2),  # Top tires, 2 pixels wide
                (diameter - 1, 0), (diameter - 1, diameter - 2)  # Bottom tires, 2 pixels wide
            ]
            for corner in tire_positions:
                agent[corner[0]:corner[0]+1, corner[1]:corner[1]+2, :] = tire_color
            # Draw a smaller rectangle for the roof in the middle of the car
            roof_width = diameter // 2 +1
            roof_height = max(4, diameter // 4)  # Ensure at least a height of 1
            roof_x = (diameter - roof_width) // 1 -2
            roof_y = (diameter - roof_height) // 2
            roof_color = (0.5, 0.5, 0.5)  # Grey roof
            agent[roof_y:roof_y+roof_height, roof_x:roof_x+roof_width+1, :] = roof_color
        elif shape == 'car':
            # Draw a simple car shape
            body_height = diameter // 2
            body_width = diameter
            agent[body_height:, :body_width, :] = color

            # Car roof (slightly smaller rectangle)
            roof_height = diameter // 4
            roof_width = diameter // 2
            roof_start_x = diameter // 4
            roof_start_y = body_height - roof_height
            agent[roof_start_y:body_height, roof_start_x:roof_start_x + roof_width, :] = (0.5, 0.5, 0.5)  # grey roof

            # Draw tires (two white squares)
            tire_size = diameter // 8
            tire_color = (1.0, 1.0, 1.0)  # white tires
            
            # Position of the tires
            tire_y = body_height + roof_height + tire_size # Positioning tires just below the car body
            tire1_x = diameter // 6
            tire2_x = body_width - diameter // 6 - tire_size
            
            agent[tire_y:tire_y + tire_size, tire1_x:tire1_x + tire_size, :] = tire_color
            agent[tire_y:tire_y + tire_size, tire2_x:tire2_x + tire_size, :] = tire_color

        elif shape == 'star':
            # Draw a star shape
            center = diameter // 2
            agent[center, :] = color  # horizontal line
            agent[:, center] = color  # vertical line

            # Diagonals
            for i in range(diameter):
                if 0 <= center - i < diameter and 0 <= center + i < diameter:
                    agent[center - i, center - i] = color
                    agent[center + i, center + i] = color
                    agent[center - i, center + i] = color
                    agent[center + i, center - i] = color
        else:  # shape == 'square'
            # Set color
            for i in range(num_channels):
                agent[:, :, i] = np.full((diameter, diameter), color[i])

        self._draw(agent, coord)

    def draw_trajectory(self, trajectory, diameter=5, shape='top_car', color=(0.0, 0.0, 1.0), ignore_occlusion=False):

        diameter = abs(diameter)
        # if diameter % 2 != 0:
        #     diameter += 1

        num_channels = 3
        agent = np.zeros((diameter, diameter, num_channels))

        if shape == 'circle':
            # Set Color
            inside = lambda x, y: (x - diameter // 2) ** 2 + (y - diameter // 2) ** 2 < (diameter // 2) ** 2
            for i in range(diameter):
                for j in range(diameter):
                    if inside(i, j):
                        agent[j, i, :] = color
        elif shape == 'car':
            # Draw a simple car shape
            body_height = diameter // 2
            body_width = diameter
            agent[body_height:, :body_width, :] = color

            # Car roof (slightly smaller rectangle)
            roof_height = diameter // 4
            roof_width = diameter // 2
            roof_start_x = diameter // 4
            roof_start_y = body_height - roof_height
            agent[roof_start_y:body_height, roof_start_x:roof_start_x + roof_width, :] = (0.5, 0.5, 0.5)  # grey roof

            # Draw tires (two white squares)
            tire_size = diameter // 8
            tire_color = (1.0, 1.0, 1.0)  # white tires
            
            # Position of the tires
            tire_y = body_height + roof_height + tire_size # Positioning tires just below the car body
            tire1_x = diameter // 6
            tire2_x = body_width - diameter // 6 - tire_size
            
            agent[tire_y:tire_y + tire_size, tire1_x:tire1_x + tire_size, :] = tire_color
            agent[tire_y:tire_y + tire_size, tire2_x:tire2_x + tire_size, :] = tire_color
        elif shape == 'top_car':
           # Draw the main body of the car
            agent[:, :, :] = color

            # Draw tires at the four corners
            tire_size = 1  # Ensure at least a size of 1
               # Draw tires at the four corners
            tire_color = (0.0, 0.9, 0.0)  # Black tires
            # Positions for tires to be rectangles of size 1x2 pixels
            tire_positions = [
                (0, 0), (0, diameter - 2),  # Top tires, 2 pixels wide
                (diameter - 1, 0), (diameter - 1, diameter - 2)  # Bottom tires, 2 pixels wide
            ]
            # print(tire_positions)
            for corner in tire_positions:
                agent[corner[0]:corner[0]+1, corner[1]:corner[1]+2, :] = tire_color
            # Draw a smaller rectangle for the roof in the middle of the car
            roof_width = 4
            roof_height = 3  # Ensure at least a height of 1
            roof_x = (diameter - roof_width) // 1 -1
            roof_y = (diameter - roof_height) // 2
            roof_color = (0.5, 0.5, 0.5)  # Grey roof
            agent[roof_y:roof_y+roof_height, roof_x:roof_x+roof_width, :] = roof_color
        elif shape == 'star':
            # Draw a star shape
            center = diameter // 2
            agent[center, :] = color  # horizontal line
            agent[:, center] = color  # vertical line

            # Diagonals
            for i in range(diameter):
                if 0 <= center - i < diameter and 0 <= center + i < diameter:
                    agent[center - i, center - i] = color
                    agent[center + i, center + i] = color
                    agent[center - i, center + i] = color
                    agent[center + i, center - i] = color
        elif shape == 'square':
            # Set color
            for i in range(num_channels):
                agent[:, :, i] = np.full((diameter, diameter), color[i])
            

        drawing = []
        for coord in trajectory:
            # print(coord)
            past, y_coord, x_coord = self._draw(agent, coord)

            copy = self.get_field_copy()

            # Add occlusion with probability `self._occlusion_rate`
            if (not ignore_occlusion) and (np.random.rand() < self._occlusion_rate):
                occlusion = self._get_occlusion_element()
                for x in range(copy.shape[1]):
                    for y in range(copy.shape[0]):
                        pixel_val = copy[y, x, :] + occlusion[y, x, :]
                        if np.any(pixel_val > 1):
                            pixel_val = 0.5 * copy[y, x, :] + occlusion[y, x, :]
                        copy[y, x, :] = pixel_val
            # if (not ignore_occlusion) and (np.random.rand() < self._occlusion_rate):
            #     occlusion = self._get_occlusion_element()
            #     pixel_vals = copy + occlusion
            #     overflows = pixel_vals > 1
            #     pixel_vals[overflows] = 0.5 * copy[overflows] + occlusion[overflows]
            #     copy = pixel_vals

            # 中心点
            # copy[coord[1], coord[0], :] = np.array([1.0, 0.0, 0.0])
            
            drawing.append(copy)
            # MODIFY
            self._field[y_coord[0]:y_coord[1]+1, x_coord[0]:x_coord[1]+1, :] = past

        return drawing

    # Private methods
    def _draw(self, element, coord):
        w, h, _ = element.shape
        assert w < self._width and h < self._width, 'The element must fit on the instance\'s field.'
        w = w // 2
        h = h // 2


        x, y = self._cartesian_to_index(coord)
        # print("center",x,y,coord,"width",w)
        
        x_lhs, y_up = max(x - w, 0), max(0, y - h)
        x_rhs, y_bottom = min(x + w, self._width), min(self._height, y + h)
        # print("corners",x_lhs, y_up, x_rhs, y_bottom)

        # Truncate element to fit inside board
        element = element.copy()
        # if x - w < 0:
        #     element = element[:, w - x:, :]

        # if x + w > self._width:
        #     element = element[:, :x_rhs - x_lhs, :]

        # if y - h < 0:
        #     element = element[h - y:, :, :]

        # if y + h > self._height:
        #     element = element[:y_bottom - y_up, :, :]

        # MODIFY
        past = self._field[y_up:y_bottom+1, x_lhs:x_rhs+1, :].copy()
        print(past.shape, element.shape)
        if past.shape == element.shape:
            # print("!!!",y_up, y_bottom, x_lhs, x_rhs)
            # MODIFY
            self._field[y_up:y_bottom+1, x_lhs:x_rhs+1, :] = element
        else:
            print('Could not perform assignment')
            print(y_up, y_bottom, x_lhs, x_rhs)
        
        # print (y_up, y_bottom, x_lhs, x_rhs)

        return past, (y_up, y_bottom), (x_lhs, x_rhs)

    def _get_occlusion_element(self):
        def rand_coord(w, h):
            return np.random.randint(low=0, high=w), np.random.randint(low=0, high=h)

        occlusion = np.zeros((self._height, self._width, 3))
        cloud_center = rand_coord(self._width, self._height)
        num_bubbles = np.random.randint(1, 3)

        for _ in range(num_bubbles):
            # Generate bubble's diameter and deviation from cloud_center
            diameter = int(np.random.uniform(4, 30))
            deviation = int(np.random.uniform(-10, 10))
            # alpha = np.random.uniform(0.1, 0.4)
            alpha = 0.5
            color = np.array([alpha, alpha, alpha])  # Use alpha in all channels for effect

            # Calculate bubble center based on deviation
            x0, y0 = cloud_center[0] + deviation, cloud_center[1] + deviation

            # Lambda function to check if a point is inside the bubble
            inside = lambda x, y: (x - x0)**2 + (y - y0)**2 < diameter**2

            # Draw bubble
            for x in range(max(0, x0 - diameter), min(self._width, x0 + diameter)):
                for y in range(max(0, y0 - diameter), min(self._height, y0 + diameter)):
                    if inside(x, y):
                        occlusion[y, x, :] = color

        return occlusion
        

    def _cartesian_to_index(self, coord):
        x, y = coord
        x_index = x  # x is assumed to be non-negative and within the field width
        # y_index = self._height - y - 1  # Flips the y-coordinate
        y_index = y  # Start y from the bottom
        return round(x_index), round(y_index)

class SimpleObstacle:

    def __init__(self, coord, diameter=4, shape='square', color=(1.0, 1.0, 1.0)):

        self.coord = coord
        self.diameter = diameter
        self.shape = shape
        self.color = color


if __name__ == '__main__':
    # agent = np.zeros((4, 4, 3))
    # agent[:2, :2, 0] = np.ones((2, 2))
    # agent[2:, :2, 1] = np.ones((2, 2))
    # agent[:2, 2:, 2] = np.ones((2, 2))
    # agent[2:, 2:, :] = np.ones((2, 2, 3))
    # agent = np.ones((100, 100, 3))

    # env = SimpleEnvironment(600, 600)
    # env.draw(agent, (0, 0))
    # env.add_agent((0, 0), radius=20, color=(1, 1, 0))
    # env.add_agent((100, -50), radius=100, color=(0, 1, 1), shape='circle')
    # env.add_agent((-180, 100), radius=100, color=(0.3, 0.12, 0.4))

    environment = SimpleEnvironment(100, 60, occlusion_rate=0.9)
    environment.add_agent((50, 30), 15, shape='circle')  # obstacle
    environment.add_agent((90, 30), 7, shape='circle', color=(1.0, 0.0, 0.0))  # objective point
    environment.add_agent((50,30), 5, shape='top_car', color=(0.0, 0.0, 1.0))  # draw vehicle

    plt.imshow(environment.get_field_copy())
    plt.show()