import numpy as np
import matplotlib.pyplot as plt

class SimpleEnvironment:
    """
    Represents the agent's environment, allowing for agents of various shapes to be added and visualized.
    """
    def __init__(self, width, height, occlusion_rate):
        assert width > 0 and height > 0, "Width and Height must be greater than zero."
        self._width = width
        self._height = height
        self._field = np.zeros((height, width, 3))
        self._occlusion_rate = occlusion_rate

    def get_field_copy(self):
        return self._field.copy()

    def add_agent(self, coord, diameter, shape='square', color=(1.0, 1.0, 1.0)):
        diameter = max(abs(diameter), 4)  # Ensure minimum size
        if diameter % 2 != 0:
            diameter += 1  # Ensure diameter is even

        agent = np.zeros((diameter, diameter, 3))
        shape_drawer = {
            'circle': self._draw_circle,
            'top_car': self._draw_top_car,
            'car': self._draw_car,
            'star': self._draw_star,
            'square': self._draw_square
        }

        draw_function = shape_drawer.get(shape, self._draw_square)  # Default to square if shape is unknown
        draw_function(agent, color)
        self._place_agent(agent, coord)

    def _draw_circle(self, agent, color):
        center = agent.shape[0] // 2
        radius_squared = (center ** 2)
        for i in range(agent.shape[0]):
            for j in range(agent.shape[1]):
                if (i - center) ** 2 + (j - center) ** 2 <= radius_squared:
                    agent[i, j] = color

    def _draw_top_car(self, agent, color):
        # Code for drawing the top view of a car
        pass

    def _draw_car(self, agent, color):
        # Code for drawing the side view of a car
        pass

    def _draw_star(self, agent, color):
        # Code for drawing a star
        pass

    def _draw_square(self, agent, color):
        agent[:, :] = color  # Fill entire area

    def _place_agent(self, agent, coord):
        x, y = coord
        half = agent.shape[0] // 2
        slice_x = slice(max(0, x - half), min(self._width, x + half))
        slice_y = slice(max(0, y - half), min(self._height, y + half))
        self._field[slice_y, slice_x] = agent[:slice_y.stop - slice_y.start, :slice_x.stop - slice_x.start]

if __name__ == '__main__':
    env = SimpleEnvironment(100, 60, 0.1)
    env.add_agent((50, 30), 10, 'circle', (1, 0, 0))
    env.add_agent((20, 20), 8, 'star', (0, 1, 0))
    env.add_agent((40, 10), 16, 'top_car', (0.5, 0.5, 0.5))

    plt.imshow(env.get_field_copy())
    plt.show()
