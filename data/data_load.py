import os

import numpy as np
import pandas as pd

from PIL import Image


class DataLoad:
    """
    Data Loader for train/val/test data set.
    """

    def __init__(self, path, dataset):

        self._path = path
        self._data_desc_path = os.path.join(self._path, dataset + '.xlsx')
        self._data = self._preprocess_data()

    def get_data(self):

        return self._data['x'], self._data['y']

    def _preprocess_data(self):

        data = pd.read_excel(self._data_desc_path)
        x, y = [], []

        for i, row in data.iterrows():

            _, name, label = row
            img_path = os.path.join(self._path, name)
            img = Image.open(img_path) #.resize((80, 80))  #.convert('L')
            img = np.asarray(img).astype("float32") / 255.0
            label = eval(label)

            x.append(img)  #.ravel())
            y.append(label)

        x = np.array(x)
        y = np.array(y)
        return dict(x=x, y=y)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Complete path where the folder with the data are located
    base_path = ''
    dataset = 'train0'    # Folder name (there should be a matching XLSX file with each image's label).
    data_loader = DataLoad(base_path, dataset)

    images, labels = data_loader.get_data()

    for i in range(len(labels)):

        if i % 50 == 0:
            img = images[i]
            label = labels[i]
            plt.imshow(img)
            plt.title(str(label))
            plt.show()
