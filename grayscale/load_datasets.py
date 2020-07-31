import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

DATADIR = "D:\Thesis\PycharmProjects\PycharmProjects\ExampleDataImages\MelanomaImages"
img_rgb = 0
IMG_SIZE = 50
data_training = []
X = []
y = []


def create_training_data():
    for img in os.listdir(DATADIR):
        try:
            img_array = cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
            # plt.imshow(img_array, cmap="gray")
            # plt.show()
            img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data_training.append([img_resize, 0])
            # break
        except Exception as e:
            print(e)


create_training_data()

for features, label in data_training:
    X.append(features)
    y.append(label)
    # break

X = np.array(X).reshape(-1, 50, 50, 1)
y = np.array(y)

# print(leno(X))

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

print(len(X))
