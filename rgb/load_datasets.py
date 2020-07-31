import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

DATADIR = "D:\Thesis\PycharmProjects\PycharmProjects\ExampleDataImages\dataset"
DATADIR_TEST = "D:\Thesis\PycharmProjects\PycharmProjects\ExampleDataImages\image_test"
CATEGORIES = ["Melanoma", "NotMelanoma"]
img_rgb = 0
img_rgb_test = 0
IMG_SIZE = 60
data_training = []
data_tests = []
X_train = []
y_train = []
X_test = []
y_test = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_resize = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                data_training.append([img_resize, class_num])
            except Exception as e:
                print(e)


print("Create data training")
create_training_data()
print(len(data_training))


def data_test():
    for category in CATEGORIES:
        path = os.path.join(DATADIR_TEST, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_resize = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                data_tests.append([img_resize, class_num])
            except Exception as e:
                print(e)


print("Create data test")
data_test()

for features, label in data_training:
    X_train.append(features)
    y_train.append(label)
    # break

print(y_train)

X_train = np.array(X_train).reshape(-1, 60, 60, 3)
y_train = np.array(y_train)

for features, label in data_tests:
    X_test.append(features)
    y_test.append(label)
    # break

X_test = np.array(X_test).reshape(-1, 60, 60, 3)
y_test = np.array(y_test)

pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

pickle_in = open("y_test.pickle", "rb")
y = pickle.load(pickle_in)

print(len(y))
# print(len())
