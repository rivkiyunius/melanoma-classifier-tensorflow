import cv2
import tensorflow as tf


def prepare(filepath):
    IMG_SIZE = 60
    image = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb / 255.0
    new_array = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# print(prepare("../predict/2.jpg"))

model = tf.keras.models.load_model("result.model")

prediction = model.predict([prepare("../predict/ISIC_0024698.jpg")])
print(prediction)
