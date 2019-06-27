import os  # a conflict in my mac system

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
# from classification_models.resnet import ResNet18, preprocess_input

from keras.applications import VGG16

conv_base = VGG16(
    weights='imagenet',
    include_top=
    False,  # we are going to remove the top layer, VGG was trained for 1000 classes, here we only have two
    input_shape=(224, 224, 3))

# img size 224*224
# https://keras.io/applications/#mobilenetv2
# mobile_base = NASNetMobile(input_shape=(224,224,3), include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
# mobile_base = ResNet18((224, 224, 3), include_top=False, weights='imagenet')

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
# this "freezes"
conv_base.trainable = False
# model.summary()

train_dir = "/Users/ranxin/Downloads/state-farm-distracted-driver-detection/imgs/train"
test_dir = "/Users/ranxin/Downloads/state-farm-distracted-driver-detection/imgs/val"

train_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_data_generator = ImageDataGenerator(rescale=1. / 255)
train_generator = train_data_generator.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=40, class_mode='categorical')
test_generator = test_data_generator.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=40,
    class_mode='categorical')

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

model.load_weights('Distracted_vgg16.h5', by_name=True)
img_path = "/Users/ranxin/Downloads/state-farm-distracted-driver-detection/imgs/test/img_1.jpg"
import cv2

new_input = cv2.imread(img_path)
print(new_input.shape)
new_input = cv2.resize(new_input, (224, 224), interpolation=cv2.INTER_CUBIC)
print(new_input.shape)
# cv2.imwrite('new_input.jpg', new_input)
new_input = new_input.reshape((-1, 224, 224, 3))
ynew = model.predict_classes(new_input)
print(type(ynew), ynew)
