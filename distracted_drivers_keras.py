from keras.applications.mobilenet_v2 import MobileNetV2
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

# img size 224*224
# https://keras.io/applications/#mobilenetv2
conv_base = MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True,
                        weights='imagenet', input_tensor=None, pooling=None, classes=1000)
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# this "freezes"
conv_base.trainable = False
model.summary()

# # Fine Tuning
# # we freeze all layers before block5_conv1
# conv_base.trainable = True
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
#
# model.summary()

train_dir = "./cat_dog_car_bike/train"
test_dir = "./cat_dog_car_bike/test"
validation_dir = "./cat_dog_car_bike/val"
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(32, 32), batch_size=20, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(32, 32),
    batch_size=20,
    class_mode='categorical')

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

# model.load_weights('my_model_weights1.h5', by_name=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=800,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=80)

# model.save_weights('my_model_weights1.h5')
