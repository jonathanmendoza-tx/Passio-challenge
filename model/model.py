import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory


# using toy dataset for classifying cats and dogs
URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file(
	'cats_and_dogs.zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (244, 244)

train_dataset = image_dataset_from_directory(train_dir,
												shuffle=True,
												batch_size=BATCH_SIZE,
												image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
													shuffle=True,
													batch_size=BATCH_SIZE,
													image_size=IMG_SIZE)

class_names = train_dataset.class_names

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

# prefetch images for increased model performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# add variations of images, flipped and rotated for better training
data_augmentation = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
	tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

# rescale images
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# assumed RGB channels will be last for input_shape
# not including the top will allow us to use the pretrained conv-net as a
# feature extractor
mnv2_model = tf.keras.applications.MobileNetV2(include_top=False,
											   weights='imagenet',)

# freeze pre-trained layer weights so they are not updated during training.
mnv2_model.trainable = False

image_batch, label_batch = next(iter(train_dataset))
feature_batch = mnv2_model(image_batch)

# image RGB value pooling
global_avg_layer = layers.GlobalAveragePooling2D()
feature_batch_average = global_avg_layer(feature_batch)

# L2 regularization layer
regularization_layer = layers.Dense(1, kernel_regularizer='l2')

# build personalized model
inputs = tf.keras.Input(shape=(244, 244, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = mnv2_model(x, training=False)
x = global_avg_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = regularization_layer(x)
model = tf.keras.Model(inputs, output)

#compile and run model
learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
				loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
				metrics=['accuracy'])

epochs = 10

history = model.fit(train_dataset,
					epochs=epochs,
					validation_data=validation_dataset)

# save model - serialize weights to HDF5
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
	model.save_weights("weights.h5")
	print(f'Model weights saved.')
