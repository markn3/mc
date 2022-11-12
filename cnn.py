print("Importing packages")
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pathlib
import matplotlib.pyplot as plt
# data_dir = 'C:\\Users\\markn\\Desktop\\mc\\CNN\\data'
data_dir = 'C:/Users/markn/Desktop/mc/CNN/data'
data_dir = pathlib.Path(data_dir)
print("data_dir: ", data_dir)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    shuffle = False,
    seed=123,
    image_size=(360, 640),
    batch_size=1)
    

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    shuffle = False,
    seed=123,
    image_size=(360, 640),
    batch_size=1)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# CNN
model = tf.keras.Sequential([
    layers.Conv2D(32, (7,7), padding='same', activation='relu', kernel_regularizer = l2(0.0005)),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, (5,5),  padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(128, (3,3),  padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(256, (3,3),  padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64),
    layers.Dense(16),
    layers.Dense(2)
])

callbacks = [EarlyStopping(monitor='val_accuracy',patience=3)]
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=50)
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
y_acc = history.history['accuracy']
y_vacc = history.history['val_accuracy']

# save model
model.save("cave_classifier_1112.model")

# plot results
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.arange(len(y_vloss)), y_vloss, marker='.', c='red')
ax1.plot(np.arange(len(y_loss)), y_loss, marker='.', c='blue')
ax1.grid()
plt.setp(ax1, xlabel='epoch', ylabel='loss')

ax2.plot(np.arange(len(y_vacc)), y_vacc, marker='.', c='red')
ax2.plot(np.arange(len(y_acc)), y_acc, marker='.', c='blue')
ax2.grid()
plt.setp(ax2, xlabel='epoch', ylabel='accuracy')

plt.show()
