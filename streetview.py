from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow_datasets as tfds


(ds_train, ds_val, ds_test), ds_info = tfds.load(
    "svhn_cropped",
    split =["train[:80%]",
            "train[80%:]",
            "test"],
    as_supervised = True,
    with_info = True
)

print("Dateset info:", ds_info)

#Preprocess the Data
def preprocess(image, label):
    image = tf.cast(image, tf.float32)/255.0
    
    return image, label

#Apply preprocessing
ds_train = ds_train.map(preprocess, num_parallel_calls =tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess,num_parallel_calls = tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1)
])

#Batching 
BATCH_SIZE = 32

ds_train = ds_train.shuffle(1024).batch(BATCH_SIZE)
ds_val = ds_val.batch(BATCH_SIZE)
ds_test = ds_test.batch(BATCH_SIZE)

#Prefetching 
AUTOTUNE = tf.data.AUTOTUNE

ds_train = ds_train.prefetch(AUTOTUNE)
ds_val  = ds_val.prefetch(AUTOTUNE)
ds_test = ds_test.prefetch(AUTOTUNE)

model = keras.Sequential([
    layers.Input(shape=(32,32,3)),
    data_augmentation,
    layers.Conv2D(32, (3,3),activation = "relu",kernel_regularizer=regularizers.l2(1e-4),padding = "same"),
    layers.BatchNormalization(),
    layers.Conv2D(32,(3,3),activation = "relu",kernel_regularizer=regularizers.l2(1e-4),padding = "same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64,(3,3),activation = "relu", kernel_regularizer=regularizers.l2(1e-4),padding = "same"),
    layers.BatchNormalization(),
    layers.Conv2D(64,(3,3),activation = "relu",kernel_regularizer=regularizers.l2(1e-4),padding ="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation="relu",kernel_regularizer=regularizers.l2(1e-4),padding ="same"),
    layers.BatchNormalization(),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation = "relu",kernel_regularizer = regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.Dense(10, activation = "softmax")
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)
tensorboard = keras.callbacks.TensorBoard(
    log_dir = "logs",
    histogram_freq = 1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 3,
    min_lr = 1e-6,
    verbose = 1,
    mode = "min"
)

early_stop = keras.callbacks.EarlyStopping(
    monitor ="val_loss",
    patience = 8,
    restore_best_weights = True,
    mode ="min"
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "best_Svhn.keras",
    monitor = "val_accuracy",
    save_best_only = True,
    mode = "max"
)

history = model.fit(
    ds_train,
    epochs = 30,
    validation_data = ds_val,
    callbacks = [tensorboard,
                 reduce_lr,
                 early_stop,
                 checkpoint]
)
loss, accuracy = model.evaluate(
    ds_test
)
print("Accuracy:", accuracy)

model.save("streetview_model.keras")






