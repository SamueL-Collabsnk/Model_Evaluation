import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
import keras_tuner as kt
import shap 
from lime import lime_image

import os,random
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
print(y_train.shape)

X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0

#ensuring labels are !-D for sklearn
y_train = y_train.ravel()
y_test = y_test.ravel()

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    stratify=y_train,
    random_state=42
)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.01),
    layers.RandomZoom(0.01),
    layers.RandomTranslation(0.01,0.01),
    layers.RandomContrast(0.01)
])

def build_model(hp):
    lr = hp.Choice("Learning_rate" , [1e-4,5e-4,1e-3])
    dense_units = hp.Int("Dense_units", min_value=32,max_value=128, step=32)
    wd = hp.Choice("Weight_decay", [0.0001,0.001,0.0])
    conv_filters = hp.Choice("Conv_filters", [32,64,128])
    
    model = keras.Sequential([
        layers.Input(shape=(32,32,3)),
        data_augmentation,
        layers.Conv2D(conv_filters,
                      (3,3),
                      activation = "relu",
                      padding = "same",
                      kernel_regularizer=regularizers.l2(wd)),
        layers.BatchNormalization(),
        
        layers.Conv2D(conv_filters,
                      (3,3),
                      activation = "relu",
                      padding = "same",
                      kernel_regularizer = regularizers.l2(wd)),
        layers.BatchNormalization(),
        
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(conv_filters,
                      (3,3),
                      activation = "relu",
                      padding = "same",
                      kernel_regularizer = regularizers.l2(wd)
                      ),
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, kernel_regularizer=regularizers.l2(wd), activation = "relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation = "softmax")
    ])
    
    model.compile(
        optimizer = keras.optimizers.Adam(
            learning_rate = lr
        ),
        loss = keras.losses.SparseCategoricalCrossentropy(),
        metrics = ["accuracy"]
    )
    
    return model

tuner = kt.RandomSearch(
    build_model,
    objective= "val_accuracy",
    max_trials= 2,
    directory = "tuning",
    project_name = "cifar10_project"
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64).prefetch(AUTOTUNE)

tuner.search(
    train_ds,
    epochs = 5,
    validation_data = val_ds,  
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(
    best_hp
)


print("Best Hyperparameter values:")
print(best_hp.values)

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
    monitor = "val_loss",
    patience = 8,
    restore_best_weights = True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "best_cifar10.keras",
    monitor = "val_accuracy",
    save_best_only = True,
    mode = "max"
)

history = best_model.fit(
    train_ds,
    epochs =30,
    validation_data = val_ds,
    callbacks = [tensorboard,reduce_lr, early_stop, checkpoint]
)

print("Train acc (last):", history.history["accuracy"][-1])
print("Val acc (best):", max(history.history["val_accuracy"]))

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test
)
 
predictions = best_model.predict(X_test)

pred_class = np.argmax(predictions, axis=1)
actual_class = y_test.flatten()

cm = confusion_matrix(actual_class,pred_class)
print(cm)
sns.heatmap(
    cm,
    annot= True,
    fmt = "d",
    cmap = "Blues"
)
plt.show()

#Shap Explainer
background = X_train[:20]
masker = shap.maskers.Image("inpaint_telea",background)
explainer = shap.Explainer(
    lambda x: best_model.predict(x),
    masker = masker
)
#Generate explanation
shap_values = explainer(np.array(X_test[:10]))
    
#Visualize
shap.image_plot(shap_values)
  

#lime explainer
idx = 0
image = X_test[idx].astype("float32") 

lime_explainer = lime_image.LimeImageExplainer()

#Generate explanation
lime_exp = lime_explainer.explain_instance(
    image,
    classifier_fn=lambda imgs: best_model.predict(np.array(imgs)),
    top_labels=5,
    hide_color= 0,
    num_samples=200
)
temp, mask = lime_exp.get_image_and_mask(
    label=int(pred_class[idx]),
    positive_only = True,
    num_features = 5,
    hide_rest = True
)

print(classification_report(actual_class, pred_class))    
