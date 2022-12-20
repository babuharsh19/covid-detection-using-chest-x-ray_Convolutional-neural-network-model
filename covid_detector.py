TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/test"

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import PIL
from tensorflow.keras.utils import load_img, img_to_array

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(224,224,3)))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1,activation="sigmoid"))

model.compile(loss=keras.losses.binary_crossentropy,optimizer = "adam",metrics=["accuracy"])

model.summary()

# Train From Scratch
# Data Augmentation
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)
test_dataset = image.ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)
print(train_generator.class_indices)
validation_generator = test_dataset.flow_from_directory(
    'dataset/test',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)

hist = model.fit(
    train_generator,
    steps_per_epoch = 6,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 2
)

# Loss is very less and accuracy is on point
model.save("Detection_Covid_19.h5")
model.evaluate(train_generator)
model.evaluate(validation_generator)

# Test Images
model = load_model("Detection_Covid_19.h5")
import os
print(train_generator.class_indices)

# Confusion Matrix
y_actual = []
y_test = []
for i in os.listdir("dataset/test/normal"):
  img = keras.utils.load_img("dataset/test/normal/"+i,target_size=(224,224))
  img = keras.utils.img_to_array(img)
  img = np.expand_dims(img,axis=0)
  p = (model.predict(img)> 0.5).astype("int32")
  y_test.append(p[0,0])
  y_actual.append(1)

for i in os.listdir("dataset/test/covid"):
  img = keras.utils.load_img("dataset/test/covid/"+i,target_size=(224,224))
  img = keras.utils.img_to_array(img)
  img = np.expand_dims(img,axis=0)
  p = (model.predict(img)> 0.5).astype("int32")
  y_test.append(p[0,0])
  y_actual.append(0)

y_actual = np.array(y_actual)
y_test = np.array(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_actual,y_test)

import seaborn as sns
print(sns.heatmap(cm,cmap = "plasma" , annot=True))

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class_names = ["Covid-19","Normal"]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="plasma"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Confusion Matrix

plt.figure()
plot_confusion_matrix(cm, classes=class_names,title='Confusion matrix for Covid-19 Detection',cmap="plasma")

# List all data in history

history = hist


print(history.history.keys())

# Summarize history for accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()