import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

physical_devices = tf.config.list_physical_devices("CPU")

# tải ảnh vào tập dữ liệu
path = "myData"
count = 0
images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        curImg = cv2.resize(curImg, (32,32))
        images.append(curImg)
        classNo.append(count)
    count += 1
images = np.array(images)
classNo = np.array(classNo)

print("Data Shapes")
print("Image", images.shape)
print("Class", classNo.shape)

# chia các tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

# hàm xử lý ảnh 
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# đưa ảnh đã xử lý vào lại các tập
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# chuyển nhãn sang mã one-hot
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# chuyển ảnh sang ảnh 1 kênh
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print("Datas Shapes")
print("Train", X_train.shape, y_train.shape)
print("Validation", X_validation.shape, y_validation.shape)
print("Test", X_test.shape, y_test.shape)

# xử lý ảnh bị nghiêng, lệch,... 
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10)

# CNN model
def myModel():
    img_row = 32
    img_col = 32
    img_channel = 1
    num_filter = 32
    num_filter2 = 64
    size_of_filter = (5, 5)
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    model = Sequential()
    model.add((Conv2D(num_filter, size_of_filter, input_shape=(img_row, img_col, img_channel), activation='relu')))
    model.add((Conv2D(num_filter, size_of_filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add((Conv2D(num_filter2, size_of_filter2, activation='relu')))
    model.add((Conv2D(num_filter2, size_of_filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    return model
print(myModel().summary())

model = myModel()
# COMPILE model
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# TRAIN model
BATCH_SIZE = 100
EPOCH = 20
CNN = model.fit(
    dataGen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCH,
    validation_data=(X_validation, y_validation),
    shuffle=1)

plt.figure(0)
plt.plot(CNN.history['loss'])
plt.plot(CNN.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.figure(1)
plt.plot(CNN.history['accuracy'])
plt.plot(CNN.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# TEST model
score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)

print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# lưu model
model.save('model.h5')
cv2.waitKey(0)