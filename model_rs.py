from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
batch_size = 10
train_datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.5,
        height_shift_range=0.5,
        rescale=1./255,
        shear_range=0.75,
        zoom_range=0.75,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range = (0.1,0.9))
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('RSdata/train', 
                                                    target_size=(512, 384),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                   color_mode='grayscale',
                                                   seed = random.randint(1,10),
                                                   shuffle = True)
validation_generator = test_datagen.flow_from_directory('RSdata/train',
                                                       target_size=(512, 384),
                                                       batch_size=batch_size,
                                                        class_mode='categorical',
                                                        color_mode='grayscale',
                                                        shuffle = True)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam

model1 = Sequential()
model1.add(Conv2D(32, (5, 5), input_shape=(512, 384, 1)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(32, (5, 5)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(32, (5, 5)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (5, 5)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (5, 5)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (5, 5)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))


model1.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model1.add(Dense(64))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(2))
model1.add(Activation('sigmoid'))

model1.summary()

model1.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
ckpt = ModelCheckpoint('CNN_model_e{epoch:02d}', monitor='val_acc', save_best_only = True, verbose = 1)
cb = [ckpt]

model1.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=cb)
model1.save_weights('first_try.h5')

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)
