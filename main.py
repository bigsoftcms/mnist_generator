import keras, datetime
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, GlobalMaxPooling2D, Dense, Reshape, Flatten, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# create models
input_img = Input(shape=(28, 28, 1))

# encoder
x = Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu')(input_img)
x = Conv2D(filters=32, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=32, kernel_size=3, padding='same', strides=2, activation='relu')(x)

x = Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=16, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=16, kernel_size=3, padding='same', strides=2, activation='relu')(x)

x = Conv2D(filters=8, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=8, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = Conv2D(filters=8, kernel_size=3, padding='same', strides=2, activation='relu')(x)

x = GlobalMaxPooling2D()(x)
encoded = Dense(10, activation='softmax', name='encoded')(x)

# reshape
x = Reshape((1, 1, 10))(encoded)

# decoder
x = Conv2DTranspose(32, kernel_size=3, padding='same', strides=3, activation='relu')(x)
x = Conv2DTranspose(16, kernel_size=3, padding='valid', strides=2, activation='relu')(x)
x = Conv2DTranspose(16, kernel_size=3, padding='same', strides=2, activation='relu')(x)
x = Conv2DTranspose(8, kernel_size=3, padding='same', strides=2, activation='relu')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)

autoencoder = Model(input_img, [encoded, decoded])
autoencoder.compile(
  optimizer=keras.optimizers.Adam(),
  loss=['categorical_crossentropy', 'binary_crossentropy'],
  loss_weights=[0.1, 1],
  metrics=['accuracy']
)

autoencoder.summary()

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# tensorboard --logdir=logs
autoencoder.fit(x_train, [y_train, x_train], epochs=50, batch_size=128, shuffle=True,
  validation_data=(x_test, [y_test, x_test]), verbose=1,
  callbacks=[
    TensorBoard(log_dir='logs/%s' % (start_time)),
    ModelCheckpoint('./models/%s.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
  ]
)

# take a look at the reconstructed digits
pred = autoencoder.predict(x_test)
pred_y = pred[0]
decoded_imgs = pred[1]

n = 10
plt.figure(figsize=(10, 4), dpi=100)
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i].reshape(28, 28))
  plt.gray()
  ax.set_axis_off()

  # display reconstruction
  ax = plt.subplot(2, n, i + n + 1)
  plt.title('%s' % (np.argmax(pred_y[i])))
  plt.imshow(decoded_imgs[i].reshape(28, 28))
  plt.gray()
  ax.set_axis_off()

plt.show()
