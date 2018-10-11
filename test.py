import keras, sys
from keras.models import load_model
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input
import numpy as np
import matplotlib.pyplot as plt

model_name = sys.argv[1]

base_model = load_model(model_name)

base_model.summary()

layers = [l for l in base_model.layers]

# load mnist dataset
(_, _), (x_test, y_test) = mnist.load_data()

inputs = Input(shape=(10,))
random_input = Input(shape=(1,))

net = None
for i in range(13, len(layers)):
  print(layers[i].name)
  if i == 13:
    net = layers[i]([inputs, random_input])
  else:
    net = layers[i](net)

# recreate new decoder only model
model = Model(inputs=[inputs, random_input], outputs=net)
model.summary()

def corrupted_logits(size, num=None):
  logits = []
  if num is None:
    num = np.random.randint(10, size=size)
  for i in range(size):
    logit = np.abs(np.random.logistic(0, 0.0005, 10))
    logit[num[i]] = 0.0
    answer = 1 - sum(logit)
    logit[num[i]] = answer
    logits.append(logit)
  return np.array(logits)

input_logits = corrupted_logits(10)
input_seed = np.argmax(input_logits, axis=1)

r_test = np.random.sample(10)

pred = model.predict([input_logits, r_test])

plt.figure(figsize=(10, 4), dpi=100)
n = len(pred)
for i, p in enumerate(pred):
  # display reconstruction
  ax = plt.subplot(2, n, i + n + 1)
  plt.title('%s' % (input_seed[i]))
  plt.imshow(p.reshape(28, 28))
  plt.gray()
  ax.set_axis_off()

plt.show()
