# This is the IMDB review data set. We are trying to do binary classification.
# reviews are labeled as 'positive' and 'negative'. words in each review are represented as numbers ranked by occurrence
# frequency.
import numpy as np

# import and label data as train and test samples
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
max([max(sequence) for sequence in train_data])

# explore original data
print('categories', np.unique(train_data))
length = [len(i) for i in train_data]
print('average length of review', np.mean(length))
print('standard deviation of length', np.std(length))
# look at a single training example
print('Label:', train_labels[0])
print(train_data[0])

# decode values back into words
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join([reverse_index.get(i - 3, "#") for i in train_data[0]])
print(decoded)


#  data preparation
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for a, b in enumerate(sequences):
        results[a, b] = 1
    return results

x_train = vectorize(train_data)
x_test = vectorize(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# build neural network
# build layers
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
# build model
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# set up a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# train with mini-batch size of 512 samples
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# visualize training and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, 21)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


