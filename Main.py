from DataLoader import MnistDataLoader
import numpy as np
import tensorflow as tf
from CapsuleNetwork import CapsuleNetwork


cpn = CapsuleNetwork()
mnist = MnistDataLoader()


def accuracy(predicted, actual):
    correct = predicted == actual

    return np.mean(correct)


noOfTrainImages = 60000
noOfTestImages = 10000
noOfEpocs = 20
BATCH_SIZE = 100
noOfClasses = 10

X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

digitCaps_len = cpn.create_CapsuleNetwork(X, parameters=None, isTraining=True)
loss = cpn.margin_loss(y=Y, y_predicted=digitCaps_len)

global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss=loss, global_step=global_step)

noOfTestBatch = int(noOfTestImages / BATCH_SIZE)
noOfTrainBatch = int(noOfTrainImages / BATCH_SIZE)
init = tf.global_variables_initializer()
save_file = "saved_models/mnist"

saver = tf.train.Saver()
with tf.Session() as session:
    session.run(init)
    for e in range(noOfEpocs):
        for n in range(noOfTrainBatch):
            start = (n * BATCH_SIZE)

            batch_x, batch_y = mnist.load_train_batch(start_index=start, batch_size=BATCH_SIZE)

            batch_x = batch_x / 255.0
            batch_x = np.reshape(batch_x, (-1, 28, 28, 1))

            _, loss1 = session.run([train_step, loss], feed_dict={X: batch_x, Y: batch_y})
        saver.save(session, save_file + "_" + str(e) + ".ckpt")
        print('epoch: {0}, loss - {1}'.format(e, loss1))

        accuracies = []
        actual = []
        predicted = []
        for n in range(noOfTestBatch):
            start = (n * BATCH_SIZE)
            test_batch_x, test_batch_y = mnist.load_test_batch(start_index=start, batch_size=BATCH_SIZE)

            test_batch_x = test_batch_x / 255.0
            test_batch_x = np.reshape(test_batch_x, (-1, 28, 28, 1))

            vector_length1 = digitCaps_len.eval(feed_dict={X: test_batch_x})

            p = np.argmax(vector_length1, axis=1)
            a = np.argmax(test_batch_y, axis=1)
            actual = np.hstack([actual, a])
            predicted = np.hstack([predicted, p])

        accuarcy = accuracy(predicted=predicted, actual=actual)
        print('Test acc - {0}'.format(accuarcy))




