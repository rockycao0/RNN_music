# rockycao
# coding: UTF-8

import numpy as np
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import getcwd
from sklearn.preprocessing import OneHotEncoder


###################################################
# In order for this code to work, you need to place this file in the same
# directory as the midi_manipulation.py file and the Pop_Music_Midi directory

import midi_manipulation


def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs


songs = get_songs('Pop_Music_Midi')  # These songs have already been converted from midi to msgpack
print("{} songs processed".format(len(songs)))
###################################################

### HyperParameters
# First, let's take a look at the hyperparameters of our model:
TRAINING_PHASE = True

n_units = 256  # number of RNN units

lowest_note = midi_manipulation.lowerBound  # the index of the lowest note on the piano roll
highest_note = midi_manipulation.upperBound  # the index of the highest note on the piano roll
note_range = highest_note - lowest_note  # the note range

num_timesteps = 10  # This is the number of timesteps that we will create at a time
eval_time_steps = 2  # how many timesteps of the input we want to use for backpropagation
n_visible = 2 * note_range * num_timesteps  # This is the size of the visible layer.
n_hidden = 50  # This is the size of the hidden layer

num_epochs = 200  # The number of training epochs that we are going to run. For each epoch we go through the entire data set.
batch_size = 100  # The number of training examples that we are going to send through the RBM at a time.
lr = tf.constant(0.005, tf.float32)  # The learning rate of our model

### Variables:
# Next, let's look at the variables we're going to use:

x = tf.placeholder(tf.float32, [batch_size, num_timesteps, n_visible], name="x")  # The placeholder variable that holds our data
y = tf.placeholder(tf.float32, [batch_size, eval_time_steps, n_visible], name="y")  # hold backprop input
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01),
                name="W")  # The weight matrix that stores the edge weights
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh"))  # The bias vector for the hidden layer
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))  # The bias vector for the visible layer


#### Helper functions.

# This function lets us easily sample from a vector of probabilities
def sample(probs):
    # Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


# This function runs the gibbs chain. We will call this function in two places:
#    - When we define the training update step
#    - When we sample our music segments from the trained RBM
def gibbs_sample(k):
    # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        # Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))  # Propagate the visible values to sample the hidden values
        xk = sample(
            tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))  # Propagate the hidden values to sample the visible values
        return count + 1, k, xk

    # Run gibbs steps for k iterations
    ct = tf.constant(0)  # counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                   gibbs_step, [ct, tf.constant(k), x])
    # This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
    # optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample


### Run the graph!
# Now it's time to start a session and run the graph!
cell = tf.contrib.rnn.BasicLSTMCell(n_units)
initial_state = cell.zero_state(batch_size, tf.float32)

rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(cell, x,initial_state=initial_state, time_major=False)

rnn_outputs_on_last_t_step = tf.slice(
    rnn_outputs,
    [0, num_timesteps - (1 + eval_time_steps), 0],
    [batch_size, eval_time_steps, n_units])

final_projection = lambda z: tf.contrib.layers.linear(z, num_outputs=n_visible, activation_fn=tf.nn.sigmoid)

predicted = tf.map_fn(final_projection, rnn_outputs_on_last_t_step)

# Error and backprop
error = tf.nn.l2_loss(tf.subtract(tf.abs(y),tf.abs(predicted)))
train_step = tf.train.AdamOptimizer(lr).minimize(error)

# Prediction error and accuracy
accuracy = tf.reduce_mean(tf.subtract(tf.abs(y),tf.abs(predicted)))

if TRAINING_PHASE:
    with tf.Session() as sess:
        # First, we train the model
        # initialize the variables of the model
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        # Run through all of the training data num_epochs times
        for epoch in tqdm(range(num_epochs)):
            error_function = np.zeros(num_epochs)
            accuracy_function = np.zeros(num_epochs)
            learning_rate_array = np.zeros(num_epochs)
            batch = 0
            tr_x = []
            tr_y = []
            for song in songs:
                # The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
                song = np.array(song)
                song = song[:int(np.floor(song.shape[0] / num_timesteps) * num_timesteps)]
                song = np.reshape(song, [int(song.shape[0] / num_timesteps), int(song.shape[1] * num_timesteps)])
                # Train the RBM on batch_size examples at a time
                for i in range(0, len(song), num_timesteps):
                    if i+num_timesteps > len(song):
                        continue
                    sample = song[i:i + num_timesteps]
                    tr_x.append(sample)
                    tr_y.append(sample[-eval_time_steps:])
                    batch += 1
                if batch == batch_size:
                    break
            tr_x = np.array(tr_x)
            tr_y = np.array(tr_y)
            training_accuracy, prediction_error, _ = sess.run(
                    [accuracy,
                     error,
                     train_step],
                    feed_dict={x: tr_x, y: tr_y})
            error_function[epoch] = prediction_error
            accuracy_function[epoch] = training_accuracy
            print(" accuracy and prediction error: {}, {}".format(training_accuracy, prediction_error))
        saver.save(sess, getcwd()+'\\weight\\LSTM-weights', global_step=epoch)

        # Plots for network optimization at end of each epoch
        plt.subplot(311)
        plt.xlabel("Batch number")
        plt.ylabel("error")
        plt.plot(error_function)
        '''
        plt.subplot(312)
        plt.xlabel("Batch number")
        plt.ylabel("Accuracy: mean difference between data point")
        plt.plot(accuracy_function)
        plt.subplot(313)
        plt.xlabel("Batch number")
        plt.ylabel("Learning rate")
        plt.plot(learning_rate_array)
        '''
        plt.show()
    # Now the model is fully trained, so let's make some music!
    # Run a gibbs chain where the visible nodes are initialized to 0
    #sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((50, n_visible))})
else:
    with tf.Session() as sess:
        # First, we train the model
        # initialize the variables of the model
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.Saver().restore(sess, getcwd()+"\\weight\\LSTM-weights-199")
        tr_x = []
        tr_y = []
        batch = 0
        for song in songs:
            # The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
            # Here we reshape the songs so that each training example is a vector with num_timesteps x 2*note_range elements
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0] / num_timesteps) * num_timesteps)]
            song = np.reshape(song, [int(song.shape[0] / num_timesteps), int(song.shape[1] * num_timesteps)])
            # Train the RBM on batch_size examples at a time
            for i in range(0, len(song), num_timesteps):
                if i + num_timesteps > len(song):
                    continue
                sample = song[i:i + num_timesteps]
                tr_x.append(sample)
                tr_y.append(sample[-eval_time_steps:])
                batch += 1
            if batch == batch_size:
                break
        tr_x = np.array(tr_x)
        tr_y = np.array(tr_y)
        output = sess.run(
            predicted,
            feed_dict={x: tr_x})
        output = output.reshape(200,n_visible)
        for i in range(output.shape[0]):
            if not any(output[i, :]):
                continue
            # Here we reshape the vector to be time x notes, and then save the vector as a midi file

            S = np.reshape(output[i, :], (num_timesteps, 2 * note_range))
            F = []
            for array in S:
                pos = np.argmax(array)
                array = np.zeros( 2 * note_range)
                array[pos] = 1
                F.append(array)
            midi_manipulation.noteStateMatrixToMidi(F, "generated_chord_{}".format(i))
