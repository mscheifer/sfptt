from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import argparse

cmd_arg_def = argparse.ArgumentParser(description="Learn from text")

cmd_arg_def.add_argument("-sd", "--seed", type=int, default = 3247238,
    help=("Seed value for randomness"))

cmd_arg_def.add_argument("-d", "--debug", help=("Make it easier to catch errors"
    "in the tensorflow implementation"), action="store_true")

cmd_arg_def.add_argument("-e", "--epochs", metavar='number_of_epochs', type=int,
    default=100, help="Number of training epochs")

cmd_arg_def.add_argument("-l", "--learning_rate", type=float, default=0.001,
    help="Learning rate factor")

cmd_arg_def.add_argument("-ds", "--dim_scale", metavar='channel_scale_factor',
    type=float, required=True, help=("The factor of change of dimensons between"
    " layers. Should be between 1 and 2. Anything more than 2 will be trying to"
    " add information that isn't there. Anything less than 1 is probably losing"
    "  almost all of the information."))

cmd_arg_def.add_argument("-s", "--summaries", action="store_true",
    help="Whether to save weight summaries")

cmd_arg_def.add_argument("-o", "--open", help="Chekpoint file to load.")

cmd_arg_def.add_argument("-f", "--forward", help="Initial string for forward inference.")

cmd_arg_def.add_argument("-tp", "--temperature", type=float, default=1.0,
    help="Sampling temperature. Lower is more conservative.")

cmd_arg_def.add_argument("-tr", "--training_directory", required=True,
    help="Directory of training data.")

cmd_arg_def.add_argument("-ch", "--checkpoint_directory", required=True,
    help="Directory of checkpoints.")

cmd_args = cmd_arg_def.parse_args()

seed = cmd_args.seed

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.set_random_seed(seed)

del seed

import shutil

try:
    print_width = shutil.get_terminal_size().columns
except AttributeError: # We're in Python 2
    print_width = 80
np.set_printoptions(threshold=np.nan, linewidth=print_width)

import src.load_books
lb = src.load_books.load_training_data(cmd_args.training_directory)

# What this is is: Every layer produces len(target) values. The
# upper layer takes those values and half of them are linearly translated up
# and the other half are linearly combined with the odd value at that layer and
# then tanh'd. The dimensions for the elements in both of those halves are the
# same.

# For the bridge values. Gotta combine it into one multiply somehow. Take each
# value and bridge it across with the half of the pariod size values that need
# it. The other half of the period gets 0 ? Or just isn't bridged. For this
# full conv version it seems like it makes sense to try to generate a timewise
# big conv multiply but for the original version a layerwise big dense layer
# is starting to look attractive to me again, as long as I zero out values that
# are not bridged (to avoid the structural looking-the-same problem I ran into
# earlier). Maybe just doing a layerwise multiply at each timestep is fine for
# this version too.

# Ok, so I'm thinking of basing this on the single linear brigde to output
# layer with 0s for non bridged layers. We can do this with timewise
# convolutions. So take every other layer result and interleave it with 0s.
# Then add it to the result from the lower layer, broadcasting the add. So each
# layer passes up a matrix that is [time, output_size] and each layer just adds
# to it. Then at the end we add a bias and do unscaled logit cross entropy. For
# the broadcasted adds, I think we can reshape and then reshape back to get it
# to broadcast how we want. So reshape it to [time / chunk_size, chunk_size,
# output_size] then broadcast our values that are [time / chunk_size, 1,
# output_size]. Then the reshape and return up. There will be remainder values
# from the division and we have to make sure those get the correct value too.

def make_weights(name, shape):
    init = tf.orthogonal_initializer(gain=1.3) # 1.3 is for tanh
    return tf.Variable(init(shape, tf.float32), name=name + "_weights")

bridge_out_size = 500

def make_bridge(conv, bridge_weights):

    conv_size = tf.shape(conv)[0]

    conv_is_odd = tf.equal(conv_size % 2, 1)
    # Boolean to disambiguate if this value is part of an upper layer pair or is a single
    is_included_in_upper_layer = tf.tile([0.0,1.0], [conv_size // 2])

    is_included_in_upper_layer = tf.cond(conv_is_odd,
        lambda: tf.concat([is_included_in_upper_layer, tf.zeros([1], tf.float32)], axis=0),
        lambda: is_included_in_upper_layer)

    with_disambiguation = tf.concat([conv, tf.expand_dims(is_included_in_upper_layer, 1)], axis=1)

    ret = tf.matmul(with_disambiguation, bridge_weights)
    # reshape to add broadcast dimension
    return tf.reshape(ret, [-1, 1, bridge_out_size])

def make_layer(lower_layer, num_output_channels, chunk_size, bridge_weight):

    assert len(lower_layer.result_op.shape) == 2
    # First dimension is what we conv over, 2nd is number of channels
    # Btw, .value fixes a really stupid bug with xavier_init because it
    # doesn't auto convert the dimension to an integer
    num_input_channels = lower_layer.result_op.shape[1].value

    filter_weights = make_weights("filter",
        [2 * num_input_channels, num_output_channels])
    filt_bias = tf.Variable(tf.zeros([1, num_output_channels],
        dtype=tf.float32), name="filter_baises")

    filt = tf.matmul(tf.reshape(lower_layer.result_op, [-1,
        2 * num_input_channels]), filter_weights)

    conv = tf.tanh(tf.add(filt, filt_bias))

    bridge_pieces = [lower_layer.bridge[:chunk_size - 1]]

    bridge_no_early_values = lower_layer.bridge[chunk_size - 1:]

    # subtract 1 here because if there's exactly chunk_size -1 values
    # after the last conv value then we want to handle that in the
    # 3rd bridge piece
    full_chunks = (tf.shape(bridge_no_early_values)[0] - 1) // chunk_size

    chunked_lower = tf.reshape(bridge_no_early_values[:full_chunks * chunk_size],
        [full_chunks, chunk_size, bridge_out_size])

    bridge_conv = make_bridge(conv, bridge_weight)

    w_br = chunked_lower + bridge_conv[0:-1]
    bridge_pieces.append(tf.reshape(w_br, [-1, bridge_out_size]))

    # slice off the remainder
    final_bridge = bridge_no_early_values[full_chunks * chunk_size:]
    # If num results is even, don't use the conv value because it is part of the
    # upper layer value
    final_bridge = final_bridge + tf.reshape(bridge_conv[-1:], [-1, bridge_out_size])

    bridge_pieces.append(final_bridge)

    class Layer:
        def __init__(self):
            # Only push up an even number of results to be convolved over. Drop
            # the most recent one if odd
            self.result_op = tf.cond(tf.equal(tf.shape(conv)[0] % 2, 1),
                lambda: conv[0:-1], lambda: conv)
            self.bridge = tf.concat(bridge_pieces, 0)

    return Layer()

max_initial_newlines = 5

max_len = max(len(book) for book in lb.train_books) + max_initial_newlines

class Input:
    def __init__(self, bridge_weight):
        self.input_seq = tf.placeholder(shape=[None], dtype=tf.int32,
            name="input_logits")

        one_hots = tf.one_hot(self.input_seq, len(lb.character_set))

        self.result_op = tf.cond(tf.equal(tf.shape(one_hots)[0] % 2, 1),
            lambda: one_hots[0:-1], lambda: one_hots)

        self.bridge = tf.reshape(make_bridge(one_hots, bridge_weight),
             [-1, bridge_out_size])

import math

# Floor because we want a power of 2 less than max_len. If we took a power of 2
# greater than max_len then we would never use that value (because we would
# never have enough input).
# TODO: can use math.log2 if we drop python2 compatability
num_layers = int(math.floor(math.log(max_len, 2)))
print("Max book length", max_len, "so number of layers is", num_layers)

memory_scale_factor = cmd_args.dim_scale

base_channels = len(lb.character_set)

layer_channels = [int(math.ceil(base_channels * (memory_scale_factor ** (i + 1))))
    for i in range(num_layers)]

# add one more input element for each layer and the input layer for the
# disabiguation boolean
bridge_weight = make_weights("bridge", [len(lb.character_set) + 1 +
    sum(layer_channels) + num_layers, bridge_out_size])

with tf.name_scope("input"):
    layer = Input(bridge_weight[:len(lb.character_set) + 1])
input_layer = layer

for i in range(num_layers):
    # Chunk size should be the number of input characters represented by each
    # output vector of the layer we are about to create
    chunk_size = 2 ** (i + 1)

    prev_ch = len(lb.character_set) + 1 + sum(layer_channels[:i]) + i
    br_weight_part = bridge_weight[prev_ch : prev_ch + layer_channels[i] + 1]

    with tf.name_scope("layer_" + str(i)):
        layer = make_layer(layer, layer_channels[i], chunk_size, br_weight_part)
        print("Layer", i, "has", layer_channels[i], "channels and",
            layer.result_op.shape[0], "values to pass up to the next layer.")

first_bridge_bias = tf.Variable(tf.zeros([1, bridge_out_size],
    dtype=tf.float32), name="first_bridge_baises")

first_bridge = tf.tanh(layer.bridge + first_bridge_bias)

layer_sizes = [500, 200]

output = first_bridge
for size in layer_sizes:
    output = tf.layers.dense(
        # Expand dims to make a fake mini-batch
        inputs = output,
        units = size,
        activation = tf.tanh,
        kernel_initializer = tf.orthogonal_initializer(gain=1.3),
        # should default to 0 initialized bias
        name="output_" + str(size))

output = tf.layers.dense(
    # Expand dims to make a fake mini-batch
    inputs = output,
    units = len(lb.character_set),
    # linear activation function here because we softmax as part of the loss
    activation = None,
    kernel_initializer = tf.orthogonal_initializer(),
    # should default to 0 initialized bias
    name="output_characters")

next_char = tf.nn.softmax(output[-1] / cmd_args.temperature)

target = tf.placeholder(tf.int32, shape=[None])

l = lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels = target, logits = output)
if cmd_args.debug:
    with tf.Assert(tf.shape(target)[0] < max_len):
        loss = l()
else:
    loss = l()
del l

optimizer = tf.train.AdamOptimizer(learning_rate=cmd_args.learning_rate,
    epsilon=0.01) # using epsilon to clip gradients. Hopefully that will
                  # help keep values from saturating.

grads_and_vars = optimizer.compute_gradients(loss)

if cmd_args.debug:
    grads = [tf.check_numerics(g, "grr") for g, _ in grads_and_vars]
else:
    grads = [g for g, _ in grads_and_vars]

train_vars = [v for _, v in grads_and_vars]

translator = dict((ord(char), "_") for char in "/:")

acc_gs = [tf.Variable(tf.fill(g.shape, np.float32(0)), trainable=False,
    name=v.name.translate(translator) + "_grad")
    for g, v in grads_and_vars]

del grads_and_vars

import time

if cmd_args.summaries:

    summary_dir = "./summaries/" + str(time.time())
    write_summaries = tf.summary.FileWriter(summary_dir)
    print("Summary dir:", summary_dir)

    for v in train_vars:
        tf.summary.histogram(v.name + "_hist", v)

    for acc_g in acc_gs:
        tf.summary.histogram(acc_g.name + "_hist", acc_g)

    all_summaries = tf.summary.merge_all()
    assert all_summaries is not None

accumulate_gradients = [ag.assign_add(g) for ag, g in zip(acc_gs, grads)]

total_minibatch_characters = tf.placeholder(tf.int32, shape=[])

apply_grads = optimizer.apply_gradients(zip((ag / tf.cast(
    total_minibatch_characters, tf.float32) for ag in acc_gs), train_vars))

book_loss_op = tf.reduce_sum(loss)

import re

def train_book(sess, orig_book):

#    print("Book length:", len(orig_book))
    # Adding some newlines at the start should help it be more translationally
    # invariant
    perturb_len = random.randint(0, 5)
    book = ("\n" * perturb_len) + orig_book

    inj = { target : [lb.char_indices[next_char] for next_char in book[1:]],
        input_layer.input_seq : lb.logits(book[:-1], np.int32) }

    book_loss, predict, _ = sess.run(
        [book_loss_op, output, accumulate_gradients], inj)

    if len(orig_book) > print_width:
        front_len = print_width // 2
        if print_width % 2 == 1: front_len += 1

        predict_book = book[0] + "".join(lb.index_chars[np.argmax(p)]
            for p in np.concatenate([
                predict[perturb_len:perturb_len + front_len - 1],
                predict[-print_width // 2:]]))

        assert len(predict_book) == print_width, len(predict_book)

        orig_to_print = orig_book[:front_len] + orig_book[-print_width // 2:]
        assert len(orig_to_print) == print_width, (
            "Make sure '+' is concatenate lists, not broadcast add")
    else:
        predict_book = book[0] + "".join(lb.index_chars[np.argmax(p)]
            for p in predict[perturb_len:])
        orig_to_print = orig_book

    def simplify_whitespace(s):
        return re.sub(r"\s", " ", s)

#    print(simplify_whitespace(orig_to_print))
#    print(simplify_whitespace(predict_book))

    return book_loss

checkpoint_dir = cmd_args.checkpoint_directory + "/"

import sys

# grr, Google why are you making me use Python 2.7 ?
def print_flushed(*strs, **kwargs):
    print(*strs, sep=kwargs.get("sep", ' '), end=kwargs.get("end", '\n'))
    sys.stdout.flush()

print_flushed("Initializing checkpoint saver...", end=' ')
saver = tf.train.Saver() # Buy default saves all variables, including stuff
# like beta power accumulators in the Optimizer

#import os # grr, disabling until I de-support python2
#os.makedirs(checkpoint_dir, exist_ok=True)
print("done.")

#TODO: ok so deliberately stop propagating gradients through some paths to see
# if initialization is still helping. So like don't propagate through the
# bridge except the top layer and then see what magnitude of gradient we get at
# the bottom layer.

book_minibatch_size = 10

def runforward(sess):
    n_char_probs = sess.run(next_char,
        { input_layer.input_seq : lb.logits(text, np.int32) })

    return lb.index_chars[np.argmax(n_char_probs)]

    f = np.random.uniform()
    prob_sum = 0
    for idx, prob in enumerate(n_char_probs):
        prob_sum += prob
        if f <= prob_sum:
            return lb.index_chars[idx]
    return lb.index_chars[-1] # might happen if what the NN gives us isn't a
    # perfect probability distribution, which it can be due to rounding errors

with tf.Session() as sess:

    print("Session start")

    if cmd_args.open is None:
        to_load_from = tf.train.latest_checkpoint(checkpoint_dir)
    else:
        to_load_from = cmd_args.open

    if to_load_from is not None:
        print_flushed("Loading from:", to_load_from, "...", end=' ')
        saver.restore(sess, to_load_from)
    else:
        print_flushed("No checkpoints, initializing for the first time...", end=' ')
        # TODO: trace this and see why it is using so much memory and time. To
        # point that the process is killed by the OS. (Probably the orthogonal
        # initializizer)
        sess.run(tf.global_variables_initializer())

        save_path = saver.save(sess, checkpoint_dir + "save", global_step=0)
    print("done.")

    if cmd_args.forward is not None:
        text = cmd_args.forward
        print(text, end='')
        while True:
            n_char = runforward(sess)
            print_flushed(n_char, end='')
            text += n_char

    lowest_loss_so_far = float('inf') # grr, 3.5 has 'math.inf'

    epochs = cmd_args.epochs

    plateaus = [[0,0]]

    total_predictions = sum(len(book) - 1 for book in lb.train_books)

    for epoch in range(epochs):
        print("\nEpoch", epoch, "\n")

        start_time = time.time()

        epoch_loss = 0

        random.shuffle(lb.train_books)

        mini_batches = [lb.train_books[pos : pos + book_minibatch_size]
            for pos in range(0, len(lb.train_books), book_minibatch_size)]

        for idx, book_batch in enumerate(mini_batches):
            sess.run([acc_g.initializer for acc_g in acc_gs])
            for book in book_batch:
                epoch_loss += train_book(sess, book)
            total_cs = sum(len(book) for book in book_batch)
            sess.run(apply_grads, { total_minibatch_characters : total_cs})
#            print((idx + 1) / len(mini_batches) * 100, "% through the epoch.")

        loss_per_character = epoch_loss / total_predictions

        print("Loss per character:", loss_per_character, end=' ')

        if loss_per_character < lowest_loss_so_far:
            diff = lowest_loss_so_far - loss_per_character
            lowest_loss_so_far = loss_per_character
            if plateaus[0][0] > 0: plateaus.insert(0, [0, diff])
            print("which is the best so far")
            print("Saved best to:", saver.save(sess, checkpoint_dir + "best"))
        else:
            plateaus[0][0] += 1
            plateau_str = ("{} {:.1e}".format(*plateau) for plateau in plateaus)
            print("which isn't better than: ", lowest_loss_so_far, *plateau_str)

        save_path = saver.save(sess, checkpoint_dir + "save", global_step=epoch)
        print("Saved to:", save_path)

        #epochs - 1 is last one
        e = epoch
        if cmd_args.summaries and (e % (epochs / 20) == 0 or e == epochs - 1):
            print("Summarizing epoch:", e)
            write_summaries.add_summary(sess.run(all_summaries), epoch)

        print("Epoch took:", time.time() - start_time, "seconds")

if cmd_args.summaries:
    write_summaries.close() # file doesn't auto close when the process exits...
