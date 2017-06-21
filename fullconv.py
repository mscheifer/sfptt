seed = 3247238

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.set_random_seed(seed)

import load_books as lb
import shutil

np.set_printoptions(threshold=np.nan, 
    linewidth=shutil.get_terminal_size().columns)

import argparse

cmd_arg_def = argparse.ArgumentParser(description="Learn from text")

cmd_arg_def.add_argument("-d", "--debug", help=("Make it easier to catch errors"
    "in the tensorflow implementation"), action="store_true")

cmd_arg_def.add_argument("-e", "--epochs", metavar='number_of_epochs', type=int,
    default=100, help="Number of training epochs")

cmd_arg_def.add_argument("-l", "--learning_rate", type=float, default=0.001,
    help="Learning rate factor")

cmd_arg_def.add_argument("-b", "--base_dims", metavar='input_layer_channels',
    type=int, required=True, help="Number of dimensons of the first layer above"
    " the input")

cmd_arg_def.add_argument("-f", "--dim_scale", metavar='channel_scale_factor',
    type=float, required=True, help=("The factor of change of dimensons between"
    " layers. Should be between 1 and 2. Anything more than 2 will be trying to"
    " add information that isn't there. Anything less than 1 is probably losing"
    "  almost all of the information."))

cmd_arg_def.add_argument("-s", "--summaries", action="store_true",
    help="Whether to save weight summaries")

cmd_args = cmd_arg_def.parse_args()

# TODO: so what this should be is: Every layer produces len(target) values. The
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
    init = tf.orthogonal_initializer(gain=1.3)
    return tf.Variable(init(shape, tf.float32), name=name + "_weights")

bridge_out_size = 100

def make_bridge(conv, bridge_weights):

    ret = conv[::2] # odd values, keep the last value if total is odd
    ret = tf.matmul(ret, bridge_weights)
    if conv.shape[0] % 2 == 1:# Check the original, size of 'ret' is always odd
        last_odd_val = ret[-1:]
        ret = ret[:-1] # don't pad from the last odd value
    ret = tf.pad(tf.expand_dims(ret, 1), [[0,0], [0,1], [0,0]])
    # Pad with 1 zero at each value then reshape again so it's like we just put
    # 0s over all of the even values, but then have a broadcast dimension over
    # the chunk size
    ret = tf.reshape(ret, [-1, 1, bridge_out_size])
    if conv.shape[0] % 2 == 1:
        ret = tf.concat([ret, tf.expand_dims(last_odd_val, 0)], 0)
    assert ret.shape[0] == conv.shape[0], str(ret.shape) + str(conv.shape)
    return ret

def make_layer(lower_layer, num_output_channels, chunk_size):

    assert len(lower_layer.result_op.shape) == 2
    # First dimension is what we conv over, 2nd is number of channels
    # Btw, .value fixes a really stupid bug with xavier_init because it
    # doesn't auto convert the dimension to an integer
    num_input_channels = lower_layer.result_op.shape[1].value

    filter_weights = make_weights("filter",
        [2 * num_input_channels, num_output_channels])
    filt_bias = tf.Variable(tf.zeros([1, num_output_channels],
        dtype=tf.float32), name="filter_baises")

    assert lower_layer.result_op.shape[0] % 2 == 0

    filt = tf.matmul(tf.reshape(lower_layer.result_op, [-1,
        2 * num_input_channels]), filter_weights)

    conv = tf.tanh(tf.add(filt, filt_bias))

    assert conv.shape[0] != 0, "This layer isn't adding anything."

    num_results_is_odd = conv.shape[0] % 2 == 1

    # If we have an odd number of results, don't push the last one up. It
    # will be bridged.
    result_op = conv[0:-1] if num_results_is_odd else conv

    bridge_pieces = [lower_layer.bridge[:chunk_size - 1]]

    bridge_no_early_values = lower_layer.bridge[chunk_size - 1:]

    full_chunks = bridge_no_early_values.shape[0].value // chunk_size

    bridge_weight = make_weights("bridge",
        [num_output_channels, bridge_out_size])

    if full_chunks > 0:

        chunked_lower = tf.reshape(bridge_no_early_values[:full_chunks * chunk_size],
            [full_chunks, chunk_size, bridge_out_size])

        if full_chunks * chunk_size < bridge_no_early_values.shape[0]:
            bridge_conv = conv[0:-1]
        else: # For the corner case where there are exactly chunk_size - 1
            # values after the last chunk
            assert full_chunks * chunk_size == bridge_no_early_values.shape[0]
            bridge_conv = conv

        w_br = chunked_lower + make_bridge(bridge_conv, bridge_weight)
        bridge_pieces.append(tf.reshape(w_br, [-1, bridge_out_size]))

    if full_chunks * chunk_size < bridge_no_early_values.shape[0]:
        # slice off the remainder
        final_bridge = bridge_no_early_values[full_chunks * chunk_size:]
        if num_results_is_odd:
            final_bridge = final_bridge + tf.matmul(conv[-1:], bridge_weight)
        # If it's even, don't use it because it is part of the upper layer value

        bridge_pieces.append(final_bridge)

    class Layer:
        def __init__(self):
            self.result_op = result_op
            self.bridge = tf.concat(bridge_pieces, 0)
            assert self.bridge.shape == lower_layer.bridge.shape, str(
                self.bridge.shape) + str(lower_layer.bridge.shape)

    return Layer()

import math

lowest_level_memory_size = 10

max_len = max(len(book) for book in lb.train_books)

class Input:
    def __init__(self):
        self.input_seq = tf.placeholder(shape=[max_len - 1,
            len(lb.character_set)], dtype=tf.float32, name="input_layer")

        bridge_weight = make_weights("bridge",
            [len(lb.character_set), bridge_out_size])

        if self.input_seq.shape[0] % 2 == 0:
            self.result_op = self.input_seq 
        else:
            self.result_op = self.input_seq[:-1]

        self.bridge = tf.reshape(make_bridge(self.input_seq, bridge_weight),
            [-1, bridge_out_size])
        self.max_layer_size = self.result_op.shape[0]

with tf.name_scope("input"):
    layer = Input()
input_layer = layer

layer_channels = cmd_args.base_dims

memory_scale_factor = cmd_args.dim_scale

# Floor because we want a power of 2 less than max_len. If we took a power of 2
# greater than max_len then we would never use that value (because we would
# never have enough input).
num_layers = int(math.floor(math.log2(max_len)))
print("Max book length", max_len, "so number of layers is", num_layers)

for i in range(num_layers):
    # Chunk size should be the number of input characters represented by each
    # output vector of the layer we are about to create
    chunk_size = 2 ** (i + 1)

    with tf.name_scope("layer_" + str(i)):
        layer = make_layer(layer, layer_channels, chunk_size)
        print("Layer", i, "has", layer_channels, "channels and",
            layer.result_op.shape[0], "values to pass up to the next layer.")

    layer_channels = int(math.ceil(layer_channels * memory_scale_factor))

first_bridge_bias = tf.Variable(tf.zeros([1, bridge_out_size],
    dtype=tf.float32), name="first_bridge_baises")

first_bridge = tf.tanh(layer.bridge + first_bridge_bias)

layer_sizes = [75, 50]

output = first_bridge
for size in layer_sizes:
    output = tf.layers.dense(
        # Expand dims to make a fake mini-batch
        inputs = output,
        units = size,
        activation = tf.tanh,
        kernel_initializer = tf.orthogonal_initializer(),
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

target = tf.placeholder(tf.int32, shape=[max_len - 1])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels = target, logits = output)

optimizer = tf.train.AdamOptimizer(learning_rate=cmd_args.learning_rate)

grads_and_vars = [(g, v) for g, v in optimizer.compute_gradients(loss) if g is not None]

if cmd_args.debug:
    grads = [tf.check_numerics(g, "grr") for g, _ in grads_and_vars]
else:
    grads = [g for g, _ in grads_and_vars]

train_vars = [v for _, v in grads_and_vars]

acc_gs = [tf.Variable(tf.fill(g.shape, np.float32(0)), trainable=False,
    name=v.name.translate(str.maketrans("/:", "__")) + "_grad")
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

accumulate_gradients = [ag.assign_add(g / len(lb.train_books)) 
    for ag, g in zip(acc_gs, grads)]
apply_grads = optimizer.apply_gradients(zip(acc_gs, train_vars))

book_loss_op = tf.reduce_sum(loss)

import shutil

print_width=shutil.get_terminal_size().columns

import re

def train_book(sess, book):

    print("Book length:", len(book))

    book = book + (' ' * (max_len - len(book))) # padding

    inj = { target : [lb.char_indices[next_char] for next_char in book[1:]],
        input_layer.input_seq : lb.one_hot(book[:-1]) }

    book_loss, predict, _ = sess.run(
        [book_loss_op, output, accumulate_gradients], inj)

    predict_book = "".join(lb.index_chars[np.argmax(p)] for p in predict[-print_width:])

    def simplify_whitespace(s):
        return re.sub(r"\s", " ", s)

    print(simplify_whitespace(book[-print_width:]))
    print(simplify_whitespace(book[0] + predict_book 
        if len(predict_book) < print_width else predict_book))

    return book_loss

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    lowest_loss_so_far = math.inf

    epochs = cmd_args.epochs

    for epoch in range(epochs):
        print("\nEpoch", epoch, "\n")

        start_time = time.time()

        sess.run([acc_g.initializer for acc_g in acc_gs])

        epoch_loss = 0

        for book in lb.train_books:
            epoch_loss += train_book(sess, book)

        total_characters = sum(len(book) for book in lb.train_books)

        print("Loss per character:", epoch_loss / total_characters, end=' ')

        if epoch_loss < lowest_loss_so_far:
            lowest_loss_so_far = epoch_loss
            print("which is the best so far")
        else:
            print() # new line

        sess.run(apply_grads)

        #epochs - 1 is last one
        e = epoch
        if cmd_args.summaries and (e % (epochs / 20) == 0 or e == epochs - 1):
            print("Summarizing epoch:", e)
            write_summaries.add_summary(sess.run(all_summaries), epoch)

        print("Epoch took:", time.time() - start_time, "seconds")

if cmd_args.summaries:
    write_summaries.close() # file doesn't auto close when the process exits...
