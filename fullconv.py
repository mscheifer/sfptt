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

def with_assert(op_factory, assert_factory):
    if cmd_args.debug:
        with tf.control_dependencies(assert_factory()):
            return op_factory()
    return op_factory()

def extract_singleton(op):
    def assert_one_element():
        fst_dim = tf.shape(op)[0]
        return [tf.Assert(tf.equal(fst_dim, 1), [fst_dim])]
    return with_assert(lambda: op[0], assert_one_element)

def is_odd(op):
    return tf.equal(op % 2, 1)

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

def make_conv_op(input, filter_weights, filt_bias):

    assert len(input.shape) == 2
    assert len(filter_weights.shape) == 3
    filt = tf.matmul(tf.reshape(input, [-1, 2 * input.shape[1].value]),
            tf.reshape(filter_weights, [-1, filter_weights.shape[-1].value]))

    return tf.tanh(tf.add(filt, filt_bias))

def make_weights(name, shape):
    init = tf.orthogonal_initializer(gain=1.3)
    return tf.Variable(init(shape, tf.float32), name=name + "_weights")

def make_bridge(conv, bridge_weights):

    ret = conv[::2] # every other value, keep the last value if odd
    ret = tf.matmul(ret, bridge_weights)
    if conv.shape[0] % 2 == 1: # Check the original, not 'ret' b/c even/2 is odd
        last_odd_val = ret[-1:]
        ret = ret[:-1] # don't pad from the last odd value
    ret = tf.pad(tf.expand_dims(ret, 0), [[0,1], [0,0], [0,0]])
    # Pad with 1 zero at each value then reshape again so it's like we just put
    # 0s over all of the even values, but then have a broadcast dimension over
    # the chunk size
    ret = tf.reshape(ret, [-1, 1, len(lb.character_set)])
    if conv.shape[0] % 2 == 1:
        ret = tf.concat([ret, tf.expand_dims(last_odd_val, 0)], 0)
    assert ret.shape[0] == conv.shape[0], str(ret.shape) + str(conv.shape)
    return ret

def make_simple_layer(lower_layer, num_output_channels, chunk_size):

    # First dimension is what we conv over, 2nd is number of channels
    # Btw, .value fixes a really stupid bug with xavier_init because it
    # doesn't auto convert the dimension to an integer
    num_input_channels = lower_layer.result_op.get_shape()[1].value

    filter_weights = make_weights("filter",
        [2, num_input_channels, num_output_channels])
    filt_bias = tf.Variable(tf.zeros([1, num_output_channels],
        dtype=tf.float32), name="filter_baises")

    conv = make_conv_op(lower_layer.result_op, filter_weights, filt_bias)

    num_results_is_odd = conv.shape[0] % 2 == 1

    # If we have an odd number of results, don't push the last one up. It
    # will be bridged.
    result_op = conv[0:-1] if num_results_is_odd else conv

    assert lower_layer.max_layer_size % 2 == 0
    max_layer_size = (lower_layer.max_layer_size // 2)
    # max layer size is when there is no bridge. We can actually set the max
    # size to be more than it will ever be in order to fulfill that definition
    # because layers are always allowed to have fewer results than their max.
    if max_layer_size % 2 == 1:
        max_layer_size += 1

    assert max_layer_size % 2 == 0, str((lower_layer.max_layer_size, max_layer_size))

    bridge_pieces = [lower_layer.bridge[:chunk_size - 1]]

    bridge_no_early_values = lower_layer.bridge[chunk_size - 1:]

    full_chunks = bridge_no_early_values.shape[0].value // chunk_size

    bridge_weight = make_weights("bridge",
        [num_output_channels, len(lb.character_set)])

    if full_chunks > 0:

        chunked_lower = tf.reshape(bridge_no_early_values[:full_chunks * chunk_size],
            [full_chunks, chunk_size, len(lb.character_set)])

        print(conv.shape[0], lower_layer.bridge.shape[0], full_chunks * chunk_size, bridge_no_early_values.shape)
        if full_chunks * chunk_size < bridge_no_early_values.shape[0]:
            bridge_conv = conv[0:-1]
        else: # For the corner case where there are exactly chunk_size - 1
            # values after the last chunk
            assert full_chunks * chunk_size == bridge_no_early_values.shape[0]
            bridge_conv = conv

        w_br = chunked_lower + make_bridge(bridge_conv, bridge_weight)
        bridge_pieces.append(tf.reshape(w_br, [-1, len(lb.character_set)]))

    if full_chunks * chunk_size < bridge_no_early_values.shape[0]:
        # slice off the remainder
        final_bridge = bridge_no_early_values[full_chunks * chunk_size:]
        if num_results_is_odd:
            final_bridge = final_bridge + tf.matmul(conv[-1:], bridge_weight)
        # If it's even, don't use it because it is part of the upper layer value

        bridge_pieces.append(final_bridge)

    class Layer:
        def __init__(self):
            self.max_layer_size = max_layer_size
            self.result_op = result_op
            print([str(piece.shape) for piece in bridge_pieces])
            self.bridge = tf.concat(bridge_pieces, 0)
            assert self.bridge.shape == lower_layer.bridge.shape, str(self.bridge.shape) + str(lower_layer.bridge.shape)

    return Layer()

def make_layer(lower_layer, num_output_channels, chunk_size, max_outputs_for_dataset):

    assert lower_layer.max_layer_size % 2 == 0
    max_conv_values = lower_layer.max_layer_size // 2

    # if the convolution values and the bridge are enough to cover the entire
    # input length then we can create a layer that doesn't need constants
    assert max_conv_values >= max_outputs_for_dataset

    return make_simple_layer(lower_layer, num_output_channels, chunk_size)

import math

lowest_level_memory_size = 10

max_len = max(len(book) for book in lb.train_books)
# TODO: can this be dynamic and grow with more input?
num_layers = int(math.ceil(math.log2(max_len)))
print("Max book length", max_len, "so number of layers is", num_layers)

class Input:
    def __init__(self):
        bridge_weight = make_weights("bridge",
             [len(lb.character_set), len(lb.character_set)])
        self.result_op = tf.placeholder(shape=[max_len, len(lb.character_set)],
            dtype=tf.float32, name="input_layer")
        self.bridge = tf.reshape(make_bridge(self.result_op, bridge_weight), [-1, len(lb.character_set)])
        self.max_layer_size = self.result_op.shape[0]

with tf.name_scope("input"):
    layer = Input()
input_layer = layer

layer_channels = cmd_args.base_dims

memory_scale_factor = cmd_args.dim_scale

for i in range(num_layers):
    # Chunk size should be the number of input characters represented by each
    # output vector of the layer we are about to create
    chunk_size = 2 ** (i + 1)
    # max_chunks is how many we need to represent every character in the
    # largest example in our dataset, rounded up.
    max_chunks = max_len // chunk_size + (0 if max_len % chunk_size == 0 else 1)

    with tf.name_scope("layer_" + str(i)):
        layer = make_layer(layer, layer_channels, chunk_size, max_chunks)
        print("Layer", i, "has", layer_channels, "channels and",
            layer.max_layer_size, "values.")

    layer_channels = int(math.ceil(layer_channels * memory_scale_factor))

print("The final layer says size 2 but only 1 value will ever be computed.",
    "It just says 2 because every layer has to have a positive even max size")

output = layer.bridge + tf.Variable(tf.zeros([1, len(lb.character_set)],
    dtype=tf.float32), name="output_baises")

target = tf.placeholder(tf.int32, shape=[max_len])

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

num_predictions = sum(len(book) - 1 for book in lb.train_books)

# dividing gradients before summing should prevent overflows here
accumulate_gradients = [ag.assign_add(g / num_predictions) for ag, g in zip(acc_gs, grads)]
apply_grads = optimizer.apply_gradients(zip(acc_gs, train_vars))

import collections
import re

def train_book(sess, book):

    num_top_to_print = 3

    predict_buffers = []
    for _ in range(num_top_to_print):
        predict_buffer = collections.deque(maxlen = 80)
        predict_buffer.append(book[0]) # we don't actually predict the first char
        predict_buffers.append(predict_buffer)

#    print("Book length:", len(book), "\n", "\n" * len(predict_buffers))

    inj = { target : [lb.char_indices[next_char] for next_char in book[1:]],
        input_layer.result_op : lb.one_hot(book[:-1]) }

    book_loss, predict, _ = sess.run(
        [loss, output, accumulate_gradients], inj)

    def simplify_whitespace(s):
        return re.sub(r"\s", " ", s)

    # partition softmax probabilities to get indices for top N for N buffers
    # Negate the predictions so we get the highest values rather than lowest
#    topP = np.argpartition(-predict, len(predict_buffers))[
#        0:len(predict_buffers)]
    # top N aren't sorted so we need to sort them. This looks weird but we
    # can index into the top N with the sorted indices of those N and it
    # will give us indices back into the original vector
#    topP = topP[np.argsort(-predict[topP])]

#    for p_char, buffer in zip(topP, predict_buffers):
#        buffer.append(simplify_whitespace(lb.index_chars[p_char]))

#    print("\033[F", "\033[F" * len(predict_buffers),
#        simplify_whitespace(book[-80:]), sep='', end=' ')
#    for predict_buffer in predict_buffers:
#        print("".join(predict_buffer))

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
