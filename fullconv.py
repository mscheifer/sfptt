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

cmd_arg_def.add_argument("-ds", "--dim_scale", metavar='channel_scale_factor',
    type=float, required=True, help=("The factor of change of dimensons between"
    " layers. Should be between 1 and 2. Anything more than 2 will be trying to"
    " add information that isn't there. Anything less than 1 is probably losing"
    "  almost all of the information."))

cmd_arg_def.add_argument("-s", "--summaries", action="store_true",
    help="Whether to save weight summaries")

cmd_arg_def.add_argument("-o", "--open", help="Chekpoint file to load.")

cmd_arg_def.add_argument("-f", "--forward", help="Initial string for forward inference.")

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
    init = tf.orthogonal_initializer(gain=1.3) # 1.3 is for tanh
    return tf.Variable(init(shape, tf.float32), name=name + "_weights")

bridge_out_size = 150

def make_bridge(conv, bridge_weights):

    ret = conv[::2] # odd values, keep the last value if total is odd
    ret = tf.matmul(ret, bridge_weights)

    last_odd_val = ret[-1:]
    conv_is_odd = tf.equal(tf.shape(conv)[0] % 2, 1)

    # Check the original, size of 'ret' is always odd
    # don't pad from the last odd value
    ret = tf.cond(conv_is_odd, lambda: ret[:-1], lambda: ret)

    ret = tf.pad(tf.expand_dims(ret, 1), [[0,0], [0,1], [0,0]])
    # Pad with 1 zero at each value then reshape again so it's like we just put
    # 0s over all of the even values, but then have a broadcast dimension over
    # the chunk size
    ret = tf.reshape(ret, [-1, 1, bridge_out_size])

    ret = tf.cond(conv_is_odd,
        lambda: tf.concat([ret, tf.expand_dims(last_odd_val, 0)], 0),
        lambda: ret)
    return ret

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

    bridge_conv = conv[0:-1]

    w_br = chunked_lower + make_bridge(bridge_conv, bridge_weight)
    bridge_pieces.append(tf.reshape(w_br, [-1, bridge_out_size]))

    num_results_is_odd = tf.equal(tf.shape(conv)[0] % 2, 1)

    #TODO: would it be faster to multiply the bridge value by
    # conv.shape[0] % 2 (so 0 when even) rather than do the conditional?

    # slice off the remainder
    final_bridge = bridge_no_early_values[full_chunks * chunk_size:]
    # If num results is even, don't use the conv value because it is part of the
    # upper layer value
    final_bridge = tf.cond(num_results_is_odd,
        lambda: final_bridge + tf.matmul(conv[-1:], bridge_weight),
        lambda: final_bridge)

    bridge_pieces.append(final_bridge)

    class Layer:
        def __init__(self):
            # If we have an odd number of results, don't push the last one up. It
            # will be bridged.
            self.result_op = tf.cond(num_results_is_odd,
                lambda: conv[0:-1], lambda: conv)
            self.bridge = tf.concat(bridge_pieces, 0)

    return Layer()

max_initial_newlines = 5

max_len = max(len(book) for book in lb.train_books) + max_initial_newlines

class Input:
    def __init__(self, bridge_weight):
        self.input_seq = tf.placeholder(shape=[None,
            len(lb.character_set)], dtype=tf.float32, name="input_layer")

        self.result_op = tf.cond(tf.equal(tf.shape(self.input_seq)[0] % 2, 1),
            lambda: self.input_seq[0:-1], lambda: self.input_seq)

        self.bridge = tf.reshape(make_bridge(self.input_seq, bridge_weight),
            [-1, bridge_out_size])

import math

# Floor because we want a power of 2 less than max_len. If we took a power of 2
# greater than max_len then we would never use that value (because we would
# never have enough input).
num_layers = int(math.floor(math.log2(max_len)))
print("Max book length", max_len, "so number of layers is", num_layers)

memory_scale_factor = cmd_args.dim_scale

base_channels = cmd_args.base_dims

layer_channels = [int(math.ceil(base_channels * (memory_scale_factor ** i)))
    for i in range(num_layers)]

bridge_weight = make_weights("bridge",
    [len(lb.character_set) + sum(layer_channels), bridge_out_size])

with tf.name_scope("input"):
    layer = Input(bridge_weight[:len(lb.character_set)])
input_layer = layer

for i in range(num_layers):
    # Chunk size should be the number of input characters represented by each
    # output vector of the layer we are about to create
    chunk_size = 2 ** (i + 1)

    prev_ch = len(lb.character_set) + sum(layer_channels[:i])
    bridge_weight_piece = bridge_weight[prev_ch : prev_ch + layer_channels[i]]

    with tf.name_scope("layer_" + str(i)):
        layer = make_layer(layer, layer_channels[i], chunk_size,
            bridge_weight_piece)
        print("Layer", i, "has", layer_channels[i], "channels and",
            layer.result_op.shape[0], "values to pass up to the next layer.")

first_bridge_bias = tf.Variable(tf.zeros([1, bridge_out_size],
    dtype=tf.float32), name="first_bridge_baises")

first_bridge = tf.tanh(layer.bridge + first_bridge_bias)

layer_sizes = [100, 75]

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

next_char = tf.nn.softmax(output[-1])

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

accumulate_gradients = [ag.assign_add(g) for ag, g in zip(acc_gs, grads)]

total_minibatch_characters = tf.placeholder(tf.int32, shape=[])

apply_grads = optimizer.apply_gradients(zip((ag / tf.cast(
    total_minibatch_characters, tf.float32) for ag in acc_gs), train_vars))

book_loss_op = tf.reduce_sum(loss)

import shutil

print_width=shutil.get_terminal_size().columns

import re

def train_book(sess, orig_book):

    print("Book length:", len(orig_book))
    # Adding some newlines at the start should help it be more translationally
    # invariant
    perturb_len = random.randint(0, 5)
    book = ("\n" * perturb_len) + orig_book

    inj = { target : [lb.char_indices[next_char] for next_char in book[1:]],
        input_layer.input_seq : lb.one_hot(book[:-1]) }

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

    print(simplify_whitespace(orig_to_print))
    print(simplify_whitespace(predict_book))

    return book_loss

checkpoint_dir = "checkpoints/"

print("Initializing checkpoint saver...", end=' ')
saver = tf.train.Saver() # Buy default saves all variables, including stuff
# beta power accumulators in the Optimizer
import os
os.makedirs(checkpoint_dir, exist_ok=True)
print("done.")

#TODO: ok so deliberately stop propagating gradients through some paths to see
# if initialization is still helping. So like don't propagate through the
# bridge except the top layer and then see what magnitude of gradient we get at
# the bottom layer.

book_minibatch_size = 5

import objgraph
import pympler.asizeof

def runforward(sess):
    n_char_probs = sess.run(next_char,
        { input_layer.input_seq : lb.one_hot(text) })
    if len(n_char_probs) == 1:
        return lb.index_chars[0]
    # The model isn't going to give us an exact probabilty distribution
    # because of rounding errors. Numpy's multinomial has an assertion
    # that the probabilities (except the last) don't exceed one. So
    # let's just fudge the second to last character (whatever it is) by
    # the difference to fix this problem.
    fudge = max(sum(n_char_probs[:-1]) - 1, 0)
    n_char_probs[-2] -= fudge
    samples = np.random.multinomial(1, n_char_probs)
    n_char = np.argmax(samples)
    assert samples[n_char] == 1
    assert np.all(np.equal(samples[:n_char], 0))
    assert np.all(np.equal(samples[n_char+1:], 0))
    return lb.index_chars[n_char]

with tf.Session() as sess:

    print("Session start")

#    print(pympler.asizeof.asized(objgraph.by_type('Tensor'), detail=1).format())

    if cmd_args.open is None:
        to_load_from = tf.train.latest_checkpoint(checkpoint_dir)
    else:
        to_load_from = cmd_args.open

    if to_load_from is not None:
        print("Loading from:", to_load_from, "...", end=' ')
        saver.restore(sess, to_load_from)
    else:
        print("No checkpoints, initializing for the first time...", end=' ')
        sess.run(tf.global_variables_initializer())
    print("done.")

    if cmd_args.forward is not None:
        text = cmd_args.forward
        print(text, end='')
        while True:
            n_char = runforward(sess)
            print(n_char, end='', flush=True)
            text += n_char

    lowest_loss_so_far = math.inf

    epochs = cmd_args.epochs

    plateaus = [0]

    for epoch in range(epochs):
        print("\nEpoch", epoch, "\n")

        start_time = time.time()

        epoch_loss = 0

        random.shuffle(lb.train_books)

        mini_batches = [lb.train_books[pos : pos + book_minibatch_size]
            for pos in range(0, len(lb.train_books), book_minibatch_size)]

        for book_batch in mini_batches:
            sess.run([acc_g.initializer for acc_g in acc_gs])
            for book in book_batch:
                epoch_loss += train_book(sess, book)
            total_cs = sum(len(book) for book in book_batch)
            sess.run(apply_grads, { total_minibatch_characters : total_cs})

        total_predictions = sum(len(book) - 1 for book in lb.train_books)

        loss_per_character = epoch_loss / total_predictions

        print("Loss per character:", loss_per_character, end=' ')

        if loss_per_character < lowest_loss_so_far:
            lowest_loss_so_far = loss_per_character
            if plateaus[0] > 0: plateaus.insert(0, 0)
            print("which is the best so far")
        else:
            plateaus[0] += 1
            print("which isn't better than: ", lowest_loss_so_far, *plateaus)

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
