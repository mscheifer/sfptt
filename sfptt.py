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

# Several layers that have some constant indexed values and some indexed values
# from the lower layer. The bottom input layer is all constants.

# So each layer has two parts, constants and a convolution of the lower layer.
# Each part has indicies which represnent how to interleave the two parts with
# dynamic_stitch. You need one index for each value whether it is a constant or
# the result of a convolution. We should only move a value from convolution to
# constant (pruning) if it's inputs in the lower layer are both already
# constant, so that the pruning is removing a finite amount of info, rather
# than a potentially variable amount that sometimes requires traversing down
# the layers. When you move the value from convolution to constant, you also go
# to the lower layer and totally remove the two constant values feeding into
# it. Then you have to go through every index at that lower layer and, if it
# was greater than the indices of the two constants you just remove (which have
# adjacent indicies) decrement the index by 2. This is because the upper layer
# is now no longer expecting a value in that position for the convolution,
# because the convolution happens on the values before the stitch.

# Ok, alg is going to be: Try to almost have all variables be the same size.
# When you add a constant to a layer, if it's size goes beyond the max, then
# randomly select two adjacent constants to move up to the next layer. Repeat
# this process at each layer as long as the size goes beyond the max. When we
# add 1 then remove 2, the size goes down to max - 1. Then when we add 1 again
# it goes back to max_size, so no layer ever has something other than max or max
# - 1 values. Could easily have that as a boolean and just mask off 1 value in
# the variable. Could set extra slot in variable to -1 for index so it won't be
# erroneously used. The problem with all of this is that there may not be 2
# adjacent constants to move up. We would want, if a value was constant, for 2
# lower values feeding into those to become constant, and so the values feeding
# into those, and so on on down.

# Btw, as of 2nd April 2017, GPU support for dynamic stitch is being worked on
# right now. https://github.com/tensorflow/tensorflow/pull/8260
# Maybe could think of another way to do it anyway, because I don't care about
# collision order for dynamic_stitch and that will make the GPU implementation
# slower.
# Btw, scatter_nd will only be on gpu in tensorflow 1.1, not 1.0

# Layer norm seems like a good idea to make gradients make sense but it would
# break our principle of the result of a tree not changing at each new timestep
# allowing up to save it into a constant.

def extract_singleton(op):
    fst_dim = tf.shape(op)[0]
    with tf.control_dependencies([tf.Assert(tf.equal(fst_dim, 1), [fst_dim])]):
        return op[0]

class Layer:
    # gate idea is inspired by wavenet. They say it helps with vanishing
    # gradients similar to LSTM gates and highway networks. I don't fully
    # understand why yet. I actually removed the gate idea because I don't yet
    # know how to integrate it with layer normalization but we'll figure it out

    def make_conv_op(self, input):
        # Need to add a fake batch dimension because all the conv functions
        # assume you are using batches. Likewise we need to extract our "batch"
        filt = extract_singleton(tf.nn.conv1d(tf.expand_dims(input, 0),
             self.filter_weights, 2, 'SAME'))

        return tf.tanh(tf.add(filt, self.filt_bias))

    def make_weights(name, shape):
        init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        return tf.Variable(init(shape), name=name + "_weights")

    def __init__(self, lower_layer, num_output_channels):
        self.num_output_channels = num_output_channels #TODO: delete

        self.push_f = lower_layer.push_input
        
        # First dimension is what we conv over, 2nd is number of channels
        # Btw, .value fixes a really stupid bug with xavier_init because it
        # doesn't auto convert the dimension to an integer
        num_input_channels = lower_layer.result_op.get_shape()[1].value

        self.filter_weights = Layer.make_weights("filter",
            [2, num_input_channels, num_output_channels])
        self.filt_bias = tf.Variable(tf.zeros([1, num_output_channels],
            dtype=tf.float32), name="filter_baises")

        conv = self.make_conv_op(lower_layer.result_op)

        self.lower_layer_res = lower_layer.result_op # TODO delete

        self.conv = conv # TODO delete

        assert lower_layer.max_layer_size % 2 == 0
        max_conv_values = lower_layer.max_layer_size // 2

        # having more constants than the conv result size will guarantee that we
        # can always promote a constant up to the higher layer. The conv shape
        # size is always odd (because it's an even number divided by 2). We
        # should use another odd number for the constant size, that way the sum
        # of the two will be even.
        num_constants = max_conv_values + 2

        # TODO: IndexSlices is similar what we're trying to do here. A list of
        # indices and a list of values. Maybe should use that here instead of
        # the two separate variables

        # Ugh, really stupid bug with fill here. Can't mix dimenions and
        # integers so we need to get the value of the dimension.
        # Setting constants to 0 instead of nan because we need at least the
        # first value to be 0 to fix the stupid junk gradient bug
        constants = tf.Variable(tf.fill([num_constants.value, num_output_channels],
            np.float32(0)), trainable=False, name="consts")

        self.constants = constants # TODO delete

        # Ugh, really stupid bug with fill here. Can't mix dimenions and
        # integers so we need to get the value of the dimension.
        const_ind_init = tf.fill([num_constants.value - 1], np.int32(-1))
        # Starting with a const value (so plus the lower value, the two values
        # will go up a layer to be convolved) prevents a nasty bug when
        # convolutions have 0 outputs (and maybe evaluated conditionally or
        # something) and going through a dynamic stitch. I dunno but it seemed
        # to be reading junk data into the gradient. 
        const_ind_init = tf.concat([tf.zeros([1], tf.int32), const_ind_init], 0)
        const_indices = tf.Variable(const_ind_init,
            trainable=False, name="const_indices")
        del const_ind_init
        # Starting with the lower layer always having a 0 come up to prevent
        # the nasty bug described above so that goes in the 1st index
        # Ugh, really stupid bug with fill here. Can't give dimension arguments
        conv_ind_init = tf.fill([max_conv_values.value - 1], np.int32(-1))
        conv_ind_init = tf.concat([tf.ones([1], tf.int32), conv_ind_init], 0)
        conv_indices = tf.Variable(conv_ind_init, trainable=False,
            name="conv_indices")
        del conv_ind_init

        self.reset = tf.group(constants.initializer, const_indices.initializer,
            conv_indices.initializer)

        self.const_indices = const_indices #TODO: delete

        self.conv_indices = conv_indices #TODO: delete

        self.max_layer_size = constants.get_shape()[0] + max_conv_values

        assert self.max_layer_size % 2 == 0, ("Output size",
            str(self.max_layer_size),
            "should be even for clean convolutions")

        # Each layer will alternate between having 1 or 0 extra values that are
        # sliced out of the stitch to feed to the upper layer (so the result is
        # an even number) but are then sent sideways to be used in the bridge.

        def not_neg_one(op):
            # filter out -1 values of op
            return tf.logical_not(tf.equal(op, -1))

        result_const_positions = not_neg_one(const_indices)
        result_conv_positions = not_neg_one(conv_indices)

        assert_proper_indices = tf.Assert(
            tf.equal(tf.shape(conv)[0], tf.shape(tf.where(result_conv_positions))[0]),
            [conv, conv_indices], 6)

        with tf.control_dependencies([assert_proper_indices]):
            values = [tf.boolean_mask(constants, result_const_positions), 
                      tf.boolean_mask(conv     , result_conv_positions)]
        indices = [tf.boolean_mask(const_indices, result_const_positions),
                   tf.boolean_mask(conv_indices , result_conv_positions)]

        all_indices = tf.concat(indices, 0)

        assert_unique_indices = tf.Assert(tf.equal(tf.shape(all_indices)[0],
            tf.shape(tf.unique(all_indices)[0])[0]), indices, 6)

        with tf.control_dependencies([assert_unique_indices]):
            stitch = tf.dynamic_stitch(indices, values)

        self.num_results = tf.shape(stitch)[0]

        num_results_is_odd = tf.equal(self.num_results % 2, 1)

        # If we have an odd number of results, don't push the last one up. It
        # will be bridged.
        self.result_op = tf.cond(num_results_is_odd, lambda: stitch[0:-1],
            lambda: stitch)

        def bridge_with_lower_layer():
            return tf.layers.dense(
                # Expand dims to make a fake mini-batch
                inputs = tf.expand_dims(tf.concat([lower_layer.bridge_val,
                    stitch[-1]], 0), 0),
                units = num_output_channels, # could be different
                activation = tf.tanh
                # should default to xavier_initializer
                # should default to 0 initialized bias
                )[0] # extract single value of fake mini batch

        def pass_up_lower_layer():
            if lower_layer.bridge_val.shape == [num_output_channels]:
                return lower_layer.bridge_val
            # Just do a linear dimensionality change if they don't match
            return tf.layers.dense(
                # Expand dims to make a fake mini-batch
                inputs = tf.expand_dims(lower_layer.bridge_val, 0),
                units = num_output_channels, # could be different
                activation = None,
                # should default to xavier_initializer
                # should default to 0 initialized bias
                )[0] # extract single value of fake mini batch

        self.bridge_val = tf.cond(num_results_is_odd, bridge_with_lower_layer,
            pass_up_lower_layer)

        del stitch

        self.had_odd_results_before = tf.placeholder(shape=[], dtype=tf.bool)

        def update_new_conv_index():
            new_conv_pos = tf.shape(conv)[0] - 1
            # Don't want to evaluate stitch directly here because the conv indices
            # and values currently don't match until we do the update.
            # Not plus one because 0 based indices
            new_result_index = tf.shape(indices[0])[0] + tf.shape(indices[1])[0]

            free_conv_space = tf.Assert(tf.equal(conv_indices[new_conv_pos], -1),
                [conv_indices])
            actually_a_new_conv_value = tf.Assert(
                tf.equal(tf.shape(tf.where(result_conv_positions))[0], tf.shape(conv)[0] - 1),
                [conv, conv_indices], 6)

            with tf.control_dependencies([free_conv_space, actually_a_new_conv_value]):
                update_new_conv_index = conv_indices[new_conv_pos].assign(
                    new_result_index)
            junk = tf.fill([2, num_output_channels], np.float32(np.nan))

            with tf.control_dependencies([update_new_conv_index]):
                # If we had an odd number of results before, than the 2 new values
                # from the lower layer will become 1 new conv result which will,
                # along with the previous bridged only value, be returned up from
                # this layer as 2 new values.
                return tf.cond(self.had_odd_results_before,
                    lambda: tf.constant(-1), lambda: tf.constant(-2)), junk

        self.lower_layer_promoted_index = tf.placeholder(tf.int32, shape=[])
        self.lower_layer_promoted_values = tf.placeholder(tf.float32, shape=[2,
            num_input_channels])

        def insert_lower_promoted_then_promote():
            # we can't just use the lower layer result op and slice on the index
            # because the lower layer has already deleted those values and shifted
            # it's own indices
            assert_even = tf.Assert(tf.equal(self.lower_layer_promoted_index % 2, 0),
                [self.lower_layer_promoted_index])
            with tf.control_dependencies([assert_even]):
                new_const_conv_position = self.lower_layer_promoted_index // 2
            # we're passing in two values to this convolution so we should be
            # extracting the only value with [0]
            new_const_value = extract_singleton(self.make_conv_op(
                self.lower_layer_promoted_values))
            new_const_index = conv_indices[new_const_conv_position]

            valid_promote_index = tf.Assert(tf.logical_not(tf.equal(new_const_index, -1)),
                [new_const_index])

            # Only casting to int32 because tf.where doesn't take a dtype argument
            # for some reason
            empty_slots = tf.to_int32(tf.reshape(
                tf.where(tf.equal(const_indices, -1)), [-1]))

            def simple_insert():
                slot = empty_slots[0] # get one of many empty slots
                assignments = [
                    valid_promote_index,
                    constants    [slot].assign(new_const_value),
                    const_indices[slot].assign(new_const_index)]

                with tf.control_dependencies([new_const_index]):
                    ident_conv = tf.identity(conv_indices)

                junk = tf.fill([2, num_output_channels], np.float32(np.nan))
                with tf.control_dependencies(assignments):
                    # Our number of results stayed the same here, so return -2
                    return ident_conv, tf.constant(-2), junk

            # to_promote_index = const_indices[found[0]]
            # to_promote_values = [constants[found[0]], constants[found[1]]
            # constants    [found[0]] = new_const_value
            # constants    [found[1]] = nan
            # const_indices[found[0]] = new_const_index
            # const_indices[found[1]] = -1
            # const_indices.map_assign(x => x > to_promote_index ? x - 2 : x)
            # conv_indices.map_assign(x => x > to_promote_index ? x - 2 : x)
            def insert_and_promote():

                # tf.scatter_nd will fill the empty slots with zeros, but zero is also
                # a valid index into 'const_indices'. To make this work, we will assign
                # all the indices incremented by 1 and then decrement after we have
                # found them
                ci_pos_1 = tf.range(1, tf.shape(const_indices)[0] + 1, dtype=tf.int32)

                assert self.max_layer_size % 2 == 0, "Reshape will be weird"
                to_find_pairs = tf.reshape(
                    tf.scatter_nd(tf.reshape(const_indices, [-1, 1]), ci_pos_1,
                    # ugh, another place that should just accept a dimension
                    [self.max_layer_size.value]), [-1, 2])

                is_const_pair = tf.logical_and(
                    tf.logical_not(tf.equal(to_find_pairs[:,0], 0)), 
                    tf.logical_not(tf.equal(to_find_pairs[:,1], 0)))

                found_pairs = tf.boolean_mask(to_find_pairs, is_const_pair)

                assert_found = tf.Assert(tf.shape(found_pairs)[0] > 0,
                    [const_indices, to_find_pairs])
                # Should find at least 1 pair by pigeonhole principle
                with tf.control_dependencies([assert_found]):
                    # need to subtract 1 because we added 1 to positions above
                    found = found_pairs[0] - 1

                const_to_promote = tf.gather(constants, found)

                assert_adjacent = tf.Assert(tf.equal(
                    const_indices[found[1]], const_indices[found[0]] + 1),
                    [const_indices, found])
                with tf.control_dependencies([assert_adjacent]):
                    const_index_to_p = const_indices[found[0]]

                with tf.control_dependencies([const_to_promote]):
                    const_update = tf.scatter_update(constants, found,
                        [new_const_value, tf.fill([num_output_channels],
                            np.float32(np.nan))])

                with tf.control_dependencies([const_index_to_p, valid_promote_index]):
                    move_index_from_conv_to_const = tf.scatter_update(const_indices,
                         found, [new_const_index, -1])

                def shift_indices(var, val):
                    greater_than_p_index = val > const_index_to_p

                    return var.assign(tf.where(greater_than_p_index, val - 2, val))

                # ugh, why doesnt scatter_update return a mutable refernce so I can
                # chain?
                shift_const_indices = shift_indices(const_indices,
                    move_index_from_conv_to_const)

                # Don't shift the conv indices until we have found the index we want to
                # move because otherwise we may move the shifted value instead
                with tf.control_dependencies([new_const_index]):
                    shift_conv_indices = shift_indices(conv_indices, conv_indices)

                with tf.control_dependencies([const_update, shift_const_indices]):
                    const_index_to_promote = tf.identity(const_index_to_p)
                    const_to_promote = tf.identity(const_to_promote)

                # const_indices[found[1]] will be const_indices[found[0]] + 1 so we
                # don't need to also return it. Just the one to promote index
                return shift_conv_indices, const_index_to_promote, const_to_promote

            # shifted_conv is to ensure correct write dependencies on conv_indices
            shifted_conv, promote_index, promote_value = tf.cond(
                tf.equal(tf.shape(empty_slots)[0], 0),
                insert_and_promote, simple_insert)

            # we need to move the new const value first before trimming out its
            # index here
            conv_indices_inc = conv_indices.assign(tf.concat([
                shifted_conv[0 : new_const_conv_position],
                # skip the 1 conv val we moved that corresponds to the 2 lower vals
                shifted_conv[new_const_conv_position + 1 :],
                # Feed -1 in here because when we promote there is never a new conv
                # value. That will happen on the next input
                tf.expand_dims(-1, 0)], axis=0))

            with tf.control_dependencies([conv_indices_inc]):
                return tf.identity(promote_index), tf.identity(promote_value)

        self.insert_lower_promoted_if_exists_then_promote = tf.cond(
            # -1 means new value but not promote
            tf.equal(self.lower_layer_promoted_index, -1),
            update_new_conv_index, # If there is no promoted value
            insert_lower_promoted_then_promote)

    # super ugly but it's hard to do enums/ADTs in tensorflow so -2 means no new
    # input vales. -1 means new input but no new promoted values, and an actual
    # index means that's the promoted index and the second arg won't be junk
    def push_input(self, char, sess):

        # evaluated before recursing
        had_odd_results_before = sess.run(self.num_results) % 2 == 1

        rec_res = self.push_f(char, sess)
        if rec_res[0] == -2: # This means there are no new result values
            return -2, np.full([2, self.num_output_channels], np.float32(np.nan))

        # else, there are new values from the below layer and we need to add
        # indices for them

        update_index, update_values = rec_res

        inj = {  self.had_odd_results_before : had_odd_results_before,
            self.lower_layer_promoted_index : update_index,
            self.lower_layer_promoted_values : update_values }

        return sess.run(self.insert_lower_promoted_if_exists_then_promote, inj)

class Input:
    def __init__(self, num_constants):
        # Starting the write head at 2 and starting with an initial 00 value (so
        # the first input will be bridges and we start with two zeros convolved
        # above) prevents a nasty bug when convolutions have 0 outputs (and
        # maybe evaluated conditionally or something) and going through a
        # dynamic stitch. I dunno but it seemed to be reading junk data into
        # the gradient. 
        write_head = tf.Variable(np.int32(2), trainable=False)

        input_channels = len(lb.character_set)

        c_init = tf.fill([num_constants-2, input_channels], np.float32(np.nan))
        # Here is the 0 value to fix nasty bug described above
        c_init = tf.concat([tf.fill([2, input_channels], np.float32(0)),
             c_init], axis=0)
        constants = tf.Variable(c_init, trainable=False)

        self.constants = constants # XXX TODO: delete

        self.reset = tf.group(write_head.initializer, constants.initializer)

        self.max_layer_size = constants.get_shape()[0]

        self.new_char = tf.placeholder(tf.float32, shape=[input_channels])

        def simple_add_input():
            write_input = constants[write_head].assign(self.new_char)

            with tf.control_dependencies([write_input]):
                inc_write_head = write_head.assign_add(1) #post inc

            junk = tf.fill([2, input_channels], np.float32(np.nan))

            # If it's even now, then it was odd before, so we have 2 new values
            return tf.cond(tf.equal(inc_write_head % 2, 0),
                lambda: (tf.constant(-1), junk),lambda: (tf.constant(-2), junk))

        def remove_2_values():
            # generate a random even index
            random_index = 2 * tf.random_uniform([], 0,
                self.max_layer_size // 2, tf.int32)

            rems = constants[random_index : random_index + 2]

            before_slice = constants[0 : random_index]
            after_slice  = constants[random_index + 2 :]
            # We need to read the values out of the graph before overwriting them
            with tf.control_dependencies([rems]):
                # Insert the new character after the other values because we're
                # always keeping the values in order.
                # Insert a nan on the end to fill out the variable to it's full size
                push_update = constants.assign(tf.concat([before_slice,
                    after_slice, tf.expand_dims(self.new_char, 0),
                    tf.fill([1, input_channels], np.float32(np.nan))], 0))

            # put the write head at the last index because we removed 2 and added 1
            # value from a full vector so we should only have 1 free space
            dec_write_head = write_head.assign(self.max_layer_size - 1)

            with tf.control_dependencies([push_update, dec_write_head]):
                rem_values = tf.identity(rems)

            # return the removed index and values
            return random_index, rem_values

        self.push = tf.cond(write_head < self.max_layer_size,
            simple_add_input, remove_2_values)

        non_nan_values = constants[0 : write_head]

        num_results_is_odd = tf.equal(tf.shape(non_nan_values)[0] % 2, 1)
        # If we have an odd number of results, don't push the last one up. It
        # will be bridged.
        self.result_op = tf.cond(num_results_is_odd, 
            lambda: non_nan_values[0:-1], lambda: non_nan_values)

        # Bridge the value that is not included
        self.bridge_val = tf.cond(num_results_is_odd, lambda: non_nan_values[-1],
            lambda: tf.zeros([input_channels], tf.float32)) 

        assert self.max_layer_size % 2 == 0, ("Output size",
            str(self.max_layer_size),
            "should be even for clean convolutions")

    def push_input(self, char, sess):

        return sess.run(self.push, { self.new_char : char })

layer_channels = 2 #TODO: can vary per layer

lowest_level_memory_size = 6

num_layers = 6 # TODO: can this be dynamic and grow with more input?

resets = []

with tf.get_default_graph().name_scope("input_layer"):
    layer = Input(lowest_level_memory_size)
input_layer = layer
resets.append(layer.reset)

conv_indices = []
const_indices = []
constants = [input_layer.constants]
convs = []

for i in range(num_layers):
    with tf.get_default_graph().name_scope("layer_" + str(i)):
        layer = Layer(layer, layer_channels)
    resets.append(layer.reset)
    conv_indices.append(layer.conv_indices)
    const_indices.append(layer.const_indices)
    constants.append(layer.constants)
    convs.append(layer.conv)

output = tf.layers.dense(
    # Expand dims to make a fake mini-batch
    inputs = tf.expand_dims(layer.bridge_val, 0),
    units = len(lb.character_set),
    # linear activation function here because we softmax as part of the loss
    activation = None
    # should default to xavier_initializer
    # should default to 0 initialized bias
    )[0] # extract single value of fake mini batch

target = tf.placeholder(tf.int32, shape=[])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels = target, logits = output)

#optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)
#optimizer = tf.train.AdamOptimizer()
optimizer = tf.train.GradientDescentOptimizer(1.0)

grads_and_vars = [(g, v) for g, v in optimizer.compute_gradients(loss) if g is not None]

grads = [tf.check_numerics(g, "grr") for g, _ in grads_and_vars]
vars = [v for _, v in grads_and_vars]

for v in vars:
    tf.summary.histogram(v.name + "_hist", v)

import time

write_summaries = tf.summary.FileWriter("./summaries/" + str(time.time()))

acc_gs = [tf.Variable(tf.fill(g.shape, np.float32(0)), trainable=False,
    name=v.name.translate(str.maketrans("/:", "__")) + "_grad")
    for g, v in grads_and_vars]

for acc_g in acc_gs:
    tf.summary.histogram(acc_g.name + "_hist", acc_g)

all_summaries = tf.summary.merge_all()

num_predictions = sum(len(book) - 1 for book in lb.train_books)

# dividing gradients before summing should prevent overflows here
accumulate_gradients = [ag.assign_add(g / num_predictions) for ag, g in zip(acc_gs, grads)]
apply_grads = optimizer.apply_gradients(zip(acc_gs, vars))

epochs = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("Epoch", epoch)

        sess.run([acc_g.initializer for acc_g in acc_gs])

        for book in lb.train_books:

            sess.run(resets)

            predict_book = book[0]
            book_loss = 0

            for char, next_char in zip(book[:-1], book[1:]):
                # one hot returns as a 2D batch but we're feeding in 1 at a time
                layer.push_input(lb.one_hot(char)[0], sess) 

                inj = { target : lb.char_indices[next_char] }
                char_loss, predict, _ = sess.run([loss, output, accumulate_gradients], inj)

#                if epoch == epochs - 1:
#                    print(predict)

                def print_array(array):
                    print("\n".join(str(val) for val in array))

                prnt = False

                if prnt and epoch == epochs - 1:
                    print("Constants")
                    print_array(sess.run(constants))

                if prnt and epoch == epochs - 1:
                    print("Convs")
                    print_array(sess.run(convs))

                if prnt and epoch == epochs - 1:
                    print("Const Indices")
                    print_array(sess.run(const_indices))

                if prnt and epoch == epochs - 1:
                    print("Conv Indices")
                    print_array(sess.run(conv_indices))

                assert not np.isnan(char_loss)

                predict_book += (lb.index_chars[np.argmax(predict)])
                book_loss += char_loss

            print(book)
            print(predict_book, "-", book_loss)

        sess.run(apply_grads)

#        if epoch % (epochs / 20) == 0 or epoch == epochs - 1: #epochs - 1 is last one
#            write_summaries.add_summary(sess.run(all_summaries), epoch)

        vars = [v for g, v in grads_and_vars]
        #print("Grads", "\n".join(str((v.name, val)) for v, val in zip(vars, acc_gs)))
        #print("Weights", [(v.name, val) for v, val in zip(vars, sess.run(vars))])
        #print("Input", sess.run(input_layer.constants))
        #print("Conv", "\n".join(str(i) for i in sess.run(conv_indices)))
        #print("Const", "\n".join(str(i) for i in sess.run(const_indices)))