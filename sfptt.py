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

cmd_arg_def.add_argument("-l", "--learning_rate", type=float, default=0.01,
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

def make_conv_op(input, filter_weights, filt_bias):
    # Need to add a fake batch dimension because all the conv functions
    # assume you are using batches. Likewise we need to extract our "batch"
    filt = extract_singleton(tf.nn.conv1d(tf.expand_dims(input, 0),
         filter_weights, 2, 'SAME'))

    return tf.tanh(tf.add(filt, filt_bias))

# TODO: better theoretical justification for scaling by 1.3 for tanh layers?
# Saxe just says > 1 in the g+ post. Other people are saying because the stddev
# of tanh(norm(0, 1)) ~ 0.63 then 1.3 will compensate for that.
#
# I'm confused by why for orthogonal initilization we can just chop the output
# side to create our rectangular matrices. Does it really only have to be
# orthogonal accross the input dimension? What about the backprop signal?
#
# Saxe (the originator of orthogonal init from Saxe et. al ICML (2014)) does
# that in his own code linked below. He also multiplies by the previous layer's
# initial matrix to work around a different issue and I'm not sure if that
# affects the orthogonality.
#
# https://plus.google.com/+SoumithChintala/posts/RZfdrRQWL6u
# I can't link directly to a comment on G+ because G+ sucks but you can find it
# by expanding comments. It's Saxe's first comment.

def make_bridge_op(result, num_results_is_odd, bridge_channels,
        lower_layer_bridge_val):

    def bridge_with_lower_layer():
        with tf.name_scope("bridge") as scope:
            return tf.layers.dense(
                # Expand dims to make a fake mini-batch
                inputs = tf.expand_dims(tf.concat([lower_layer_bridge_val,
                    result[-1]], 0), 0),
                units = bridge_channels,
                activation = tf.tanh,
                # TODO: check if this is the best gain for tanh
                kernel_initializer = tf.orthogonal_initializer(gain=1.3),
                # should default to 0 initialized bias
                # WTF, why does dense() set the scope to the name rather than just
                # using the current scope?
                name = scope
                )[0] # extract single value of fake mini batch

    def pass_up_lower_layer():
        if lower_layer_bridge_val.shape == [bridge_channels]:
            return lower_layer_bridge_val
        with tf.name_scope("bridge") as scope:
            # Just do a linear dimensionality change if they don't match
            return tf.layers.dense(
                # Expand dims to make a fake mini-batch
                inputs = tf.expand_dims(lower_layer_bridge_val, 0),
                units = bridge_channels,
                activation = None,
                kernel_initializer = tf.orthogonal_initializer(),
                # should default to 0 initialized bias
                # WTF, why does dense() set the scope to the name rather than just
                # using the current scope?
                name = scope + "_reshape"
                )[0] # extract single value of fake mini batch

    return tf.cond(num_results_is_odd, bridge_with_lower_layer,
        pass_up_lower_layer)

def make_weights(name, shape):
    init = tf.orthogonal_initializer(gain=1.3)
    return tf.Variable(init(shape, tf.float32), name=name + "_weights")

def make_simple_layer(lower_layer, num_output_channels):

    # First dimension is what we conv over, 2nd is number of channels
    # Btw, .value fixes a really stupid bug with xavier_init because it
    # doesn't auto convert the dimension to an integer
    num_input_channels = lower_layer.result_op.get_shape()[1].value

    filter_weights = make_weights("filter",
        [2, num_input_channels, num_output_channels])
    filt_bias = tf.Variable(tf.zeros([1, num_output_channels],
        dtype=tf.float32), name="filter_baises")

    #ugh dumb hack around nasty bug
    lower_layer_result = tf.concat([np.zeros([2, num_input_channels]), lower_layer.result_op], 0)
    conv = make_conv_op(lower_layer_result, filter_weights, filt_bias)[1:] # slice off hack

    num_results_is_odd = is_odd(tf.shape(conv)[0])

    # If we have an odd number of results, don't push the last one up. It
    # will be bridged.
    result_op = tf.cond(num_results_is_odd, lambda: conv[0:-1],
        lambda: conv)

    assert lower_layer.max_layer_size % 2 == 0
    max_layer_size = (lower_layer.max_layer_size // 2)
    # max layer size is when there is no bridge. We can actually set the max
    # size to be more than it will ever be in order to fulfill that definition
    # because layers are always allowed to have fewer results than their max.
    if max_layer_size % 2 == 1:
        max_layer_size += 1

    assert max_layer_size % 2 == 0, str((lower_layer.max_layer_size, max_layer_size))

    class Layer:
        def __init__(self):
            self.reset = tf.no_op()
            self.max_layer_size = max_layer_size
            self.result_op = result_op
            self.bridge_val = make_bridge_op(
                conv,
                num_results_is_odd,
                num_output_channels, # could be different
                lower_layer.bridge_val)

            # Even though we read this value more than once when the assert is
            # on, TF won't execute the op more than once.
            lower_push = lower_layer.push_op

            def assert_no_promote():
                return [tf.Assert(lower_push[0] < 0, [lower_push])]

            self.push_op = with_assert(lambda: lower_push, assert_no_promote)

    return Layer()

def make_layer(lower_layer, num_output_channels, max_outputs_for_dataset):

    assert lower_layer.max_layer_size % 2 == 0
    max_conv_values = lower_layer.max_layer_size // 2

    # if the convolution values and the bridge are enough to cover the entire
    # input length then we can create a layer that doesn't need constants
    if max_conv_values >= max_outputs_for_dataset:
        return make_simple_layer(lower_layer, num_output_channels)

    # First dimension is what we conv over, 2nd is number of channels
    # Btw, .value fixes a really stupid bug with xavier_init because it
    # doesn't auto convert the dimension to an integer
    num_input_channels = lower_layer.result_op.get_shape()[1].value

    filter_weights = make_weights("filter",
        [2, num_input_channels, num_output_channels])
    filt_bias = tf.Variable(tf.zeros([1, num_output_channels],
        dtype=tf.float32), name="filter_baises")

    def make_local_conv_op(input):
        return make_conv_op(input, filter_weights, filt_bias)

    conv = make_local_conv_op(lower_layer.result_op)

    # Having more constants than the conv result size will guarantee that we
    # can always promote a constant up to the higher layer.
    # However, we don't need to ever promote if the max conv values + the
    # number of constants is less than 'max_outputs_for_dataset'
    extra_constants_for_dataset = max(
        # Make sure this isn't less than 1 because we currently have that hack
        # where we start with a 0 in the constants.
        tf.Dimension(max_outputs_for_dataset) - max_conv_values, 1)
    # The conv shape size may be odd. If so we should use another odd number
    # for the constant size, that way the sum of the two will be even.
    if max_conv_values % 2 != extra_constants_for_dataset % 2:
        extra_constants_for_dataset += 1
    num_constants = min(extra_constants_for_dataset, max_conv_values + 2)

    del extra_constants_for_dataset

    # TODO: IndexSlices is similar what we're trying to do here. A list of
    # indices and a list of values. Maybe should use that here instead of
    # the two separate variables

    # Ugh, really stupid bug with fill here. Can't mix dimenions and
    # integers so we need to get the value of the dimension.
    # Setting constants to 0 instead of nan because we need at least the
    # first value to be 0 to fix the stupid junk gradient bug
    constants = tf.Variable(tf.fill([num_constants.value, num_output_channels],
        np.float32(0)), trainable=False, name="consts")

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

    reset = tf.group(constants.initializer, const_indices.initializer,
        conv_indices.initializer)

    max_layer_size = constants.get_shape()[0] + max_conv_values

    assert max_layer_size % 2 == 0, ("Output size", str(max_layer_size),
        "should be even for clean convolutions", constants.get_shape()[0],
        max_conv_values)

    # Each layer will alternate between having 1 or 0 extra values that are
    # sliced out of the stitch to feed to the upper layer (so the result is
    # an even number) but are then sent sideways to be used in the bridge.

    result_const_positions = tf.not_equal(const_indices, -1)
    result_conv_positions  = tf.not_equal(conv_indices , -1)

    result_const_indices = tf.boolean_mask(const_indices, result_const_positions)
    result_conv_indices  = tf.boolean_mask(conv_indices , result_conv_positions )

    def assert_proper_indices():
        return [tf.Assert(tf.equal(tf.shape(conv)[0],
            tf.shape(tf.where(result_conv_positions))[0]),
            [tf.shape(conv), conv_indices], 6)]

    values = with_assert(lambda: [
            tf.boolean_mask(constants, result_const_positions), 
            tf.boolean_mask(conv     , result_conv_positions )],
            assert_proper_indices)

    indices = [result_const_indices, result_conv_indices]

    def assert_unique_indices():
        # unique returns 2 things
        unique_indices = tf.unique(tf.concat(indices, 0))[0]
        unique_shape = tf.shape(unique_indices)[0]
        return [tf.Assert(tf.equal(
            tf.shape(indices[0])[0] + tf.shape(indices[1])[0], unique_shape),
            indices + [unique_shape, unique_indices], 6)]

    stitch = with_assert(lambda: tf.dynamic_stitch(indices, values),
        assert_unique_indices)

    num_results_is_odd = is_odd(tf.shape(stitch)[0])

    # If we have an odd number of results, don't push the last one up. It
    # will be bridged.
    result_op = tf.cond(num_results_is_odd, lambda: stitch[0:-1],
        lambda: stitch)

    bridge_val = make_bridge_op(stitch, num_results_is_odd,
        num_output_channels, # could be different
        lower_layer.bridge_val)

    del stitch

    # super ugly but it's hard to do enums/ADTs in tensorflow so -2 means no new
    # input vales. -1 means new input but no new promoted values, and an actual
    # index means that's the promoted index and the second arg won't be junk
    lower_layer_promoted_index, lower_layer_promoted_values = lower_layer.push_op

    def absorb_new_result():    

        def update_new_conv_index():
            # The position of the new conv value will be however many indices
            # we already had plus one
            new_conv_pos = tf.shape(result_conv_indices)[0]
            # Don't want to evaluate stitch directly here because the conv indices
            # and values currently don't match until we do the update.
            # Not plus one because 0 based indices
            new_result_index = tf.shape(indices[0])[0] + tf.shape(indices[1])[0]

            def conv_asserts():
                enough_indices = tf.Assert(
                    tf.less(new_conv_pos, tf.shape(conv_indices)[0]),
                    [new_conv_pos, conv_indices])
                free_conv_space = tf.Assert(
                    tf.equal(conv_indices[new_conv_pos], -1),
                    [conv_indices])
                return [enough_indices, free_conv_space]

            update_new_conv_index = with_assert(lambda: conv_indices[new_conv_pos].assign(
                new_result_index), conv_asserts)

            junk = tf.fill([2, num_output_channels], np.float32(np.nan))

            with tf.control_dependencies([update_new_conv_index]):
                # If we have an even number of results (which happens when the
                # new index is odd, than the 2 new values from the lower layer
                # will become 1 new conv result which will, along with the
                # previous bridged only value, be returned up from this layer
                # as 2 new values.
                return tf.cond(is_odd(new_result_index),
                    lambda: tf.constant(-1), lambda: tf.constant(-2)), junk

        def insert_lower_promoted_then_promote():
            # we can't just use the lower layer result op and slice on the index
            # because the lower layer has already deleted those values and shifted
            # it's own indices
            def assert_even():
                return [tf.Assert(tf.equal(lower_layer_promoted_index % 2, 0),
                    [lower_layer_promoted_index])]

            new_const_conv_position = with_assert(
                lambda: lower_layer_promoted_index // 2, assert_even)
            # we're passing in two values to this convolution so we should be
            # extracting the only value with [0]
            new_const_value = extract_singleton(make_local_conv_op(
                lower_layer_promoted_values))
            new_const_index = conv_indices[new_const_conv_position]

            # Only casting to int32 because tf.where doesn't take a dtype argument
            # for some reason
            empty_slots = tf.to_int32(tf.reshape(
                tf.where(tf.equal(const_indices, -1)), [-1]))

            def simple_insert():
                slot = empty_slots[0] # get one of many empty slots
                assignments = [constants    [slot].assign(new_const_value),
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

                assert max_layer_size % 2 == 0, "Reshape will be weird"
                to_find_pairs = tf.reshape(
                    tf.scatter_nd(tf.reshape(const_indices, [-1, 1]), ci_pos_1,
                    # ugh, another place that should just accept a dimension
                    [max_layer_size.value]), [-1, 2])

                is_const_pair = tf.logical_and(
                    tf.not_equal(to_find_pairs[:,0], 0),
                    tf.not_equal(to_find_pairs[:,1], 0))

                found_pairs = tf.boolean_mask(to_find_pairs, is_const_pair)

                def assert_found():
                    # Should find at least 1 pair by pigeonhole principle
                    return [tf.Assert(tf.shape(found_pairs)[0] > 0,
                        [const_indices, to_find_pairs])]
                # need to subtract 1 because we added 1 to positions above
                found = with_assert(lambda: found_pairs[0] - 1, assert_found)

                const_to_promote = tf.gather(constants, found)

                def assert_adjacent():
                    return [tf.Assert(tf.equal(
                        const_indices[found[1]], const_indices[found[0]] + 1),
                        [const_indices, found])]
                const_index_to_p = with_assert(lambda: const_indices[found[0]],
                     assert_adjacent)

                with tf.control_dependencies([const_to_promote]):
                    const_update = tf.scatter_update(constants, found,
                        [new_const_value, tf.fill([num_output_channels],
                            np.float32(np.nan))])

                with tf.control_dependencies([const_index_to_p]):
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

            def valid_promote_index():
                return [tf.Assert(tf.not_equal(new_const_index, -1),
                    [new_const_index])]
            # shifted_conv is to ensure correct write dependencies on conv_indices
            shifted_conv, promote_index, promote_value = with_assert(lambda:
                tf.cond(tf.equal(tf.shape(empty_slots)[0], 0),
                insert_and_promote, simple_insert), valid_promote_index)

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

        return tf.cond(
            # -1 means new value but not promote
            tf.equal(lower_layer_promoted_index, -1),
            update_new_conv_index, # If there is no promoted value
            insert_lower_promoted_then_promote)

    with tf.control_dependencies([lower_layer_promoted_values]):
        junk = tf.fill([2, num_output_channels], np.float32(np.nan))

        push_op = tf.cond(tf.equal(lower_layer_promoted_index, -2),
            lambda: (tf.constant(-2), junk), absorb_new_result)

    class Layer:
        def __init__(self):
            self.reset = reset
            self.max_layer_size = max_layer_size
            self.result_op = result_op
            self.bridge_val = bridge_val
            self.push_op = push_op

    return Layer()

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

        self.reset = tf.group(write_head.initializer, constants.initializer)

        self.max_layer_size = constants.get_shape()[0]

        self.new_char = tf.placeholder(tf.float32, shape=[input_channels])

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

        self.push_op = tf.cond(write_head < self.max_layer_size,
            simple_add_input, remove_2_values)

import math

lowest_level_memory_size = 10

max_len = max(len(book) for book in lb.train_books)
# TODO: can this be dynamic and grow with more input?
num_layers = int(math.ceil(math.log2(max_len)))
print("Max book length", max_len, "so number of layers is", num_layers)

resets = []

with tf.name_scope("input_layer"):
    layer = Input(lowest_level_memory_size)
input_layer = layer
resets.append(layer.reset)

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
        layer = make_layer(layer, layer_channels, max_chunks)
        print("Layer", i, "has", layer_channels, "channels and",
            layer.max_layer_size, "values.")

    resets.append(layer.reset)

    layer_channels = int(math.ceil(layer_channels * memory_scale_factor))

print("The final layer says size 2 but only 1 value will ever be computed.",
    "It just says 2 because every layer has to have a positive even max size")

output = tf.layers.dense(
    # Expand dims to make a fake mini-batch
    inputs = tf.expand_dims(layer.bridge_val, 0),
    units = len(lb.character_set),
    # linear activation function here because we softmax as part of the loss
    activation = None,
    kernel_initializer = tf.orthogonal_initializer(),
    # should default to 0 initialized bias
    name="output_characters"
    )[0] # extract single value of fake mini batch

target = tf.placeholder(tf.int32, shape=[])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels = target, logits = output)

#optimizer = tf.train.AdadeltaOptimizer(learning_rate=cmd_args.learning_rate)
#optimizer = tf.train.AdamOptimizer()
optimizer = tf.train.GradientDescentOptimizer(cmd_args.learning_rate)

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

if cmd_args.summaries:
    import time

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
    sess.run(resets)

    book_loss = 0

    # TODO: reduce maxlen to terminal width if smaller
    book_buffer = collections.deque(maxlen = 80) 
    predict_buffer = collections.deque(maxlen = 80)

    book_buffer.append(book[0])
    predict_buffer.append(book[0]) # we don't actually predict the first char

    print("Book length:", len(book), "\n\n")

    for progress, (char, next_char) in enumerate(zip(book[:-1], book[1:])):
        # one hot returns as a 2D batch but we're feeding in 1 at a time
        sess.run(layer.push_op, { input_layer.new_char : lb.one_hot(char)[0] }) 

        inj = { target : lb.char_indices[next_char] }
        char_loss, predict, _ = sess.run(
            [loss, output, accumulate_gradients], inj)

        def simplify_whitespace(s):
            return re.sub(r"\s", " ", s)

        book_loss += char_loss

        book_buffer.append(simplify_whitespace(next_char))
        predict_buffer.append(
            simplify_whitespace(lb.index_chars[np.argmax(predict)]))

        print("\033[F\033[F", "".join(book_buffer), sep='')
        print("".join(predict_buffer), end=' ')
        print("%3.2f%%" % ((progress / len(book)) * 100))

    return book_loss

checkpoint_dir = "checkpoints/"
# Because we only have to save the weights, we could actually vary the constant
# sizes between training sessions
saver = tf.train.Saver(tf.trainable_variables())
import os
os.makedirs(checkpoint_dir, exist_ok=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    to_load_from = tf.train.latest_checkpoint(checkpoint_dir)

    if to_load_from is not None:
        saver.restore(sess, to_load_from)
        print("Loaded from:", to_load_from)

    lowest_loss_so_far = math.inf

    epochs = cmd_args.epochs

    for epoch in range(epochs):
        print("\nEpoch", epoch, "\n")

        sess.run([acc_g.initializer for acc_g in acc_gs])

        epoch_loss = 0

        for book in lb.train_books:
            epoch_loss += train_book(sess, book)

        print("Loss:", epoch_loss, end=' ')

        if epoch_loss < lowest_loss_so_far:
            lowest_loss_so_far = epoch_loss
            print("which is the best so far")
        else:
            print() # new line

        sess.run(apply_grads)

        save_path = saver.save(sess, checkpoint_dir + "save", global_step=epoch)
        print("Saved to:", save_path)

        #epochs - 1 is last one
        e = epoch
        if cmd_args.summaries and (e % (epochs / 20) == 0 or e == epochs - 1):
            print("Summarizing epoch:", e)
            write_summaries.add_summary(sess.run(all_summaries), epoch)

if cmd_args.summaries:
    write_summaries.close() # file doesn't auto close when the process exits...
