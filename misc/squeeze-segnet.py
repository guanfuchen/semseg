#!/usr/bin/python
# -*- coding: UTF-8 -*-

# code is from nanfackg@gmail.com

import tensorflow as tf
import tensorflow.contrib.slim as slim
from glob import glob
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
# from get_class_weights import median_frequency_balancing
import time
import os
from matplotlib import pyplot as plt

global a
import numpy as np


def fire_module(input, fire_id, channel, s1, e1, e3, ):
    """
    Basic module that makes up the SqueezeNet architecture. It has two layers.
     1. Squeeze layer (1x1 convolutions)
     2. Expand layer (1x1 and 3x3 convolutions)
    :param input: Tensorflow tensor
    :param fire_id: Variable scope name
    :param channel: Depth of the previous output
    :param s1: Number of filters for squeeze 1x1 layer
    :param e1: Number of filters for expand 1x1 layer
    :param e3: Number of filters for expand 3x3 layer
    :return: Tensorflow tensor
    """

    fire_weights = {
        'conv_s_1': tf.Variable(tf.truncated_normal([1, 1, channel, s1], stddev=0.001), name='weights_conv_s_1'),
        'conv_e_1': tf.Variable(tf.truncated_normal([1, 1, s1, e1], stddev=0.001), name='weights_conv_e_1'),
        'conv_e_3': tf.Variable(tf.truncated_normal([3, 3, s1, e3], stddev=0.001), name='weights_conv_e_3')}

    fire_biases = {'conv_s_1': tf.Variable(tf.truncated_normal([s1]), name='bias_conv_s_1'),
                   'conv_e_1': tf.Variable(tf.truncated_normal([e1]), name='bias_conv_e_1'),
                   'conv_e_3': tf.Variable(tf.truncated_normal([e3]), name='bias_conv_e_3')}

    with tf.name_scope(fire_id):
        output = tf.nn.conv2d(input, fire_weights['conv_s_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_s_1')
        output = tf.nn.relu(tf.nn.bias_add(output, fire_biases['conv_s_1']))

        expand1 = tf.nn.conv2d(output, fire_weights['conv_e_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_1')
        expand1 = tf.nn.bias_add(expand1, fire_biases['conv_e_1'])

        expand3 = tf.nn.conv2d(output, fire_weights['conv_e_3'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_3')
        expand3 = tf.nn.bias_add(expand3, fire_biases['conv_e_3'])

        result = tf.concat([expand1, expand3], 3, name='concat_e1_e3')
        return tf.nn.relu(result)


def dfire_module(input, dfire_id, channel, s1_D, e1_D, e3_D, ):
    fire_weights = {'conv_s_1_D': tf.Variable(tf.truncated_normal([1, 1, e1_D + e3_D, s1_D], stddev=0.001)),
                    'conv_e_1_D': tf.Variable(tf.truncated_normal([1, 1, channel, e1_D], stddev=0.001)),
                    'conv_e_3_D': tf.Variable(tf.truncated_normal([3, 3, channel, e3_D], stddev=0.001))}

    fire_biases = {'conv_s_1_D': tf.Variable(tf.truncated_normal([s1_D])),
                   'conv_e_1_D': tf.Variable(tf.truncated_normal([e1_D])),
                   'conv_e_3_D': tf.Variable(tf.truncated_normal([e3_D]))}

    with tf.name_scope(dfire_id):
        expand1 = tf.nn.conv2d(input, fire_weights['conv_e_1_D'], strides=[1, 1, 1, 1], padding='SAME',
                               name='conv_e_1_D')
        expand1 = tf.nn.bias_add(expand1, fire_biases['conv_e_1_D'])

        expand3 = tf.nn.conv2d(input, fire_weights['conv_e_3_D'], strides=[1, 1, 1, 1], padding='SAME',
                               name='conv_e_3_D')
        expand3 = tf.nn.bias_add(expand3, fire_biases['conv_e_3_D'])

        result = tf.concat([expand1, expand3], 3, name='concat_e1_D_e3_D')
        result = tf.nn.relu(result)
        output = tf.nn.conv2d(result, fire_weights['conv_s_1_D'], strides=[1, 1, 1, 1], padding='SAME',
                              name='conv_s_1_D')
        output = tf.nn.relu(tf.nn.bias_add(output, fire_biases['conv_s_1_D']))
        return output


def unpool(updates, mask, k_size=[1, 2, 2, 1], output_shape=None, scope='unpool'):
    '''
   # NOTE! this function is based on the implementation by kwotsin in
    # https://github.com/kwotsin/TensorFlow-ENet
    - inputs(Tensor): a 4D tensor of shape [batch_size, height, width, num_channels] that represents the input block to be upsampled
    - mask(Tensor): a 4D tensor that represents the argmax values/pooling indices of the previously max-pooled layer
    - k_size(list): a list of values representing the dimensions of the unpooling filter.
    - output_shape(list): a list of values to indicate what the final output shape should be after unpooling
    - scope(str): the string name to name your scope
    OUTPUTS:
    - ret(Tensor): the returned 4D tensor that has the shape of output_shape.
    '''
    with tf.variable_scope(scope):
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]  # mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, shape=output_shape)
        return ret


def squeeze_segnet(input, classes, keep_prob, nb_classes):
    """
    SqueezeNet model written in tensorflow. It provides AlexNet level accuracy with 50x fewer parameters
    and smaller model size.
    :param input: Input tensor (4D)
    :param classes: number of classes for classification
    :return: Tensorflow tensor
    """
    weights = {'conv1': tf.Variable(tf.truncated_normal([7, 7, 3, 96])),
               'conv10': tf.Variable(tf.truncated_normal([1, 1, 512, classes])),
               'conv10_D': tf.Variable(tf.truncated_normal([3, 3, classes, 512])),
               'conv1_D': tf.Variable(tf.truncated_normal([2, 2, nb_classes, 96]))}

    biases = {'conv1': tf.Variable(tf.truncated_normal([96])),
              'conv10': tf.Variable(tf.truncated_normal([classes])),
              'conv10_D': tf.Variable(tf.truncated_normal([512])),
              'conv1_D': tf.Variable(tf.truncated_normal([nb_classes]))}

    output_shape0 = input.get_shape().as_list()
    out = tf.shape(input)
    with tf.name_scope('Squeeze-SegNet'):
        with tf.name_scope('Conv1'):
            output = tf.nn.conv2d(input, weights['conv1'], strides=[1, 2, 2, 1], padding='SAME', name='conv1')
            output = tf.nn.relu(tf.nn.bias_add(output, biases['conv1']))
            print 'conv1'
            print output.get_shape().as_list()
            # output_shape1 = output.get_shape().as_list()
            out1 = tf.shape(output)
        with tf.name_scope('Maxpool1_with_indices'):
            output, pooling_indices1 = tf.nn.max_pool_with_argmax(output,
                                                                  ksize=[1, 3, 3, 1],
                                                                  strides=[1, 2, 2, 1],
                                                                  padding='SAME', name="maxpool1_with_mask")
            # output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')
            print 'maxpool'
            print output.get_shape().as_list()

        with tf.name_scope('Fire2'):
            output = fire_module(output, s1=16, e1=64, e3=64, channel=96, fire_id='fire2')
            print 'fire2'
            print output.get_shape().as_list()
        with tf.name_scope('Fire3'):
            output = fire_module(output, s1=16, e1=64, e3=64, channel=128, fire_id='fire3')
            print 'fire3'
            print output.get_shape().as_list()
        with tf.name_scope('Fire4'):
            output = fire_module(output, s1=32, e1=128, e3=128, channel=128, fire_id='fire4')
            print 'fire4'
            print output.get_shape().as_list()

        # output_shape4 = output.get_shape().as_list()
        out4 = tf.shape(output)

        with tf.name_scope('Maxpool4_with_mask'):
            # output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')
            output, pooling_indices4 = tf.nn.max_pool_with_argmax(output,
                                                                  ksize=[1, 3, 3, 1],
                                                                  strides=[1, 2, 2, 1],
                                                                  padding='SAME', name="maxpool4_with_mask")
            print 'maxpool4'
            print output.get_shape().as_list()
        with tf.name_scope('Fire5'):
            output = fire_module(output, s1=32, e1=128, e3=128, channel=256, fire_id='fire5')
            print 'fire5'
            print output.get_shape().as_list()
        with tf.name_scope('Fire6'):
            output = fire_module(output, s1=48, e1=192, e3=192, channel=256, fire_id='fire6')
            print 'fire6'
            print output.get_shape().as_list()
        with tf.name_scope('Fire7'):
            output = fire_module(output, s1=48, e1=192, e3=192, channel=384, fire_id='fire7')
            print 'fire7'
            print output.get_shape().as_list()
        with tf.name_scope('Fire8'):
            output = fire_module(output, s1=64, e1=256, e3=256, channel=384, fire_id='fire8')
            print 'fire8'
            print output.get_shape().as_list()
        # output_shape8 = output.get_shape().as_list()
        out8 = tf.shape(output)

        with tf.name_scope('Maxpool8_with_indices'):
            # output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool8')
            output, pooling_indices8 = tf.nn.max_pool_with_argmax(output,
                                                                  ksize=[1, 3, 3, 1],
                                                                  strides=[1, 2, 2, 1],
                                                                  padding='SAME', name="maxpool8_with_mask")
            print 'maxpool8'
            print output.get_shape().as_list()
        with tf.name_scope('Fire9'):
            output = fire_module(output, s1=64, e1=256, e3=256, channel=512, fire_id='fire9')
            print  'fire9'
            print output.get_shape().as_list()
        with tf.name_scope('Dropout'):
            output = tf.nn.dropout(output, keep_prob=keep_prob, name='dropout9')

        with tf.name_scope('Conv10'):
            output = tf.nn.conv2d(output, weights['conv10'], strides=[1, 1, 1, 1], padding='SAME', name='conv10')
            output = tf.nn.relu(tf.nn.bias_add(output, biases['conv10']))
            print 'conv10'
            print output.get_shape().as_list()

        with tf.name_scope('Conv10_D'):
            # output = tf.nn.avg_pool(output, ksize=[1, 13, 13, 1], strides=[1, 2, 2, 1], padding='SAME', name='avgpool10')
            output = tf.nn.conv2d(output, weights['conv10_D'], strides=[1, 1, 1, 1], padding='SAME', name='conv10_D')
            output = tf.nn.relu(tf.nn.bias_add(output, biases['conv10_D']))
            print 'conv10_D'
            print output.get_shape().as_list()
        with tf.name_scope('DFire9'):
            output = dfire_module(output, s1_D=512, e1_D=32, e3_D=32, channel=512, dfire_id='Dfire9')
            print 'dfire9'
            print output.get_shape().as_list()

        with tf.name_scope('Unpool8'):
            output = unpool(output, pooling_indices8, output_shape=out8, scope='unpool8')
            print 'unpool8'
            print output.get_shape().as_list()

        with tf.name_scope('DFire8'):
            output = dfire_module(output, s1_D=384, e1_D=32, e3_D=32, channel=512, dfire_id='Dfire8')
            print 'dfire8'
            print output.get_shape().as_list()
        with tf.name_scope('DFire7'):
            output = dfire_module(output, s1_D=384, e1_D=24, e3_D=24, channel=384, dfire_id='Dfire7')
            print 'dfire7'
            print output.get_shape().as_list()
        with tf.name_scope('DFire6'):
            output = dfire_module(output, s1_D=256, e1_D=24, e3_D=24, channel=384, dfire_id='Dfire6')
            print 'dfire6'
            print output.get_shape().as_list()
        with tf.name_scope('DFire5'):
            output = dfire_module(output, s1_D=256, e1_D=16, e3_D=16, channel=256, dfire_id='Dfire5')
            print 'dfire5'
            print output.get_shape().as_list()
        with tf.name_scope('Unpool4'):
            output = unpool(output, pooling_indices4, output_shape=out4, scope='unpool4')
            print 'unpool4'
            print output.get_shape().as_list()
        with tf.name_scope('DFire4'):
            output = dfire_module(output, s1_D=128, e1_D=16, e3_D=16, channel=256, dfire_id='Dfire4')
            print 'dfire4'
            print output.get_shape().as_list()
        with tf.name_scope('DFire3'):
            output = dfire_module(output, s1_D=128, e1_D=8, e3_D=8, channel=128, dfire_id='Dfire3')
            print 'dfire3'
            print output.get_shape().as_list()
        with tf.name_scope('DFire2'):
            output = dfire_module(output, s1_D=96, e1_D=8, e3_D=8, channel=128, dfire_id='Dfire2')
            print 'dfire2'
            print output.get_shape().as_list()
        with tf.name_scope('Unpool1'):
            output = unpool(output, pooling_indices1, output_shape=out1, scope='unpool1')
            print 'unpool1'
            print output.get_shape().as_list()

        with tf.name_scope('Conv1_D'):
            output = tf.nn.conv2d_transpose(output, weights['conv1_D'],
                                            [out[0], output_shape0[1], output_shape0[2], nb_classes], [1, 2, 2, 1],
                                            padding='SAME', name='conv1_D')
            output = tf.nn.bias_add(output, biases['conv1_D'])
            output = tf.nn.tanh(output, 'activation_finale')
            print 'conv1_D'
            print output.get_shape().as_list()
            probabilities = tf.nn.softmax(output, name='logits_to_softmax')
        return output, probabilities


def weighted_cross_entropy(onehot_labels, logits, class_weights):
    '''
    A quick wrapper to compute weighted cross entropy.
    ------------------
    Technical Details
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want.
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.
    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------
    INPUTS:
    - onehot_labels(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - logits(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    - class_weights(list): A list where each index is the class label and the value of the index is the class weight.
    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    '''

    weights = onehot_labels * class_weights
    weights = tf.reduce_sum(weights, 3)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)
    print weights

    return loss


def preprocess(image, annotation=None, height=360, width=480):
    # Convert the image and annotation dtypes to tf.float32 if needed
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.cast(image, tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image.set_shape(shape=(height, width, 3))

    if not annotation == None:
        annotation = tf.image.resize_image_with_crop_or_pad(annotation, height, width)
        annotation.set_shape(shape=(height, width, 1))

        return image, annotation

    return image


if __name__ == '__main__':

    # Getting images
    image_files = sorted(glob('/home/labogeraldo/PycharmProjects/tensorflow/CamVid/train/**'))
    annotation_files = sorted(glob('/home/labogeraldo/PycharmProjects/tensorflow/CamVid/trainannot/**'))
    with open('train_trainnot.txt', 'w') as f:
        for i in range(len(image_files)):
            f.write(image_files[i] + ' ' + annotation_files[i] + '\n')
    image_val_files = sorted(glob('/home/labogeraldo/PycharmProjects/tensorflow/CamVid/val/**'))
    annotation_val_files = sorted(glob('/home/labogeraldo/PycharmProjects/tensorflow/CamVid/valannot/**'))
    with open('val_valannot.txt', 'w') as f:
        for i in range(len(image_val_files)):
            f.write(image_val_files[i] + ' ' + annotation_val_files[i] + '\n')

    batch_size = 8
    eval_batch_size = 8

    # Training parameters
    initial_learning_rate = 0.001
    num_epochs_before_decay = 100
    num_epochs = 300
    learning_rate_decay_factor = 0.1
    weight_decay = 0.0001
    epsilon = 1e-8

    num_batches_per_epoch = len(image_files) / batch_size
    num_steps_per_epoch = num_batches_per_epoch
    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
    image_height = 360
    image_width = 480
    num_classes = 12

    # class_weights = median_frequency_balancing()
    class_weights = None

    tf.reset_default_graph()
    # x = tf.placeholder(tf.float32, shape=[batch_size, 360 , 480, 3], name="x-input")
    # prob=tf.placeholder(tf.float32, shape=[1], name='dropout_val')
    with tf.Graph().as_default() as graph:

        images = tf.convert_to_tensor(image_files)

        annotations = tf.convert_to_tensor(annotation_files)
        input_queue = tf.train.slice_input_producer(
            [images, annotations])  # Slice_input producer shuffles the data by default.

        # Decode the image and annotation raw content
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_image(image, channels=3)
        annotation = tf.read_file(input_queue[1])
        annotation = tf.image.decode_image(annotation)

        # preprocess and batch up the image and annotation
        preprocessed_image, preprocessed_annotation = preprocess(image, annotation)
        images, annotations = tf.train.batch([preprocessed_image, preprocessed_annotation], batch_size=batch_size,
                                             allow_smaller_final_batch=True)

        # Create the model inference
        print images.shape
        logits, probabilities = squeeze_segnet(input=images, classes=1000, keep_prob=1, nb_classes=num_classes)

        # perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations = tf.reshape(annotations, shape=[batch_size, image_height, image_width])
        annotations_ohe = tf.one_hot(annotations, num_classes, axis=-1)

        print "toto"
        print annotations_ohe.get_shape().as_list()
        print logits.get_shape().as_list()
        # Actually compute the loss
        print class_weights
        loss = weighted_cross_entropy(logits=logits, onehot_labels=annotations_ohe, class_weights=class_weights)
        total_loss = tf.losses.get_total_loss()

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)

        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, annotations)
        mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions, labels=annotations,
                                                                          num_classes=num_classes)
        metrics_op = tf.group(accuracy_update, mean_IOU_update)


        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, metrics_op):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()

            # for i in range(a.shape[0]):
            #    np.savetxt('activation'+str(i)+'.csv',a[i].reshape(a[i].shape[0]*a[i].shape[1],a[i].shape[2]),delimiter=',')
            total_loss, global_step_count, accuracy_val, mean_IOU_val, _ = sess.run(
                [train_op, global_step, accuracy, mean_IOU, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to show some results
            print 'global step %s: loss: %.4f (%.2f sec/step)    Current Streaming Accuracy: %.4f    Current Mean IOU: %.4f', \
                global_step_count, total_loss, time_elapsed, accuracy_val, mean_IOU_val

            return total_loss, accuracy_val, mean_IOU_val


        # ================VALIDATION BRANCH========================
        # Load the files into one input queue
        images_val = tf.convert_to_tensor(image_val_files)
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        input_queue_val = tf.train.slice_input_producer([images_val, annotations_val])

        # Decode the image and annotation raw content
        image_val = tf.read_file(input_queue_val[0])
        image_val = tf.image.decode_jpeg(image_val, channels=3)
        annotation_val = tf.read_file(input_queue_val[1])
        annotation_val = tf.image.decode_png(annotation_val)

        # preprocess and batch up the image and annotation
        preprocessed_image_val, preprocessed_annotation_val = preprocess(image_val, annotation_val, image_height,
                                                                         image_width)
        images_val, annotations_val = tf.train.batch([preprocessed_image_val, preprocessed_annotation_val],
                                                     batch_size=eval_batch_size, allow_smaller_final_batch=True)

        logits_val, probabilities_val = squeeze_segnet(input=images_val, classes=1000, keep_prob=1,
                                                       nb_classes=num_classes)

        # perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations_val = tf.reshape(annotations_val, shape=[eval_batch_size, image_height, image_width])
        annotations_ohe_val = tf.one_hot(annotations_val, num_classes, axis=-1)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded. ----> Should we use OHE instead?
        predictions_val = tf.argmax(probabilities_val, -1)
        accuracy_val, accuracy_val_update = tf.contrib.metrics.streaming_accuracy(predictions_val, annotations_val)
        mean_IOU_val, mean_IOU_val_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions_val,
                                                                                  labels=annotations_val,
                                                                                  num_classes=num_classes)
        metrics_op_val = tf.group(accuracy_val_update, mean_IOU_val_update)

        # Create an output for showing the segmentation output of validation images
        segmentation_output_val = tf.cast(predictions_val, dtype=tf.float32)
        segmentation_output_val = tf.reshape(segmentation_output_val, shape=[-1, image_height, image_width, 1])
        segmentation_ground_truth_val = tf.cast(annotations_val, dtype=tf.float32)
        segmentation_ground_truth_val = tf.reshape(segmentation_ground_truth_val,
                                                   shape=[-1, image_height, image_width, 1])


        def eval_step(sess, metrics_op):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, accuracy_value, mean_IOU_value = sess.run([metrics_op, accuracy_val, mean_IOU_val])
            time_elapsed = time.time() - start_time

            # Log some information
            print '---VALIDATION--- Validation Accuracy: %.4f    Validation Mean IOU: %.4f    (%.2f sec/step)', \
                accuracy_value, mean_IOU_value, time_elapsed

            return accuracy_value, mean_IOU_value


        # =====================================================

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/validation_accuracy', accuracy_val)
        tf.summary.scalar('Monitor/training_accuracy', accuracy)
        tf.summary.scalar('Monitor/validation_mean_IOU', mean_IOU_val)
        tf.summary.scalar('Monitor/training_mean_IOU', mean_IOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/Validation_original_image', images_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_output', segmentation_output_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_ground_truth', segmentation_ground_truth_val, max_outputs=1)
        my_summary_op = tf.summary.merge_all()

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir='logdir', summary_op=None, init_fn=None)

        # Run the managed session
        with sv.managed_session() as sess:
            for step in xrange(int(num_steps_per_epoch * num_epochs)):
                # At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    print 'Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs
                    learning_rate_value = sess.run([lr])
                    print 'Current Learning Rate: %s', learning_rate_value

                # Log the summaries every 10 steps or every end of epoch, which ever lower.
                if step % min(num_steps_per_epoch, 10) == 0:
                    loss, training_accuracy, training_mean_IOU = train_step(sess, train_op, sv.global_step,
                                                                            metrics_op=metrics_op)

                    # Check the validation data only at every third of an epoch
                    if step % (num_steps_per_epoch / 3) == 0:
                        for i in xrange(len(image_val_files) / eval_batch_size):
                            validation_accuracy, validation_mean_IOU = eval_step(sess, metrics_op_val)

                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                # If not, simply run the training step
                else:
                    loss, training_accuracy, training_mean_IOU = train_step(sess, train_op, sv.global_step,
                                                                            metrics_op=metrics_op)

            # We log the final training loss
            print 'Final Loss: %s', loss
            print 'Final Training Accuracy: %s', training_accuracy
            print 'Final Training Mean IOU: %s', training_mean_IOU
            print 'Final Validation Accuracy: %s', validation_accuracy
            print 'Final Validation Mean IOU: %s', validation_mean_IOU

            # Once all the training has been done, save the log files and checkpoint model
            print 'Finished training! Saving model to disk now.'
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
            photo_dir = 'photo_dir'
            if not os.path.exists(photo_dir):
                os.mkdir(photo_dir)

            # Plot the predictions - check validation images only
            print 'Saving the images now...'
            predictions_value, annotations_value = sess.run([predictions_val, annotations_val])

            for i in xrange(eval_batch_size):
                predicted_annotation = predictions_value[i]
                annotation = annotations_value[i]

                plt.subplot(1, 2, 1)
                plt.imshow(predicted_annotation)
                plt.subplot(1, 2, 2)
                plt.imshow(annotation)
                plt.savefig(photo_dir + "/image_" + str(i))

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     writer=tf.summary.FileWriter('output_graph', sess.graph)
    #     writer.close()