#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which is used to demostrate the tensorflow based
AI pipeline from training, test, monitoring and serving.

It could be used as an example to run in Kuberentes environment 

The codes are written based on MNINST examples from 
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

The codes to shutdown gracefully ps is gotten from
https://github.com/tensorflow/tensorflow/issues/4713

Usage:
#python mnistv1.py --ps_hosts=master:2222 --worker_hosts=worker1:2222,worker2:2222 --job_name=ps --task_index=0 --log_dir=./mnistlog --data_dir=./mnistdata --model_dir=./mnistmodel
#python mnistv1.py --ps_hosts=master:2222 --worker_hosts=worker1:2222,worker2:2222 --job_name=worker --task_index=0 --log_dir=./mnistlog --data_dir=./mnistdata --model_dir=./mnistmodel
#python mnistv1.py --ps_hosts=master:2222 --worker_hosts=worker1:2222,worker2:2222 --job_name=worker --task_index=1 --log_dir=./mnistlog --data_dir=./mnistdata --model_dir=./mnistmodel

View tensorboard:
#tensorboard --logdir=./mnistlog

Start tensor serving with trained model:
#bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=./mnistmodel

Test tensor serving:
#bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000 --concurrency=10
"""

import os
import math
import time
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100, "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_float("learning_rate", 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float("dropout_rate", 0.9, "Keep probability for training dropout")
tf.app.flags.DEFINE_integer("max_steps", 10000, "Number of steps to run trainer")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")
tf.flags.DEFINE_boolean("fake_data", False, "If true, uses fake data for unit testing")
tf.app.flags.DEFINE_string("data_dir", "MNIST_data", "Directory for storing mnist data")
tf.app.flags.DEFINE_string("model_dir", "trained_model", "Directory for storing the trained model")
tf.app.flags.DEFINE_string("log_dir", "training_log", "Directory for storing the training log")
tf.app.flags.DEFINE_string("model_version", "1", "The version of the model")

FLAGS = tf.app.flags.FLAGS
IMAGE_PIXELS = 28

def create_done_queue(i):
    """Queue used to signal death for i'th ps shard. Intended to have 
    all workers enqueue an item onto it to signal doneness."""
  
    with tf.device("/job:ps/task:%d" % (i)):
        return tf.FIFOQueue(len(FLAGS.worker_hosts.split(",")), tf.int32, shared_name="done_queue" + str(i))
  
def create_done_queues():
    return [create_done_queue(i) for i in range(len(FLAGS.ps_hosts.split(",")))]

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        #server.join()
        sess = tf.Session(server.target)
        queue = create_done_queue(FLAGS.task_index)
  
        # wait until all workers are done
        for i in range(len(FLAGS.worker_hosts.split(","))):
            sess.run(queue.dequeue())
            print("ps %d received done %d" % (FLAGS.task_index, i))
     
        print("ps %d: quitting"%(FLAGS.task_index))

    elif FLAGS.job_name == "worker":
        start_time = time.time()
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

            x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

            image_shaped_input = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 1])
            #tf.summary.image('input', image_shaped_input, 10)
                
            hid1_w = tf.Variable(
                tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], stddev=1.0 / IMAGE_PIXELS),
                name="hid1_w")
            hid1_b = tf.Variable(tf.constant(0.1, shape=[FLAGS.hidden_units]), name="hid1_b")
            hidden1 = tf.nn.relu(tf.nn.xw_plus_b(x, hid1_w, hid1_b))
            
            keep_prob = tf.placeholder_with_default(tf.constant(1.0, shape=[]), shape=[], name="keep_prob")
            dropped = tf.nn.dropout(hidden1, keep_prob)
            
            hid2_w = tf.Variable(
                tf.truncated_normal([FLAGS.hidden_units, 10], stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
                name="hid2_w")
            hid2_b = tf.Variable(tf.constant(0.1, shape=[10]), name="hid2_b")
            y = tf.identity(tf.nn.xw_plus_b(dropped, hid2_w, hid2_b))
            
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
            cross_entropy = tf.reduce_mean(diff)
            
            global_step = tf.Variable(0)

            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
                cross_entropy,
                global_step=global_step)

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            saver = tf.train.Saver()

            summary_op = tf.summary.merge_all()

            init_op = tf.global_variables_initializer()
     
            enq_ops = []
            for q in create_done_queues():
                qop = q.enqueue(1)
                enq_ops.append(qop)

            sv = tf.train.Supervisor(
                is_chief=(FLAGS.task_index == 0), 
                logdir=FLAGS.log_dir, 
                init_op=init_op,
                summary_op=summary_op, 
                saver=saver, 
                global_step=global_step, 
                save_model_secs=600)
                
            mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

            with sv.managed_session(server.target) as sess:
                step = 0

                while not sv.should_stop() and step < FLAGS.max_steps:

                    def feed_dict(train):
                        #Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
                        if train or FLAGS.fake_data:
                            xs, ys = mnist.train.next_batch(FLAGS.batch_size, fake_data=FLAGS.fake_data)
                            k = FLAGS.dropout_rate
                        else:
                            xs, ys = mnist.test.images, mnist.test.labels
                            k = 1.0
                        return {x: xs, y_: ys, keep_prob: k}

                    _, step = sess.run([train_step, global_step], feed_dict=feed_dict(True))

                    if step % 100 == 0:
                        print("global step: {}, accuracy:{}".format(step, 
                            sess.run(accuracy,
                                feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
                                #feed_dict=feed_dict(False))))

                # For export multiple inputs please refer to https://github.com/tensorflow/serving/issues/9
                if sv.is_chief:
                    sess.graph._unsafe_unfinalize()
                    classification_inputs = utils.build_tensor_info(x)
                    classification_outputs_classes = utils.build_tensor_info(y)
                
                    classification_signature = signature_def_utils.build_signature_def(
                        inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
                        outputs={
                            signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
                            signature_constants.CLASSIFY_OUTPUT_SCORES: classification_outputs_classes
                        },
                        method_name=signature_constants.CLASSIFY_METHOD_NAME)
                
                    tensor_info_x = utils.build_tensor_info(x)
                    tensor_info_y = utils.build_tensor_info(y)
                
                    prediction_signature = signature_def_utils.build_signature_def(
                        inputs={'images': tensor_info_x},
                        outputs={'scores': tensor_info_y},
                        method_name=signature_constants.PREDICT_METHOD_NAME)
                    export_path = os.path.join(
                        compat.as_bytes(FLAGS.model_dir),
                        compat.as_bytes(FLAGS.model_version))
                    builder = saved_model_builder.SavedModelBuilder(export_path)
                    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
                
                    builder.add_meta_graph_and_variables(
                        sess,
                        [tag_constants.SERVING],
                        signature_def_map={
                            'predict_images': prediction_signature,
                            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,},
                        clear_devices=True,
                        legacy_init_op=legacy_init_op)

                    sess.graph.finalize()
                    builder.save()

                #signal to ps shards that we are done
                for op in enq_ops:
                    sess.run(op)

            sv.stop()
        end_time = time.time()
        print('running time:{}'.format(end_time - start_time))


if __name__ == "__main__":
    tf.app.run()
