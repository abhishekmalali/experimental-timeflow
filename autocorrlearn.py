import tensorflow as tf
import numpy as np
import ujson as json
from tqdm import tqdm
import sys
import timeflow as tflow
from batchlayers import LSTMLayerBatch, MeanLayerBatch, RegLayerBatch
from inputs import inputs

tf.app.flags.DEFINE_string('train', 'data/corr/records/train/train.tfrecords', 'Train data')
tf.app.flags.DEFINE_string('train_ckpt', './ckpts/model.ckpt', 'Train checkpoint file')
tf.app.flags.DEFINE_string('train_logs', 'tmp/corr/2', 'Log directory')
tf.app.flags.DEFINE_integer('batch', 100, 'Batch size')
tf.app.flags.DEFINE_integer('steps', 100000, 'Number of training iterations')
FLAGS = tf.app.flags.FLAGS

def build_model(input_placeholder, input_size, hidden_layer_size, target_size):
    with tf.variable_scope('LSTM_layer'):
        lstm_layer = LSTMLayerBatch(input_size, hidden_layer_size, input_placeholder)
    with tf.variable_scope('Mean_Layer'):
        mean_layer = MeanLayerBatch(lstm_layer)
    with tf.variable_scope('Reg_Layer'):
        reg_layer = RegLayerBatch(hidden_layer_size, target_size, mean_layer)
    return reg_layer.get_outputs()

def train():
    data, labels = inputs(FLAGS.train, FLAGS.batch)
    input_size = 2
    hidden_layer_size = 10
    target_size = 1
    # Invoking the model building function
    outputs = build_model(data,
                          input_size,
                          hidden_layer_size,
                          target_size)

    with tf.variable_scope('RMSE'):
        rmse = tflow.utils.metrics.RMSE(outputs, labels)
    tf.summary.scalar("RMSE", rmse)
    summary_op = tf.summary.merge_all()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
    train_step = optimizer.minimize(rmse)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        summary_op = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(FLAGS.train_logs, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in tqdm(range(FLAGS.steps + 1)):
            sess.run(train_step)

            if step % 10 == 0:
                summary = sess.run(summary_op)
                writer.add_summary(summary, step)
                writer.flush()

            if step % 1000 == 0:
                saver.save(sess, FLAGS.train_ckpt)

        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    with tf.device('/gpu:0'):
        train()

if __name__ == '__main__':
  tf.app.run()
