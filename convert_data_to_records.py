# This python file converts the data to the tfrecords format

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import ujson as json
import numpy as np
from datetime import datetime
import os
import sys
import threading

tf.app.flags.DEFINE_string('train', 'data/corr/raw/train', 'Training data directory')
tf.app.flags.DEFINE_string('output', 'data/corr/records/train', 'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1, 'Number of shards in training TFRecord files')
tf.app.flags.DEFINE_integer('threads', 1, 'Number of threads to preprocess the images')
FLAGS = tf.app.flags.FLAGS
IGNORE_FILENAMES = ['.DS_Store']

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _convert_to_example(filename, data_buffer, label):
    sdata, tdata = data_buffer
    example = tf.train.Example(features=tf.train.Features(feature={
    'sdata': _float_feature(sdata),
    'tdata': _float_feature(tdata),
    'label': _float_feature([label])
  }))
    return example

def _convert_to_list(str_dict):
    signal = str_dict['signal']
    time_samples = str_dict['time_samples']
    return (signal, time_samples)

def _convert_to_numpy(str_dict):
    signal = np.reshape(np.array(str_dict['signal']),
                        (len(str_dict['signal']), 1))
    time_samples = np.reshape(np.array(str_dict['time_samples']),
                              (len(str_dict['time_samples']), 1))
    value = np.concatenate((signal, time_samples), axis=1)
    return value

def data_coder(str_dict):
    data = _convert_to_list(str_dict)
    label = str_dict['phi']
    return data, label

def _process_data(filename):
    with tf.gfile.FastGFile(filename, 'r') as f:
        dict_data = f.read()

    # Converting the string data to a dictionary
    str_dict = json.loads(dict_data)
    data, label = data_coder(str_dict)
    return data, label

def _process_data_files_batch(thread_index, ranges, name, filenames, num_shards):
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        if num_shards == 1:
          output_filename = '%s.tfrecords' % name
        else:
          output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
          filename = filenames[i]

          if filename.split('/')[-1] in IGNORE_FILENAMES:
            continue

          data, label = _process_data(filename)

          example = _convert_to_example(filename, data, label)
          writer.write(example.SerializeToString())
          shard_counter += 1
          counter += 1

          if not counter % 1000:
            print('%s [thread %d]: Processed %d of %d data files in thread batch.' %
                  (datetime.now(), thread_index, counter, num_files_in_thread))
            sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d data files to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d data files to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_data_files(name, filenames, num_shards):
    # Break all files into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, filenames, num_shards)
        t = threading.Thread(target=_process_data_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d data files in data set.' %
        (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _process_dataset(name, directory, num_shards):
    file_path = '%s/*' % directory
    filenames = tf.gfile.Glob(file_path)
    _process_data_files(name, filenames, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.threads, ('Please make the FLAGS.threads commensurate with FLAGS.train_shards')
  print('Saving results to %s' % FLAGS.output)

  _process_dataset('train', FLAGS.train, FLAGS.train_shards)

if __name__ == "__main__":
    tf.app.run()
