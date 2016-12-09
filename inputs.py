import tensorflow as tf

def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
    serialized_example,
    features={
      'tdata': tf.FixedLenFeature([300, 1], tf.float32),
      'sdata': tf.FixedLenFeature([300, 1], tf.float32),
      'label': tf.FixedLenFeature([1], tf.float32)
    })
    tdata = features['tdata']
    sdata = features['sdata']
    label = features['label']
    # Concatenating the data to build a 300x2 tensor
    data = tf.concat(1, [tdata, sdata])
    return data, label

def inputs(train_filename, batch_size):
    data, label = read_and_decode_single_example(train_filename)
    data_batch, labels_batch = tf.train.shuffle_batch(
    [data, label], batch_size=batch_size,
    capacity=2000,
    min_after_dequeue=1000)
    return data_batch, labels_batch

"""
def main(unused_argv):
    train_filename = 'data/corr/records/train/train.tfrecords'
    batch_size = 10
    input_ = inputs(train_filename, batch_size)

if __name__ == '__main__':
    tf.app.run()
"""
