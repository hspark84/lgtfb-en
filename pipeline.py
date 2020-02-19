import tensorflow as tf

def _parse_function_train_raw(example_proto):
  # parsing
  features = {'wavform': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64),
              'wav_id': tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)

  # decode waveform
  waveform = tf.decode_raw(parsed_features['wavform'], tf.float32)
  waveform = tf.reshape(waveform, [-1,1,1])		# 1D sequence with variable length

  # repeating
  waveform = tf.cond(tf.less(tf.shape(waveform)[0], 66650),
                     lambda: tf.tile(waveform, [tf.floordiv(66650,tf.shape(waveform)[0])+1, 1, 1]),
                     lambda: waveform)

  # cropping
  waveform = tf.random_crop(waveform, [66650,1,1]) 

  # label
  label    = tf.cast(parsed_features['label'], tf.int64)

  # waf_id
  wav_id = parsed_features['wav_id']

  return waveform, label, wav_id


def _parse_function_eval_raw(example_proto):
  # parsing
  features = {'wavform': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64),
              'wav_id': tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)

  # decode waveform
  waveform = tf.decode_raw(parsed_features['wavform'], tf.float32)
  waveform = tf.reshape(waveform, [-1,1,1])		# 1D sequence with variable length

  # repeating
  waveform = tf.cond(tf.less(tf.shape(waveform)[0], 66650),
                     lambda: tf.tile(waveform, [tf.floordiv(66650,tf.shape(waveform)[0])+1, 1, 1]),
                     lambda: waveform)

  # label
  label    = tf.cast(parsed_features['label'], tf.int64)

  # waf_id
  wav_id = parsed_features['wav_id']

  return waveform, label, wav_id

def get_train_dataset_raw(file_paths, batch_size=100):
  dataset = tf.data.TFRecordDataset(file_paths)
  dataset = dataset.map(_parse_function_train_raw, num_parallel_calls=4)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(batch_size)
  return dataset

def get_eval_dataset_raw(file_paths, batch_size=1):
  dataset = tf.data.TFRecordDataset(file_paths)
  dataset = dataset.map(_parse_function_eval_raw, num_parallel_calls=4)
  dataset = dataset.batch(batch_size)
  return dataset




