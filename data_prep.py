import os
import ipdb
from audioread import NoBackendError

import librosa
import numpy as np
import tensorflow as tf
import csv

def bytes_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a byte array.
  '''

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a 64 bit integer.
  '''

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def prepare_tfrecord_raw(example_paths, destination_path):
  # Open a TFRecords file for writing.
  writer = tf.python_io.TFRecordWriter(destination_path)
  for idx in range(len(example_paths)):
    # Load an audio file for preprocessing.
    print('Extracting %s' % example_paths[idx])
    try:
      samples, sr = librosa.core.load(example_paths[idx], sr=44100)
      samples, _  = librosa.effects.trim(samples)
      samples = (samples - samples.mean()) / (10*samples.std() + np.finfo(np.float).eps)
    except NoBackendError:
      print('Warning: Could not load {}.'.format(example_paths[idx]))
      continue

    # parsing filename
    wav_id   = os.path.split(example_paths[idx])[-1].split('-')[1]
    label    = int(os.path.split(example_paths[idx])[-1].split('.')[0].split('-')[-1].split('_')[0])

    example = tf.train.Example(features=tf.train.Features(feature={
      'wavform': bytes_feature(samples.flatten().tostring()),
      'label': int64_feature(label),
      'wav_id': bytes_feature(wav_id)
    }))
    writer.write(example.SerializeToString())
  writer.close()


# original dataset
SOUND_FILE_DIR = 'Data/ESC-50-master/audio'
for fold in range(5):
  wav_paths = [os.path.join(SOUND_FILE_DIR, f) for f in os.listdir(SOUND_FILE_DIR) if f.endswith('.wav') and f.startswith(str(fold+1))]
  dest_path = os.path.join('Data/raw_esc_' + str(fold+1) +'.tfrecords')
  prepare_tfrecord_raw(wav_paths, dest_path)

# augmented dataset
SOUND_FILE_DIR_DA = 'Data/ESC-50-master/audio_da'
#for fold in range(5):
for fold in range(1,2):	# for only the second gt_da
  wav_paths = [os.path.join(SOUND_FILE_DIR_DA, f) for f in os.listdir(SOUND_FILE_DIR_DA) if f.endswith('.wav') and f.startswith(str(fold+1))]
  dest_path = os.path.join('Data/raw_esc_da_' + str(fold+1) +'.tfrecords')
  prepare_tfrecord_raw(wav_paths, dest_path)




