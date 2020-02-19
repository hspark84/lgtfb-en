data_sets_all_raw = [
    'Data/raw_esc_1.tfrecords',
    'Data/raw_esc_2.tfrecords',
    'Data/raw_esc_3.tfrecords',
    'Data/raw_esc_4.tfrecords',
    'Data/raw_esc_5.tfrecords'
]

data_sets_all_raw_da = [
    'Data/raw_esc_1.tfrecords',
    'Data/raw_esc_2.tfrecords',
    'Data/raw_esc_3.tfrecords',
    'Data/raw_esc_4.tfrecords',
    'Data/raw_esc_5.tfrecords',
    'Data/raw_esc_da_1.tfrecords',
    'Data/raw_esc_da_2.tfrecords',
    'Data/raw_esc_da_3.tfrecords',
    'Data/raw_esc_da_4.tfrecords',
    'Data/raw_esc_da_5.tfrecords'
]

train_set_raw = []
valid_set_raw = []
test_set_raw = []
for i in range(5):
  # test set
  test_idx = i
  test_set = [data_sets_all_raw[i]]
  test_set_raw.append(test_set)

  # valid set
  valid_idx = (i + 1) % 5
  valid_set = [data_sets_all_raw[valid_idx]]
  valid_set_raw.append(valid_set)

  # train set
  train_idx = range(5)
  train_idx = list(set(train_idx) - set([test_idx, valid_idx]))
  train_set = [data_sets_all_raw[i] for i in train_idx]
  train_set_raw.append(train_set)

train_set_raw_da = []
valid_set_raw_da = []
test_set_raw_da = []
for i in range(5):
  # test set
  test_idx = i
  test_set = [data_sets_all_raw_da[test_idx]]
  test_set_raw_da.append(test_set)

  # valid set
  valid_idx = (i + 1) % 5
  valid_set = [data_sets_all_raw_da[valid_idx]]
  valid_set_raw_da.append(valid_set)

  # train set
  train_idx = range(10)
  train_idx = list(set(train_idx) - set([test_idx, test_idx+5, valid_idx, valid_idx+5]))
  train_set = [data_sets_all_raw_da[k] for k in train_idx]
  train_set_raw_da.append(train_set)

