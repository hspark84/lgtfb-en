import os
import time
import tensorflow as tf
import numpy as np
from sys import argv
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import train_set_raw, valid_set_raw, test_set_raw,train_set_raw_da, valid_set_raw_da, test_set_raw_da
from pipeline import get_train_dataset_raw, get_eval_dataset_raw

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)


# hyper parameters 
BATCH_SIZE = 100
EPOCHS = 150
LEARNING_RATE = 1e-2
LR_DECAY_BASE = 1.02
#LEARNING_RATE = 1e-4
#LR_DECAY_BASE = 1.00
weight_decay = 1e-3
ALPHA = 1.0		# mixup alpha
N_CLASSES = 50 
DKP = 0.50		# keep prob for dropout
EMA_DECAY = 0.90

trial   = 0
CV_IDX  = 0 # [0,... 4]

tbpath = 'Tensorboard/lgtfb_' + str(CV_IDX) + '_' + str(trial)
mpath = 'Model/lgtfb_' + str(CV_IDX) + '_' + str(trial)

tf.gfile.MkDir(tbpath)
tf.gfile.MkDir(mpath)


def _get_filter_norm(filt):
  return tf.sqrt(tf.reduce_sum(filt*filt,[0,1,2],keep_dims=True)+1e-4)

# initialize center frequencies by using mel-scale
def init_cent_freq(num_chan, max_f = 22050.0):
  mel_f  = np.linspace(0, 1127*np.log(1+max_f/700), num_chan+1, True)   # num_chan+1 points from 0 to max_mel
  ori_f  = 700 * (np.exp(mel_f / 1127) - 1)
  return ori_f

def freq2tf(freq, nBins=128, level=7):
  freq   = freq / freq[-1]                         # in [0~1]
  centBin = nBins/2
  tree_freq = freq[centBin:centBin+1]
  for lev in range(1,level):
    for n in range(2**lev):
      step = nBins / (2**lev)
      st   = step*n
      ed   = step*(n+1)
      cr = (st+ed)/2
      tree_freq = np.concatenate((tree_freq, [(freq[cr]-freq[st])/(freq[ed]-freq[st])]))
  return tree_freq

def tf2freq(tree_freq, level=7, LS=3):
  freq = tf.zeros(shape=[1],dtype=tf.float32)
  for lev in range(level):
    N = freq.get_shape().as_list()[0]
    st = 2**lev-1
    ed = st*2 + 1
    freq = tf.reshape(tf.stack([freq,freq],1),shape=[N*2])
    split = tf.reshape(tf.stack([tf.log(tree_freq[st:ed]+1e-8),tf.log(1-tree_freq[st:ed]+1e-8)],1),shape=[N*2])
    freq = freq + split
  N     = freq.get_shape().as_list()[0]
  freq  = tf.stack([freq]*LS,1)
  split = tf.log(tf.constant([[1.0/LS]*LS],dtype=tf.float32))
  freq  = tf.reshape(freq + split, shape=[N*LS])
  nFreqDiff = tf.exp(freq)
  freq = tf.cumsum(nFreqDiff)
  maxfreq = freq[-1]
  freq = freq / maxfreq
  nFreqDiff = nFreqDiff / maxfreq
  return [freq, nFreqDiff]


def inverse_sigmoid(sig_out):
  return -np.log(1/sig_out -1 + 1e-8)


def LGTFB(X, kSize=2048, nBins=128, LS=3, nChan=3, is_training=False):
  # new
  init_freq      = init_cent_freq(nBins)
  level          = np.log2(nBins).astype(int)
  tree_freq_init = freq2tf(init_freq, nBins, level)
  tree_freq_init = inverse_sigmoid(tree_freq_init)
  tree_freq      = tf.nn.sigmoid(tf.get_variable('freq', [nBins-1], initializer=tf.constant_initializer(tree_freq_init)))
  [freq,nFreqDiff]  = tf2freq(tree_freq, level, LS)                

  freq           = tf.reshape(freq, [1,nBins*LS,1])        
  nFreqDiff      = tf.reshape(nFreqDiff, [1,nBins*LS,1]) * 2 
  # gamma parameter
  scale     = tf.nn.sigmoid(tf.get_variable('scale', [1,1,nChan], initializer=tf.zeros_initializer() ))
  shape     = tf.exp(tf.get_variable('shape', [1,1,nChan], initializer=tf.zeros_initializer()))
#  scale     = tf.nn.sigmoid(tf.get_variable('scale', [1,nBins*LS,nChan], initializer=tf.zeros_initializer() ))
#  shape     = tf.exp(tf.get_variable('shape', [1,nBins*LS,nChan], initializer=tf.zeros_initializer()))
  # make filters
  n      = tf.cumsum(tf.ones(shape=[kSize,nBins*LS,1],dtype=tf.float32),0)	# [2048x128x1] 
  gamma_1  = tf.pow(n/kSize,shape-1)
  gamma_2  = tf.exp(-np.pi*nFreqDiff*scale*n)
  gamma  = gamma_1 * gamma_2
  gamma  = gamma / tf.reduce_mean(gamma,0,keep_dims=True)
  tone   = tf.cos(np.pi*(freq*n))			
  kernel = gamma * tone				
  kernel = tf.reshape(kernel,[kSize,1,1,nBins*LS*nChan])
  kernel /= _get_filter_norm(kernel)
  # calc filter bank output
  fbank  = tf.nn.conv2d(X, kernel, [1,128,1,1], padding='VALID')
#  fbank  = tf.nn.conv2d(X, kernel, [1,256,1,1], padding='VALID')
  fbank  = tf.log(tf.abs(fbank)+1)
  tsX    = tf.shape(fbank)
  fbank  = tf.reshape(fbank, [tsX[0],tsX[1],nBins*LS,nChan])
  fbank  = tf.nn.max_pool(fbank, ksize=[1, 4, LS, 1], strides=[1, 4, LS, 1], padding='VALID') 
#  fbank  = tf.nn.max_pool(fbank, ksize=[1, 2, LS, 1], strides=[1, 2, LS, 1], padding='VALID') 
  return fbank


# equal-loudness normalization
def EN(X, depth_radius=5):
  slX         = X.get_shape().as_list()
  in_channels = slX[3]
  kernel      = tf.constant(1.0, shape=[depth_radius, depth_radius, in_channels, 1], dtype=tf.float32)
  weight      = tf.nn.softmax(tf.get_variable('weight',[2,3], initializer=tf.constant_initializer(np.array([[-1,-1,1],[-1,-1,1]]))))

  # means 
  mu_SEN   = tf.reduce_mean(X, [1], keep_dims=True)		# [B x 1 X F x C]
  mu_TEN   = tf.reduce_mean(X, [2], keep_dims=True)		# [B x T X 1 x C]
  mu_LSTEN = tf.nn.depthwise_conv2d(X,kernel,[1,1,1,1],padding='SAME') / (depth_radius*depth_radius) # [B x T x F x C]
  mu_ws  = weight[0,0]*mu_SEN  + weight[0,1]*mu_TEN + weight[0,2]*mu_LSTEN

  Xzm    = X - mu_ws
  Xzm2   = tf.square(Xzm)

  # variances
  var_SEN  = tf.reduce_mean(Xzm2, [1], keep_dims=True)		# [B x 1 X F x C]
  var_TEN  = tf.reduce_mean(Xzm2, [2], keep_dims=True)		# [B x T X 1 x C]
  var_LSTEN = tf.nn.depthwise_conv2d(Xzm2,kernel,[1,1,1,1],padding='SAME') / (depth_radius*depth_radius) # [B x T x F x C]
  var_ws  = weight[1,0]*var_SEN + weight[1,1]*var_TEN + weight[1,2]*var_LSTEN
  
  # normalization
  Xn          = tf.div(Xzm, tf.sqrt(var_ws+1e-8))
  return Xn

def CNN_JS3(X, is_training=False):
  # 1st convolutional layer
  h_conv1 = tf.layers.conv2d(X, 32, [5, 5], [1, 1], 'same', use_bias=False)
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME') 
  h_bn1   = tf.layers.batch_normalization(h_pool1, training=is_training)
  h_1     = tf.nn.relu(h_bn1)
  # 2nd convolutional layer
  h_conv2 = tf.layers.conv2d(h_1, 64, [5, 5], [1, 1], 'same', use_bias=False)
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME')
  h_bn2   = tf.layers.batch_normalization(h_pool2, training=is_training)
  h_2     = tf.nn.relu(h_bn2)
  # 3nd convolutional layer
  h_conv3 = tf.layers.conv2d(h_2, 128, [5, 5], [1, 1], 'same', use_bias=False)
  h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME')
  h_bn3   = tf.layers.batch_normalization(h_pool3, training=is_training)
  h_3     = tf.nn.relu(h_bn3)
  return h_1, h_2, h_3

# temporal average pooling and flatteing 2
def TAP_FLAT(X, kp = 1.0, RC = 32, is_training=False):
  tsX   = tf.shape(X)                                                   # [B x T x F x C]
  slX   = X.get_shape().as_list()
  Xtap  = tf.reduce_mean(X, 1)                                          # [B x F x C]
  Xr    = tf.reshape(Xtap, [tsX[0], slX[2]*slX[3]])                     # [B x FC]
  Xrd   = tf.nn.dropout(Xr, keep_prob = kp)                             # [B x FC]
  M     = tf.layers.dense(Xrd, RC, use_bias=False)                      # [B x RC]
  M     = tf.layers.batch_normalization(M, training=is_training)
  M     = tf.nn.relu(M)                                                 # [B x RC]
  return M

def get_getter(ema):
  def ema_getter(getter, name, *args, **kwargs):
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var
  return ema_getter

def Model(X, drop_kp=1.0, is_training=False, scope='Model', reuse=None, getter=None):
  with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
    GTFB = LGTFB(X, kSize=2048, nBins=128, LS=3, nChan=3, is_training=is_training)	
    GTFB = EN(GTFB,9) 
    c1, c2, c3 = CNN_JS3(GTFB, is_training)
    h3 = TAP_FLAT(c3, drop_kp, 256, is_training)
    h3 = tf.nn.dropout(h3, keep_prob=drop_kp)
    logit = tf.layers.dense(h3, N_CLASSES)
  return logit

def mixup(X, Y):
  tsX = tf.shape(X)
  weight   = tf.convert_to_tensor(np.random.beta(ALPHA,ALPHA,BATCH_SIZE), tf.float32)
  weight = weight[:tsX[0]]
  x_weight = tf.expand_dims(tf.expand_dims(tf.expand_dims(weight,-1),-1),-1)
  y_weight = tf.expand_dims(weight,-1)
  index    = tf.convert_to_tensor(np.random.permutation(BATCH_SIZE))
  index    = tf.boolean_mask(index, tf.cast(index, tf.int32) < tsX[0])
  X = (x_weight * X + (1.0-x_weight) * tf.gather(X, index)) / tf.sqrt(tf.square(x_weight) + tf.square(1-x_weight))
  Y = y_weight * Y + (1.0-y_weight) * tf.gather(Y, index)
  return [X, Y]

# Build our dataflow graph.
GRAPH = tf.Graph()
with GRAPH.as_default():
  ## placeholders
  is_training = tf.placeholder(tf.bool)
  drop_kp = tf.placeholder(tf.float32, shape=())

  ## input processing
  # datasets
  dataset_train = get_train_dataset_raw(train_set_raw_da[CV_IDX], batch_size=BATCH_SIZE)
  dataset_valid = get_eval_dataset_raw(valid_set_raw_da[CV_IDX])
  dataset_test  = get_eval_dataset_raw(test_set_raw_da[CV_IDX])
  # reinitializable iterator to get waveforms
  iterator        = tf.data.Iterator.from_structure(dataset_train.output_types, (tf.TensorShape([None,None,1,1]), tf.TensorShape([None]), tf.TensorShape([None])))
  iter_init_train = iterator.make_initializer(dataset_train)
  iter_init_valid = iterator.make_initializer(dataset_valid)
  iter_init_test  = iterator.make_initializer(dataset_test)
  next_element = iterator.get_next()
  WAVEFORMS  = next_element[0]
  LABELS     = next_element[1]
  WAV_ID     = next_element[2]
  # waveform normalization
  Xm, Xv = tf.nn.moments(WAVEFORMS, axes=[1], keep_dims=True)
  IX = tf.div((WAVEFORMS-Xm),tf.sqrt(Xv+1e-8))
  # mix-up
  LABELS_mix = tf.one_hot(LABELS, N_CLASSES, dtype=tf.float32)	# [B x 10]
  [IX, LABELS_mix] = tf.cond(is_training, lambda: mixup(IX, LABELS_mix), lambda: [IX, LABELS_mix])

  ## Model
  pred_Y = Model(IX, drop_kp, is_training)
  ## EMA (exponential moving averaging)
  ema_dec   = tf.placeholder(tf.float32, shape=())
  ema       = tf.train.ExponentialMovingAverage(decay=ema_dec, zero_debias=True)
  var_model = tf.get_collection('trainable_variables', 'Model')
  EMA_OP    = ema.apply(var_model)
  ema_Y = Model(IX, drop_kp, is_training, reuse=True, getter=get_getter(ema))

  ## objective function
  # sound classification loss
  sc_loss = tf.reduce_mean(-1.0 * tf.reduce_sum(LABELS_mix*tf.log(tf.nn.softmax(pred_Y) + 1e-10),1) )	# for mixup
  tf.summary.scalar("sc_loss", sc_loss)
  # l2_loss
  tvars = tf.trainable_variables()
  l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if 'dense' in v.name and 'kernel' in v.name])
  tf.summary.scalar("l2_loss", l2_loss)
  # final cost
  COST = sc_loss + weight_decay*l2_loss

  ## calc num of tvars 
  nvars = 0
  for var in tvars:
    sh = var.get_shape().as_list()
    print(var.name, sh)
    nvars += np.prod(sh)
  print(nvars, 'total variables')

  ## computing gradients and optimization
  lr = tf.placeholder(tf.float32, shape=())
# OPTIMIZER = tf.train.AdamOptimizer(LEARNING_RATE)	# default epsilon 1e-08
  OPTIMIZER = tf.train.AdamOptimizer(lr)	# default epsilon 1e-08
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    grads,_ = tf.clip_by_global_norm(tf.gradients(COST,tvars),1)	# compute gradients and do clipping
    APPLY_GRADIENT_OP = OPTIMIZER.apply_gradients(zip(grads, tvars))	# apply gradients

  # evaluation
  correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(LABELS_mix,1))	# for mixup accuracy
  ACCURACY = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
  # confusion matrix
  CONF_MAT = tf.confusion_matrix(LABELS, tf.argmax(pred_Y, 1), num_classes=N_CLASSES)
  # ema evaluation for EMA
  ema_correct_pred = tf.equal(tf.argmax(ema_Y, 1), tf.argmax(LABELS_mix,1))	# for mixup accuracy
  EMA_ACCURACY = tf.reduce_mean(tf.cast(ema_correct_pred, dtype=tf.float32))
  # confusion matrix for EMA
  EMA_CONF_MAT = tf.confusion_matrix(LABELS, tf.argmax(ema_Y, 1), num_classes=N_CLASSES)

  SUMMARIES_OP = tf.summary.merge_all()


# Start training the model.
with tf.Session(graph=GRAPH) as SESSION:
  # initialize first
  SESSION.run(tf.global_variables_initializer())
  # Create a tensorflow summary writer.
  SUMMARY_WRITER = tf.summary.FileWriter(tbpath, graph=GRAPH)
  # Create a tensorflow graph writer.
  GRAPH_WRITER = tf.train.Saver(max_to_keep=1)

  steps = 0     # for tf.summary stepB
  best_cost = float('Inf')
  best_acc  = float(0)
  best_model = 0

  train_acc_hist  = []
  valid_acc_hist  = []
  test_acc_hist   = []
  ema_valid_acc_hist  = []
  ema_test_acc_hist   = []

  for EPOCH in range(EPOCHS):
    # initialize an iterator over the training dataset.
    SESSION.run(iter_init_train)
    iters = 0
    costs = 0.0
    accs  = 0.0

    lr_decay = LR_DECAY_BASE ** (EPOCH)
    lr_epoch = LEARNING_RATE / lr_decay
    print('Epoch %d, Leanring rate = %.7f' % (EPOCH, lr_epoch))

    start_time = time.time()
    while True:
      try:
        _, summaries, COST_VAL, ACC_VAL = SESSION.run([APPLY_GRADIENT_OP, SUMMARIES_OP, COST, ACCURACY],
                           feed_dict={drop_kp: DKP, is_training: True, lr: lr_epoch})
        costs += COST_VAL
        accs  += ACC_VAL
        iters += 1
        if iters % 100 == 0:
          end_time = time.time()
          DURATION = end_time - start_time
          start_time = end_time
          print('Epoch %d, Iters %d, cost = %.6f (%.3f sec)' % (EPOCH, iters, (costs/iters), DURATION))
          SUMMARY_WRITER.add_summary(summaries, steps)
          steps += 1
      except tf.errors.OutOfRangeError:
        break
    end_time = time.time()
    DURATION = end_time - start_time
    SUMMARY_WRITER.add_summary(summaries, steps)
    steps += 1
    print('Epoch %d, Train cost = %.6f, acc = %.6f (%.3f sec)' % (EPOCH, (costs/iters), (accs/iters),DURATION))
    train_acc_hist.append(accs/iters)

    ### EMA ###
    if EPOCH == 0:
      ema_dec_tmp = 0
    else:
      ema_dec_tmp = EMA_DECAY
    _ = SESSION.run([EMA_OP], feed_dict={ema_dec: ema_dec_tmp})

    # Valid
    SESSION.run(iter_init_valid)
    iters = 0
    accs  = 0.0
    while True:
      try:
        ACC_VAL = SESSION.run(EMA_ACCURACY, feed_dict={drop_kp: 1.0, is_training: False, lr: 0.0})
        accs  += ACC_VAL
        iters += 1
      except tf.errors.OutOfRangeError:
        break
    final_acc  = (accs/iters)
    print('(EMA) Epoch %d, Valid acc = %.6f' % (EPOCH,  final_acc))
    ema_valid_acc_hist.append(final_acc)

    # update best model
    if EPOCH > 19:
      if final_acc > best_acc:
        print('Best epoch is changed from %d to %d, Acc %.4f to %.4f' % (best_model, EPOCH, best_acc, final_acc))
        best_acc = final_acc
        best_model = EPOCH
        GRAPH_WRITER.save(SESSION, mpath+'/model', global_step=EPOCH)   # save the best model

    # Test
    SESSION.run(iter_init_test)
    iters = 0
    accs  = 0.0
    while True:
      try:
        ACC_VAL = SESSION.run(EMA_ACCURACY, feed_dict={drop_kp: 1.0, is_training: False, lr: 0.0})
        accs  += ACC_VAL
        iters += 1
      except tf.errors.OutOfRangeError:
        break
    final_acc  = (accs/iters)
    print('(EMA) Epoch %d, Test acc = %.6f' % (EPOCH,  final_acc))
    ema_test_acc_hist.append(final_acc)

  # test with the best model
  GRAPH_WRITER.restore(SESSION, tf.train.latest_checkpoint(mpath))
  print('Test using the best model %d in %s' % (best_model, mpath))
  SESSION.run(iter_init_test)
  iters = 0
  accs = 0.0
  conf_mat = np.zeros((50,50)).astype(int)
  while True:
    try:
      [ACCS_VAL, CONF_MAT_VAL] = SESSION.run([EMA_ACCURACY, EMA_CONF_MAT], feed_dict={drop_kp: 1.0, is_training: False, lr: 0.0})
      accs += ACCS_VAL
      conf_mat = conf_mat + CONF_MAT_VAL
      iters += 1
    except tf.errors.OutOfRangeError:
      break
  print('Test Accuracy : %.4f' % (accs/iters))
  print(conf_mat)

