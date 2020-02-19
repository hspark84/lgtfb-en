import os
import glob
import muda
import jams
import ipdb

def deform_and_save(j_orig, deform, path):
  if hasattr(j_orig.sandbox.muda, '_audio'):
    jams_tmp = list(deform.transform(j_orig))[0]
    muda.save(path+'.wav', path+'.jams', jams_tmp)
 
parent_dir = 'Data/ESC-50-master/audio'
parent_dir_da = 'Data/ESC-50-master/audio_da'
try:
    os.makedirs(parent_dir_da)
except OSError:
    if not os.path.isdir(parent_dir_da):
        raise 

## deformation definition
# time stretching
D_TS_0p81 = muda.deformers.TimeStretch(rate=0.81)
D_TS_0p93 = muda.deformers.TimeStretch(rate=0.93)
D_TS_1p07 = muda.deformers.TimeStretch(rate=1.07)
D_TS_1p23 = muda.deformers.TimeStretch(rate=1.23)
# pitch shifting
D_PS_m3p5 = muda.deformers.PitchShift(n_semitones=-3.5)
D_PS_m2p5 = muda.deformers.PitchShift(n_semitones=-2.5)
D_PS_m2p0 = muda.deformers.PitchShift(n_semitones=-2.0)
D_PS_m1p0 = muda.deformers.PitchShift(n_semitones=-1.0)
D_PS_p1p0 = muda.deformers.PitchShift(n_semitones=1.0)
D_PS_p2p0 = muda.deformers.PitchShift(n_semitones=2.0)
D_PS_p2p5 = muda.deformers.PitchShift(n_semitones=2.5)
D_PS_p3p5 = muda.deformers.PitchShift(n_semitones=3.5)
# dynamic range compression
D_DRC_ms = muda.deformers.DynamicRangeCompression(preset='music standard')
D_DRC_fs = muda.deformers.DynamicRangeCompression(preset='film standard')
D_DRC_sp = muda.deformers.DynamicRangeCompression(preset='speech')
D_DRC_ra = muda.deformers.DynamicRangeCompression(preset='radio')

for fn in glob.glob(os.path.join(parent_dir, '*.wav')):
  print(fn)
  fn_wav = fn.split('/')[-1].split('.')[0]
  fn_da = os.path.join(parent_dir_da, fn_wav)
   
  # load audio with jams via muda
  jam = jams.JAMS()
  j_orig = muda.load_jam_audio(jam, fn, sr=44100)

  # deformation
  deform_and_save(j_orig, D_TS_0p81, fn_da+'_TS_0p81')
  deform_and_save(j_orig, D_TS_0p93, fn_da+'_TS_0p93')
  deform_and_save(j_orig, D_TS_1p07, fn_da+'_TS_1p07')
  deform_and_save(j_orig, D_TS_1p23, fn_da+'_TS_1p23')

  deform_and_save(j_orig, D_PS_m3p5, fn_da+'_PS_m3p5')
  deform_and_save(j_orig, D_PS_m2p5, fn_da+'_PS_m2p5')
  deform_and_save(j_orig, D_PS_m2p0, fn_da+'_PS_m2p0')
  deform_and_save(j_orig, D_PS_m1p0, fn_da+'_PS_m1p0')

  deform_and_save(j_orig, D_PS_p1p0, fn_da+'_PS_p1p0')
  deform_and_save(j_orig, D_PS_p2p0, fn_da+'_PS_p2p0')
  deform_and_save(j_orig, D_PS_p2p5, fn_da+'_PS_p2p5')
  deform_and_save(j_orig, D_PS_p3p5, fn_da+'_PS_p3p5')

  deform_and_save(j_orig, D_DRC_ms, fn_da+'_DRC_ms')
  deform_and_save(j_orig, D_DRC_fs, fn_da+'_DRC_fs')
  deform_and_save(j_orig, D_DRC_sp, fn_da+'_DRC_sp')
  deform_and_save(j_orig, D_DRC_ra, fn_da+'_DRC_ra')
      
   
