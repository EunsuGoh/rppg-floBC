import zlib
import numpy as np
import cv2
import sys

def decode ():
  dec_encimg = zlib.decompress(sys.argv[0])

  dec_np_ecimg = np.frombuffer(dec_encimg,dtype=np.uint8)
  dec_img = cv2.imdecode(dec_np_ecimg,1)[:,:,0]
  dec_arr = np.reshape(dec_img,-1)
  len_arr_int = dec_arr[-2]
  min_val = dec_arr[-1]
  dec_weight = dec_arr[:len_arr_int]
  weight = (dec_weight-np.abs(min_val))/1000
  print(weight);
  return weight

decode()