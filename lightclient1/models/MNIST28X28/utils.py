# %%
import sys
import cv2
import pandas as pd
import numpy as np
import h5py
import zlib
# %%
################################
# Formatted print back to node
################################
def send_to_node(newModel_flag, initial_model, update_vector):
    if len(update_vector) == 0:
        print("VECTOR[]ENDVECTOR")
    else:
        print(len(update_vector))
        print(len(initial_model))
        print("VECTOR[", flush=True, end="")
        if newModel_flag:
            for i in range(len(update_vector) - 1):
                print(update_vector[i], flush=True, end=",")
            print(update_vector[-1], flush=True, end="")
        else:
            for i in range(len(update_vector) - 1):
                print(update_vector[i] - initial_model[i], flush=True, end=",")
            print(update_vector[-1] - initial_model[-1], flush=True, end="")
        print("]ENDVECTOR",end="\n",flush=True)
# %%
################################
# Reading dataframe
################################
def read_input(index):
    if len(sys.argv) < (index+1):
        raise Exception('No dataset path found')
# /Users/daeyeolkim/eunsu/work/FLoBC/data/PhysNet_UBFC_test.hdf5
    print("reading dataset...")
    # sys.argv[index] => 3번째 data path
    df = h5py.File(sys.argv[index], "r")
    print("dataset reading success")
    # df = pd.read_csv(sys.argv[index])
    # df = pd.read_csv("data.csv")
    if len(df) == 0:
        raise Exception('Empty dataset')
    return df

# %%
def readNewModel_flag(index):
    if len(sys.argv) < (index+1):
        raise Exception('No new model flag found')
    
    return sys.argv[index]

# %%
################################
# Reading weights list
################################
def read_weights(index):
    if len(sys.argv) < (index+1):
        raise Exception('No weights list found')
    print("this is weight",sys.argv[index])
    weights_list_path = sys.argv[index]
    weights_list = open(weights_list_path, "r").readline().split("|")
    if len(weights_list) == 0:
        raise Exception('Empty weights list')
    print('reading weight...')
    weights_list = [float(i) for i in weights_list] 
    print("done")
    return weights_list

# %%
def flattenWeights(model):
  arr = np.array(model.get_weights())
  for i in range (0, len(arr)):
          arr[i] = arr[i].flatten()

  arr = np.concatenate(arr)

  ## encoding start
#   arr_comp = arr*1000
#   min_val = np.min(arr_comp)
#   arr_comp += np.abs(np.min(arr_comp))
#   arr_int = arr_comp.astype(np.uint8)

#   min_len = int(np.ceil(np.sqrt(len(arr_int))))
#   min_tot_len = min_len**2

#   sub_len = int(min_tot_len - len(arr_int))
#   hyper_param = np.asarray([len(arr_int), np.abs(min_val)],dtype = np.uint8)
#   zero_arr = np.zeros(shape=sub_len,dtype=np.uint8)
#   concat = np.append(arr_int,zero_arr)
#   imaging = np.reshape(concat,newshape=(min_len,min_len))

#   encoded_param=[int(cv2.IMWRITE_JPEG_QUALITY),50]
#   result,encimg = cv2.imencode('.jpg',imaging,encoded_param)
#   encimg = np.append(encimg, hyper_param)
#   comp_encimg = zlib.compress(encimg.tobytes())
  ## encoding end

  list = arr.tolist()
  return list
#   return comp_encimg

# %%
def trainModel(model, data_train, label_train):
    print('#################### model training start ####################')
    model.fit(data_train, label_train, epochs=1, verbose=1)
    print('#################### model training done ####################')
    return model

# %%
def rebuildModel(new_model, list):
    # if (newModel_flag):
    #     list = []
    #     np.random.seed(0)
    #     list = np.random.uniform(low = -0.09, high = 0.09, size = new_model.count_params()).tolist()
    start = 0
    for i in range (0, len(new_model.layers)):
        # bound = np.array(new_model.layers[i].get_weights(), dtype="object").size
        bound = len(new_model.layers[i].get_weights())
        weights = []
        for j in range (0, bound):
            print("**********************************************")
            print("i",len(new_model.layers));
            print("j",bound);
            # print(new_model.layers[i])
            # print(new_model.layers[i].get_weights())
            print("**********************************************")
            size = (new_model.layers[i].get_weights()[j]).size
            print("newmodel layer size = ",size)
            arr = np.array(list[start:start+size])
            # arr = python_bus(weight)
            print("arr = ",arr)
            # print(new_model.layers[i].get_weights()[j])
            print("---list---")
            print(len(list))
            # return arr
            arr = arr.reshape(new_model.layers[i].get_weights()[j].shape)
            weights.append(arr)
            start += size
        if (bound > 0):
            new_model.layers[i].set_weights(weights)
    return new_model