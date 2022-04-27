from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import pandas as pd
import numpy as np
import keras_tuner as kt
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, models
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils.loader import Loader
from utils.image import draw_around_box, draw_pose, get_area_of_interest
from utils.trainer import Trainer
from utils.additional import create_column, plot_the_loss_curve, plot_gp
from utils.additional import rotate_image
from nnmodel_keras import train_model, Placing, train_model_bay_opt, haltCallback, extract_history, KerasHyperModel
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from skimage.util import random_noise
from sklearn.metrics import roc_curve,auc

# Pandas Options
# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)

# 1. Init loader, this will load the (non-image) dataset into memory
loader = Loader()
print(f'Dataset has {len(loader)} grasp attempts.')

# # 2. Split into Training / Validation / Test set
training_set, validation_set, test_set = Trainer.split(loader.episodes, seed=42)
print(f'Training set length: {len(training_set)}')
print(f'Validation set length: {len(validation_set)}')
print(f'Test set length: {len(test_set)}')

a_space = np.linspace(-1.484, 1.484, 16)  # [rad] # Don't use a=0.0


df = np.zeros((len(training_set*5),32,32)) # multiply by 3 when with augmentation


# Collecting training image data 
# test_set_data = []
ind = 0
for index, episode in enumerate(training_set):
    action = loader.get_action(index, action_id=0)
    rgbd_image = loader.get_image(index, action_id=0, camera ='rd', as_float = True)
    area = get_area_of_interest(rgbd_image, action['pose'], size_cropped=(200, 200), size_result=(32, 32))
    area_noise = random_noise(area, mode='gaussian')#, amount = 0.1)
    area_flip_hor = cv2.flip(area, 0)
    area_flip_ver = cv2.flip(area, 1)
    area_aug = cv2.rotate(area, cv2.ROTATE_180)
    x_a = np.asarray(area_aug).astype(np.float32)
    x_hor = np.asarray(area).astype(np.float32)
    x_ver = np.asarray(area).astype(np.float32)
    x_n = np.asarray(area).astype(np.float32)
    x = np.asarray(area).astype(np.float32)
    for i in range(0,len(area)):

        df[i] = df[i].astype(object)    # Required conversion to fill in pandas data frame
        df[ind,i] = x[i]   # Adding the pixel matrix to given column at given index
        df[ind+1,i] = x_a[i]
        df[ind+2,i] = x_hor[i]
        df[ind+3,i] = x_ver[i]
        df[ind+4,i] = x_n[i]
    ind +=5
    #ind+=2

df_reshaped = df.reshape(-1,32,32,1)
print(len(df_reshaped))
print('Finished collecting training data')

# Collect testing data
#test_gripper_index= np.zeros(len(test_set))
test_set_df = np.zeros((len(test_set),32,32)) # multiply by 3 when with augmentation

for index, episode in enumerate(test_set):
    action = loader.get_action(index, action_id=0)
    rgbd_image = loader.get_image(index, action_id=0, camera ='rd', as_float = True)
    area = get_area_of_interest(rgbd_image, action['pose'], size_cropped=(200, 200), size_result=(32, 32))
    x = np.asarray(area).astype(np.float32)
    for i in range(0,len(area)):

        #df[i] = df[i].astype(object)    # Required conversion to fill in pandas data frame
        test_set_df[index,i] = x[i]   # Adding the pixel matrix to given column at given index

test_set_df_reshaped = test_set_df.reshape(-1,32,32,1)
#print(test_set_df)

# Collect validation data
validation_set_df = np.zeros((len(validation_set),32,32))

for index, episode in enumerate(validation_set):
    action = loader.get_action(index, action_id=0)
    rgbd_image = loader.get_image(index, action_id=0, camera ='rd', as_float = True)
    area = get_area_of_interest(rgbd_image, action['pose'], size_cropped=(200, 200), size_result=(32, 32))
    x = np.asarray(area).astype(np.float32)
    for i in range(0,len(area)):

        #df[i] = df[i].astype(object)    # Required conversion to fill in pandas data frame
        validation_set_df[index,i] = x[i]   # Adding the pixel matrix to given column at given index

validation_set_df_reshaped =validation_set_df.reshape(-1,32,32,1)


print('Finished collecting training.validation,test data')

# 4. Getting labels
#training_set_im = pd.DataFrame(test_set_data)
ind_r = 0
#train_set_reward = create_column(training_set, 'actions', 0, 'reward')
train_set_reward = np.zeros(len(training_set*5))
#train_set_reward = np.zeros(len(training_set))
for i in range (0,len(training_set)):
        train_set_reward[ind_r] = training_set[i]['actions'][0]['reward']
        train_set_reward[ind_r+1] = train_set_reward[ind_r]
        train_set_reward[ind_r+2] = train_set_reward[ind_r]
        train_set_reward[ind_r+3] = train_set_reward[ind_r]
        train_set_reward[ind_r+4] = train_set_reward[ind_r]
        #ind_r +=2
        ind_r+=5

validation_set_reward = create_column(validation_set, 'actions', 0, 'reward')
test_set_reward = create_column(test_set, 'actions', 0, 'reward')

validation_tp = (validation_set_df_reshaped, validation_set_reward['reward'])
    

test_inf = np.zeros((len(test_set),37,32,32))
for index, episode in enumerate(test_set):
	action = loader.get_action(index, action_id=0)
	d_image = loader.get_image(index,action_id=0,camera='rd', as_float= True)
	area = get_area_of_interest(d_image,action['pose'], size_cropped=(200,200), size_result=(32,32))
	#x = np.asaray(area).astype(np.float32)
	for i in range(0,16):
		rotx = rotate_image(area,a_space[i])
		x = np.asarray(rotx).astype(np.float32)
		for k in range(0,len(area)):
			test_inf[index,i,k] = x[k]
#print(validation_tp)
# 5 Deffining hyper parameters, creating model and training it

lr = 1e-4 # 1e-4 is the base case
epi = 100 # 1000 is the base case
bsi = 64 # 64 is the base case
mo = 0.9 # 0.9 is the base case

filepath = "C:/Users/User/Documents/UNI/MEng Project/PYFILES/latest_model.tf"
filepath_2 = "C:/Users/User/Documents/UNI/MEng Project/PYFILES/TensorBoard/"
#checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath, monitor='binary_crossentropy', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
#stopping_callback = haltCallback()

# Deffining and compiling the CNN model

NNmod=Placing()
mymodel=NNmod.define_grasp_model(3)
mymodel.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr, momentum = mo),
     			loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryCrossentropy(),tf.keras.metrics.SparseCategoricalAccuracy()])
#----------------------------------------
#Simple model training
epochs, loss, rbce, vloss, vbce, acc = train_model(mymodel, df_reshaped, epi, train_set_reward, validation_data = validation_tp, batch_size=bsi, 
                                                        callbacks = [cp_callback,
                                                                    #EarlyStopping(monitor='val_loss',patience=60),
                                                                    ReduceLROnPlateau(factor=0.2,verbose=1, patience=60, min_lr=5e-6),
                                                                    tf.keras.callbacks.TensorBoard(filepath_2)
                                                                    ])
#----------------------------------------
# Hyper parameter optimisation

hp = kt.HyperParameters()
#tuner = kt.BayesianOptimization(NNmod.build_model,objective=kt.Objective("reshape_binary_crossentropy", direction="min"),max_trials = 15,overwrite = True)
#tuner.search(df_reshaped, train_set_reward,epochs=epi, validation_data=validation_tp,batch_size=bsi, callbacks = [cp_callback,
#                                                                                                                ReduceLROnPlateau(factor=0.2,verbose=1, patience=60, min_lr=5e-6),
#                                                                                                                tf.keras.callbacks.TensorBoard(filepath_2)
#                                                                                                                ])

#tuner.results_summary()
#best_trial=tuner.oracle.get_best_trials()[0].trial_id
#loss, val_loss, bce, val_bce, acc, val_acc = extract_history(best_trial)



##------------------------- Old Optimizer Architecture------------------------------
#optimizer = BayesianOptimization(
#	f=nnfunc,
#	pbounds = pbounds,
#	verbose = 2,
#	random_state = 1)

#optimizer.maximize(init_points = 2, n_iter = 3,)

#for i, res in enumerate(optimizer.res):
#    print("Iteration {}: \n\t{}".format(i, res))
    

#print(optimizer.max)
##-----------------------------------------------------------------------------------

#epochs = np.linspace(0,epi,num=epi)

#plot_the_loss_curve(epochs, loss, bce, val_loss, val_bce)

#print(acc)
#mymodel.evaluate(x=test_set_df_reshaped, y=test_set_reward['reward'], batch_size=bsi)

#rewards = np.zeros((len(test_set),37,3))
#for index, episode in enumerate(test_set):
#	rewards[index] = mymodel.predict(test_inf[index])[0]
#print(rewards)
#Evaluating models performance
#out_max = np.zeros((len(rewards),37))
#temp2 = np.zeros((len(rewards)))
#for i in range(0,len(test_set)):
#	for k in range(0,37):
#		temp = np.max(rewards[i][k])
#		out_max[i][k] = temp
#	temp2[i] = np.max(out_max[i])
	#if(temp2[i]>treshold):
	#	temp2[i] = 1
	#else:
	#	temp2[i] = 0
#print(tf.math.confusion_matrix(test_set_reward['reward'],temp2))
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_set_reward['reward'],temp2)
#auc_keras = auc(fpr_keras, tpr_keras)

#Graph
#plt.figure(1)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve')
#plt.legend(loc='best')
#plt.show()


