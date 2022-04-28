from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import keras
import visualkeras
from tensorflow.keras import layers, Model, models
from keras.callbacks import ModelCheckpoint
from utils.loader import Loader
from utils.image import draw_around_box, draw_pose, get_area_of_interest
from utils.trainer import Trainer
from utils.additional import create_column, plot_the_loss_curve
from nnmodel import create_model, train_model, Placing
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_curve,auc
#from orthographical import OrthographicImage
import time

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)

# 1. Init loader, this will load the (non-image) dataset into memory
loader = Loader()
print(f'Dataset has {len(loader)} grasp attempts.')


# 2. Split into Training / Validation / Test set
training_set, validation_set, test_set = Trainer.split(loader.episodes, seed=42)
print(f'Training set length: {len(training_set)}')
print(f'Validation set length: {len(validation_set)}')
print(f'Test set length: {len(test_set)}')
positive = 0
negative = 0
#train_gripper_index = np.zeros(len(training_set))
for i in range(0,len(training_set)):
    if training_set[i]['actions'][0]['reward']==1:
        positive +=1
    else:
        negative +=1
        
for i in range(0,len(validation_set)):
    if validation_set[i]['actions'][0]['reward']==1:
        positive +=1
    else:
        negative +=1
        
for i in range(0,len(test_set)):
    if test_set[i]['actions'][0]['reward']==1:
        positive +=1
    else:
        negative +=1  

print("POS samples: ",positive)
print("NE samples: ",negative)
# Predifining np dataframes for gripper index and image data
# Preparing the rotated data to account for robot rotation
size_input = (752, 480)
size_original_cropped = (200, 200)
size_output = (32, 32)
size_cropped = (110, 110)
size_rotated = (160, 160)
size_resized = (
    int(round(size_input[0] * size_output[0] / size_original_cropped[0])),
    int(round(size_input[1] * size_output[1] / size_original_cropped[1]))
    )
scale_factors = (
    float(size_original_cropped[0]) / size_output[0],
    float(size_original_cropped[1]) / size_output[1]
    )
a_space = np.linspace(-1.484, 1.484, 16)  # [rad] # Don't use a=0.0

        
test_gripper_index= np.zeros(len(test_set))
test_set_df = np.zeros((len(test_set),32,32))

def rotate_image(image, angle):
	rad_todeg = (angle*180)/np.pi
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, rad_todeg, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result

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
	
#test_inf_reshaped = test_inf.reshape(-1,32,32,1)	

for index, episode in enumerate(test_set):
    action = loader.get_action(index, action_id=0)
    rgbd_image = loader.get_image(index, action_id=0, camera ='rd', as_float = True)
    area = get_area_of_interest(rgbd_image, action['pose'], size_cropped=(200, 200), size_result=(32, 32))
    x = np.asarray(area).astype(np.float32)
    for i in range(0,len(area)):

        #df[i] = df[i].astype(object)    # Required conversion to fill in pandas data frame
        test_set_df[index,i] = x[i]   # Adding the pixel matrix to given column at given index

test_set_df_reshaped = test_set_df.reshape(-1,32,32,1)


print('Finished collecting training.validation,test data')

# 4. Getting labels

validation_set_reward = create_column(validation_set, 'actions', 0, 'reward')
test_set_reward = create_column(test_set, 'actions', 0, 'reward')

#validation_tp = (validation_set_df_reshaped, validation_set_reward)
    

#print(validation_tp)
# 5 Deffining hyper parameters, creating model and training it

learning_rate = 0.004386 # Baseline was 1e-4
epochs = 500 # Baseline for testing was 2000, possible that it should have been lower?
batch_size = 64 # Baseline was 64
singlethresh = 0.23333333333333336
filepath = "C:/Users/User/Documents/UNI/MEng Project/PYFILES/archive/models/full.tf"
cp_callback = ModelCheckpoint(filepath, monitor='binary_crossentropy', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=batch_size)






# Loading and preparing the model

model = models.load_model(filepath,compile=False)
tf.keras.models.save_model(model,"C:/Users/User/Documents/UNI/MEng Project/PYFILES/archive/models/full.h5")
print(model.summary())
print(model.input)
print(model.output)
visualkeras.layered_view(model)
NNmod = Model(inputs=model.input, outputs=model.output[0])
NNmod.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
   			loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryCrossentropy()])

# Testing phase
NNmod.evaluate(x=test_set_df_reshaped, y=test_set_reward['reward'], batch_size=batch_size)

# Calculate time taken to evaluate a data sample
start = time.perf_counter()
NNmod.predict(test_set_df_reshaped)
end = time.perf_counter()
timetaken = (end-start) / len(test_set_df_reshaped)

print("Timetaken: ",timetaken)

rewards = np.zeros((len(test_set),37,3))

# Comment Marker
for index, episode in enumerate(test_inf):
    rewards[index] = NNmod.predict(test_inf[index])
print("Average reward: ",np.mean(rewards))
#Evaluating models performance for ROC
out_max = np.zeros((len(rewards),37))
temp2 = np.zeros((len(rewards)))
for i in range(0,len(test_set)):
    for k in range(0,37):
        temp = np.max(rewards[i][k])
        out_max[i][k] = temp
        temp2[i] = np.max(out_max[i])
    if(temp2[i]>singlethresh):
        temp2[i] = 1
    else:
        temp2[i] = 0


#-----------------------------------
# Threshold optimisation algorithm
#
# out_max = np.zeros((len(rewards),37))
# temp2 = np.zeros((len(rewards)))
# tempauc=0
# tempthresh=0
# #!!! The strategy to find thresholds
# n_points = 400
# thresholds = np.linspace(0.1, 0.3,num=n_points)
# temp2 = NNmod.predict(test_set_df)
# print("AVERAGE: ",temp2)
# for j in range(0,len(thresholds)):
#     tt = np.zeros(len(temp2))
#     for i in range (0,len(temp2)):
#         for k in range(0,37):
#             temp = np.max(rewards[i][k])
#             out_max[i][k] = temp
#             temp2[i] = np.max(out_max[i])
#             tt[i] = max(temp2[i])
#             if tt[i]>thresholds[j]:
#                 tt[i] = 1
#             else:
#                 tt[i] = 0
#     fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_set_reward['reward'],tt)
#     print("Threshold: ",thresholds[j])
#     auc_keras = auc(fpr_keras, tpr_keras)
#     print("AUC:",auc_keras)
#     if auc_keras>0.5:
#         if auc_keras>tempauc:
#             tempthresh = thresholds[j]
#             tempauc = auc_keras
#         print("!!!!!!!!!!!")
# print(tt)
# print(np.mean(tt))
# print("Best AUC: ",tempauc)
# print("The threshold", tempthresh)


# ROC and Confusion Matrix
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_set_reward['reward'],temp2)
auc_keras = auc(fpr_keras, tpr_keras)
print(tf.math.confusion_matrix(test_set_reward['reward'],temp2))
#Graph
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='ROC (AUC = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()






