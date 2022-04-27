import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.loader import Loader
from utils.image import get_area_of_interest
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from matplotlib import gridspec
import cv2
#%matplotlib inline
# Function to store image
def store_image(image, name):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/"+name+".png",image)
# Function to create image borders
def getBordered(image, width):
    bg = np.zeros(image.shape)
    _, contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = 0
    bigcontour = None
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area > biggest:
            biggest = area
            bigcontour = contour
    return cv2.drawContours(bg, [bigcontour], 0, (255, 255, 255), width).astype(bool)
# Create pandas column from grasping dataset
def create_column(datain,firstlabel,firstindex,secondlabel):
    retreived_data=[]    # List to contain the required features
    for i in range (0,len(datain)):
        retreived_data.append(datain[i][firstlabel][firstindex][secondlabel])
    out_data_set=pd.DataFrame(retreived_data,columns = [secondlabel])
    return out_data_set
# Create training data
def create_data_set(datain,firstlabel,firstindex):
    retreived_data=[]
    for i in range (0,len(datain)):
        retreived_data.append(datain[i][firstlabel][firstindex])
    out_data_set = pd.DataFrame(retreived_data)
    return out_data_set

# Function to rotate the image
def rotate_image(image, angle):
	rad_todeg = (angle*180)/np.pi
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, rad_todeg, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result
# Function to plot loss curve
def plot_the_loss_curve(epochs, loss, rbce, vloss, vbce):
  #Plot a curve of loss vs. epoch.

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.plot(epochs, loss, 'r', label="Loss")
  plt.legend()
  plt.show()
  
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Binary Cross-Entropy")
  plt.plot(epochs, rbce, 'g', label = "Binary Cross-Entropy")
  plt.legend()
  plt.show()
  
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Validation Loss")
  plt.plot(epochs, vloss, 'b', label = "Validation Loss")
  plt.legend()
  plt.show()
  
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Validation Binary Cross-Entropy")
  plt.plot(epochs,vbce, 'c', label = "Validation Binary Cross-Entropy")
  plt.legend()
  plt.show()
# Obsolete functions for old Bayesian Optimisation implementation
def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    x_obs = np.array([[res["params"]["lr"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
