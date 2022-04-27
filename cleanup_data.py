#from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import cv2
import pandas as pd
import numpy as np
from hough_transform import process_image_with_shadow, find_edges, process_image
import os
from utils.loader import Loader
from utils.image import draw_around_box, draw_pose, get_area_of_interest
from utils.trainer import Trainer
from utils.additional import store_image
from matplotlib import pyplot as plt

# Function to record dataset id's to list
def conv_ids_to_list(filepath):
    read1 = open(filepath,"r")
    lines = read1.readline()
    stringline = lines.splitlines()
    out = []
    temp = ""
    for i in range(0,len(lines)):
        if(stringline[0][i]!=','):
            temp += stringline[0][i]
        else:
            out.append(int(temp))
            temp = ""
            
    return out
# Function to compare histograms
def compare_histograms(base, compare_image):
    base = cv2.normalize(base,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    hist_base = cv2.calcHist(base, [0], None, [256], [0, 256])
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    #plt.hist(hist_base)
    #plt.title('Histogram for gray scale image')
    #plt.show()
    compare_image = cv2.normalize(compare_image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    hist_test1 = cv2.calcHist(compare_image, [0], None, [256], [0, 256])
    cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    #plt.hist(hist_test1)
    #plt.title('Histogram for test gray scale image')
    #plt.show()
    # cv.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    
    # Correlation comparison method between histograms was chosen
    # The treshold is chosen as abs(0.1)
    base_test1 = cv2.compareHist(hist_base, hist_test1, 0)
    return base_test1
# Function to create figure examples   
def test_and_produce():    
    loader = Loader()
    action_1 = loader.get_action(7, action_id=0)
    rgbd_image_1 = loader.get_image(7, action_id=0, camera='rd', as_float=True)
    area = get_area_of_interest(rgbd_image_1, action_1['pose'], size_cropped=(200, 200), size_result=(200, 200))
    
    action_2 = loader.get_action(6, action_id=0)
    rgbd_image_2 = loader.get_image(6, action_id=0, camera='rd', as_float=True)
    area_2 = get_area_of_interest(rgbd_image_2, action_2['pose'], size_cropped=(200, 200), size_result=(200, 200))
    
    #area = process_image_with_shadow(area)
    #area = find_edges(area)
    #area_2 = process_image_with_shadow(area_2)
    #area_2 = find_edges(area_2)
    cv2.imshow('base', area) 
    store_image(area,"histcompare")
    cv2.imshow('compare', area_2)
    store_image(area_2,"histcompare2")
    cv2.waitKey(0)
    compare_histograms(area, area_2) 

def main(argv):
    i = 0
    
    # Init loader, this will load the (non-image) dataset into memory and load initial empty box case
    loader = Loader()
    action_1 = loader.get_action(430, action_id=0)
    rgbd_image_1 = loader.get_image(6, action_id=0, camera='rd', as_float=True)
    empty_box = get_area_of_interest(rgbd_image_1, action_1['pose'], size_cropped=(200, 200), size_result=(200, 200))
    i = 0
    while i<len(loader):
        sample_action =  loader.get_action(i, action_id=0)   
        rgbd_image = loader.get_image(i, action_id=0, camera='rd', as_float=True)
        area = get_area_of_interest(rgbd_image, action_1['pose'], size_cropped=(200, 200), size_result=(200, 200))
        comparison = compare_histograms(empty_box, area)
        if comparison>0.01:
            # Report ID and show image
            print("Sample ID:", i)
            cv2.imshow('base', area) 
    # Removing entries
    #loader.remove_entry(0, action_id=0)
    # for i in range(0,len(loader)):
    #     if(i in out or i in out_2 or i in out_3):
    #          print("Removing: ",i)
    #          loader.remove_entry(i, action_id=0)
    #          print("Removed: ",i)
        
    # Check for updated grasp data set length
    #logout.close()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
    
    
# OLD implementation
# while i<len(loader):
#         # Load the new sample 
#         action = loader.get_action(i, action_id=0)
#         rgbd_image = loader.get_image(i, action_id=0, camera='rd', as_float=True)
#         draw_pose(rgbd_image, action['pose'])
#         area = get_area_of_interest(rgbd_image, action['pose'], size_cropped=(200, 200), size_result=(200, 200))
#         # Compare new sample with empty box and if treshold is met proceed
#         if(#(abs(compare_histograms(empty_box, area)>=0.01)) and 
#            (action['reward']==1)):
#            cv2.imshow('Comparison_image', area) 
#            cv2.waitKey(32)
#            areax = cv2.flip(area,0)
#            cv2.imshow('Flipped_Comparison_image', areax)
#            cv2.waitKey(32)
#            print("Store Image? Y=Yes")
#            save_input = input()
#            if(save_input=="Y"):
#                #store_image(area,str(i)+"img")
#                area = cv2.normalize(area, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#                areax = cv2.normalize(areax, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#                cv2.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/"+str(i)+".bmp",area)
#                cv2.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/flip"+str(i)+".bmp",areax)
               
#                logout.write(str(i))
#                logout.write(",")
#            print("Reward is: ",action['reward'])
#            print("Index: ",i)
#            print("\n \n \n \n")
#            #print("Delete this entry? 4 - Yes")
#            #delinput = int(input())
#            #if(delinput == 4):
#         #    print("Deleting")
#         #    loader.remove_entry(i, action_id = 0)
#         #else:
#         #    print("Moving on.")
#         i+=1
        
#     # Check for updated grasp data set length
#     logout.close()
#     print(f'Dataset has {len(loader)} grasp attempts.')
    
#     # Split into Training / Validation / Test set
#     training_set, validation_set, test_set = Trainer.split(loader.episodes, seed=42)
#     print(f'Training set length: {len(training_set)}')
#     print(f'Validation set length: {len(validation_set)}')
#     print(f'Test set length: {len(test_set)}')
    
#     return 0    