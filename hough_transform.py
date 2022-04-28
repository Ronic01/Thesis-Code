import sys
import math
import cv2 as cv
import numpy as np
from utils.loader import Loader
from utils.image import draw_around_box, draw_pose, get_area_of_interest, draw_grasp_line
from utils.additional import getBordered
# Convert image to the type for opencv
def process_image(image):
    area = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # fgbg = cv.createBackgroundSubtractorMOG2(500,200.0)
    # area = fgbg.apply(area)
    #fgbg.getBackgroundImage(area2)
    
    # Dilate and Blur image, return normalised image
    dilated_img = cv.dilate(area, np.ones((7,7), np.uint8))
    bg_img = cv.medianBlur(dilated_img, 21)
    diff_img = 255 - cv.absdiff(area, bg_img)
    norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    return norm_img
 # Canny edge detection algorithm 
def find_edges(image):
    dst = cv.Canny(image, 100, 200, None,3, True)
    dst = cv.dilate(dst, None)
    dst = cv.erode(dst, None)
    return dst
# Processing method to remove background and shadows
def process_image_with_shadow(image):
    # Create and apply background subtractor
    img = process_image(image)
    fgbg = cv.createBackgroundSubtractorMOG2()
    img = fgbg.apply(img)
    return img
# Find object visual geometric centre
def get_centroid(image):
    # convert the grayscale image to binary image
    ret,thresh = cv.threshold(image,127,255,0)

    # find contours in the binary image
    contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.circle(image, (cX, cY), 5, (0, 0, 255), -1)

   # display the image
    cv.imshow("Image with added centroid", image)
    cv.waitKey(0)
    return image        
# Main
def main(argv):
    
    # Load a grasp image
    loader = Loader()
    episode_index=0 
    action = loader.get_action(episode_index, action_id=0)
    rgbd_image = loader.get_image(episode_index, action_id=0, camera='rcd', as_float=True)
    draw_around_box(rgbd_image, action['box_data'])
    grey_image = loader.get_image(episode_index, action_id=0, camera='rd', as_float=True)
    img = loader.get_image(episode_index, action_id=0, camera='rd', as_float=True)
    draw_around_box(grey_image, action['box_data'])
    draw_pose(img, action['pose'])
    cv.imshow('image', rgbd_image.mat)  # OpenCV uses uint8 format
    cv.waitKey(0)
    area = get_area_of_interest(rgbd_image, action['pose'], size_cropped=(200, 200), size_result=(200, 200))  # original 32x32[px]
    area_g = get_area_of_interest(grey_image, action['pose'], size_cropped=(200, 200), size_result=(200, 200))
    area_is = get_area_of_interest(img, action['pose'], size_cropped=(200, 200), size_result=(200, 200))
    cv.imshow('image1', area_is)  # OpenCV uses uint8 format
    cv.waitKey(0)
    # Image processing
    empty_img = np.zeros((200,200,3),np.uint8)
    area_i = process_image(area)
    area_g = process_image(area_g)
    area_g = get_centroid(area_g)
    #fgbg = cv.createBackgroundSubtractorMOG2()
    #area = fgbg.apply(area)
    
    # Making copies
    area_0 = np.copy(area_i)
    cv.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/im2.png",area_0)
    area_1 = np.copy(area_i)
    cv.imshow("dst",area_i)
    cv.waitKey(0)
    
    # Edge detection and reffining
    dst = find_edges(area_i)
    cv.imshow("dst",area_g)
    cv.waitKey(0)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    # Find the Hough Lines
    lines = cv.HoughLines(dst, 2, np.pi / 180, 100, None, 50, 10)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
            cv.line(area_0, pt1, pt2, (0,0,0), 1, cv.LINE_AA)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):

            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            #cv.line(empty_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            #cv.imshow("cdstP", cdstP)
            #cv.imshow("empty", empty_img)
            #cv.waitKey(0)
            #cv.line(area_1, pt1, pt2, (0,0,0), 1, cv.LINE_AA)
    l0 = linesP[0][0]
    l1 = linesP[1][0]
    
    x1_avg = (l0[0]+l0[2])/2
    y1_avg = (l0[1]+l0[3])/2
    
    x2_avg = (l1[0]+l1[2])/2
    y2_avg = (l1[1]+l1[3])/2
    print(x1_avg,y1_avg,'\n',x2_avg,y2_avg)
    # Display and record images
    cv.line(area_is, (int(x1_avg), int(y1_avg)), (int(x2_avg), int(y2_avg)), (0,0,255), 3, cv.LINE_AA)
    cv.imshow("empty", area_is)
    area_is = process_image(area_is)
    cv.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/autolinescentre.bmp",area_is)
    cv.waitKey(0)
     
    cv.imshow("Source", area)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    #cv.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/im3.png",cdst)
    cv.imshow("im standard", area_0)
    #cv.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/im4.png",area_0)
    cv.imshow("im probabilistic", area_1)
    #cv.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/im5.png",area_1)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    #cv.imwrite(r"C:/Users/User/Documents/UNI/MEng Project/PYFILES/plots/im6.png",cdstP)
    
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])