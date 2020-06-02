import cv2
import numpy as np
import time
import argparse

"""
Construct Argument Parser
"""
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input image")
args = vars(ap.parse_args())



'''
This function required for mouse callback function.
The funtion is used to draw box around object based on mouse click
'''
def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, topLeft_clicked, bottomRight_clicked
    #call mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        #initial value of points is reset
        if topLeft_clicked and bottomRight_clicked:
            topLeft_clicked = False
            bottomRight_clicked = False
            pt1 = (0,0)
            pt2 = (0,0)
        #get coordinates of top left corner
        if not topLeft_clicked:
            pt1 = (x,y)
            topLeft_clicked = True
        #get coordinates of bottom right corner
        elif not bottomRight_clicked:
            pt2 = (x,y)
            bottomRight_clicked = True
'''
Laplace based hash algorithm.
The algoritm is used to produce hash code based on attain image.
Laplace operator is used to enhance the intensities of edge.
'''
def lhash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resize_gray = cv2.resize(gray,(8,8))
    laplacian = cv2.Laplacian(resize_gray,cv2.CV_64F)
    abs_laplacian=cv2.convertScaleAbs(laplacian)
    ret,binary = cv2.threshold(abs_laplacian,127,255,cv2.THRESH_BINARY)
    return sum([2 ** i for (i, v) in enumerate(binary.flatten()) if v])

def phash(image):
    """
    Perceptual Based Hash function is developed to make qualitative comparison 
    with Laplace hash function
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resize_gray = cv2.resize(gray,(8,8))
    dct=cv2.dct(np.float32(resize_gray))
    dct_avg=np.mean(dct)
    result_array=(dct < dct_avg)*dct
    return sum([2 ** i for (i, v) in enumerate(result_array.flatten()) if v])

def ahash(image):
    """
    Average Based Hash function is developed to make qualitative comparison 
    with Laplace hash function
    """
    resize_image= cv2.resize(image,(8,8))
    gray_result = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    gray_avg=int(np.mean(gray_result))
    ret,binary_mean = cv2.threshold(gray_result,127,255,gray_avg,cv2.THRESH_BINARY)
    return sum([2 ** i for (i, v) in enumerate(binary_mean.flatten()) if v])
'''
Sliding Window function is used to perform computation search 
in frame based on window size
'''
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
'''
This function is used to compute hamming distance between two hash code
'''
def hammingDistance(x, y):
      ans = 0
      for i in range(63,-1,-1):
         h1= x>>i&1
         h2 = y>>i&1
         ans+= not(h1==h2)
      return ans

'''
Main Function of the program
'''
#capture video based on defined path/location of dataset
cap = cv2.VideoCapture(args["video"])
cv2.namedWindow('Hashing',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Hashing', 200,200)
cv2.setMouseCallback('Hashing', draw_rectangle)


#initiate condition of first frame of video
firstFrame = True

while True: 
    #reset all callback variables
    pt1 = (0,0)
    pt2 = (0,0)
    topLeft_clicked = False
    bottomRight_clicked = False
    ret, frame = cap.read()

    #capture and draw box on first object in the first frame/scene
    while firstFrame: 
        cv2.imshow('Hashing', frame)
        if topLeft_clicked: 
            cv2.circle(frame, center=pt1, radius=1, color=(0,255,0), thickness=-1)
        if topLeft_clicked and bottomRight_clicked: 
            cv2.rectangle(frame, pt1, pt2, (0,255,0), 1)     
        refPt=[]
        refPt.append(pt1)
        refPt.append(pt2)
        roi_firstframe=frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        #press "C" to continue to the next frame
        if cv2.waitKey(1) &0xFF == ord('c'):
            firstFrame = False     
            #cv2.imwrite('first_frame.jpg', roi_firstframe)
            #img = cv2.imread('first_frame.jpg')
            #print(type(lhash(roi_firstframe)))
            #print(lhash(roi_firstframe))
            break
    
    key=cv2.waitKey(100) &0xFF
    #press "P" to pause the video and capture object in the current frame
    if key == ord('p'):
        print('frame number:',cap.get(cv2.CAP_PROP_POS_FRAMES)) #print current frame number
        while True:
            cv2.imshow('Hashing', frame)
            if topLeft_clicked: 
                cv2.circle(frame, center=pt1, radius=1, color=(0,255,0), thickness=-1)
            if topLeft_clicked and bottomRight_clicked: 
                cv2.rectangle(frame, pt1, pt2, (0,255,0), 2)
            refPt=[]
            refPt.append(pt1)
            refPt.append(pt2)
            roi_nextframe=frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

            #Define sliding window size
            w_width=32
            w_height=32
            distance_lhash=[]
            distance_phash=[]
            distance_ahash=[]
            refPt_nextframe_stack=[]
            
            #start the sliding windows search
            for (x, y, window) in sliding_window(roi_nextframe, stepSize=8, windowSize=(w_width, w_height)):
                clone = roi_nextframe.copy()
                if window.shape[0] != w_height or window.shape[1] != w_width:
                    continue              
                cv2.rectangle(clone, (x, y), (x + w_width, y + w_height), (0, 255, 0), 1)
                refPt_nextframe=[]
                refPt_nextframe.append((x,y))
                refPt_nextframe.append((x + w_width, y + w_height))
                roi_2=clone[refPt_nextframe[0][1]:refPt_nextframe[1][1], refPt_nextframe[0][0]:refPt_nextframe[1][0]]
                #convert the image from first frame and current frame to hash code
                #calculate the hamming distance
                result_lhash=hammingDistance(lhash(roi_firstframe),lhash(roi_2))
                result_phash=hammingDistance(phash(roi_firstframe),phash(roi_2))
                result_ahash=hammingDistance(ahash(roi_firstframe),ahash(roi_2))
                distance_lhash.append(result_lhash)
                distance_phash.append(result_phash)
                distance_ahash.append(result_ahash)       

                refPt_nextframe_stack.append(((x,y),(x + w_width, y + w_height)))   
                time.sleep(0.025)
                
            key2 = cv2.waitKey(1) or 0xff
            #press "P" to continue video
            if key2 == ord('p'):
                rectangle_track_lhash=list(refPt_nextframe_stack[distance_lhash.index(min(distance_lhash))])
                rectangle_track_phash=list(refPt_nextframe_stack[distance_phash.index(min(distance_phash))])
                rectangle_track_ahash=list(refPt_nextframe_stack[distance_ahash.index(min(distance_ahash))])
                
                #draw rectangle
                cv2.rectangle(clone, rectangle_track_lhash[0], rectangle_track_lhash[1], (0, 255, 0), 1) #green for lhash
                cv2.rectangle(clone, rectangle_track_phash[0], rectangle_track_phash[1], (0, 0, 255), 1) #red for phash
                cv2.rectangle(clone, rectangle_track_ahash[0], rectangle_track_ahash[1], (255, 0, 0), 1) #blue for ahash
                cv2.imshow("Re-Identification", clone)
                cv2.waitKey(0)
                break

    #show the main window
    cv2.imshow('Hashing', frame)
    #press "Q" to exit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()