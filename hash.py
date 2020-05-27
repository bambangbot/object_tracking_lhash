import cv2
import numpy as np
import time


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
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    abs_laplacian=cv2.convertScaleAbs(laplacian)
    ret,binary = cv2.threshold(resize_gray,127,255,cv2.THRESH_BINARY)
    # cv2.imshow('frame',binary)
    # cv2.waitKey(1)
    return sum([2 ** i for (i, v) in enumerate(binary.flatten()) if v])
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
         b1= x>>i&1
         b2 = y>>i&1
         ans+= not(b1==b2)
      return ans


#capture video
cap = cv2.VideoCapture('highway4.avi')
cv2.namedWindow(winname='Hashing')
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
            cv2.rectangle(frame, pt1, pt2, (0,255,0), 2)     
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
            w_width=48
            w_height=48
            distance=[]
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
                result=hammingDistance(lhash(roi_firstframe),lhash(roi_2))
                refPt_nextframe_stack.append(((x,y),(x + w_width, y + w_height)))
                distance.append(result)          
                time.sleep(0.025)
                
            key2 = cv2.waitKey(1) or 0xff
            #press "P" to continue the video
            if key2 == ord('p'):
                # print(refPt_nextframe)
                # print(distance)
                print(min(distance))
                print(distance.index(min(distance)))
                t=list(refPt_nextframe_stack[distance.index(min(distance))])
                print(t)
                print("area:",area_detected)
                #show the re-identitication image
                cv2.imshow("Re-Identification", clone)
                cv2.waitKey(0)
                break

    cv2.imshow('Hashing', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()