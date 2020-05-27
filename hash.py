import cv2
import numpy as np
import time


#mouse callback function#
def draw_rectangle(event, x, y, flags, param):

    global pt1, pt2, topLeft_clicked, bottomRight_clicked

    #mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        #reset
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
#laplace hash algorithm
def lhash(image):
    """
    Laplace Based Hash
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resize_gray = cv2.resize(gray,(8,8))
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    abs_laplacian=cv2.convertScaleAbs(laplacian)
    ret,binary = cv2.threshold(resize_gray,127,255,cv2.THRESH_BINARY)
    # cv2.imshow('frame',binary)
    # cv2.waitKey(1)
    return sum([2 ** i for (i, v) in enumerate(binary.flatten()) if v])

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

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

firstFrame = True
while True: 
    #initially we haven't drawn anything
    pt1 = (0,0)
    pt2 = (0,0)
    topLeft_clicked = False
    bottomRight_clicked = False
    ret, frame = cap.read()
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

        if cv2.waitKey(1) &0xFF == ord('c'):
            firstFrame = False
            
            #cv2.imwrite('first_frame.jpg', roi_firstframe)
            #img = cv2.imread('first_frame.jpg')
            #print(type(lhash(roi_firstframe)))
            #print(lhash(roi_firstframe))
            break

    key=cv2.waitKey(100) &0xFF

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



            w_width=48
            w_height=48
            distance=[]
            refPt_nextframe_stack=[]
            
            
            
            for (x, y, window) in sliding_window(roi_nextframe, stepSize=8, windowSize=(w_width, w_height)):
                clone = roi_nextframe.copy()
                if window.shape[0] != w_height or window.shape[1] != w_width:
                    continue
                
                cv2.rectangle(clone, (x, y), (x + w_width, y + w_height), (0, 255, 0), 1)
                
                
                refPt_nextframe=[]
                refPt_nextframe.append((x,y))
                refPt_nextframe.append((x + w_width, y + w_height))
                # if len(refPt_nextframe) == 2:
                roi_2=clone[refPt_nextframe[0][1]:refPt_nextframe[1][1], refPt_nextframe[0][0]:refPt_nextframe[1][0]]
                
                result=hammingDistance(lhash(roi_firstframe),lhash(roi_2))

                refPt_nextframe_stack.append(((x,y),(x + w_width, y + w_height)))
                distance.append(result)

                
                time.sleep(0.025)
                
            key2 = cv2.waitKey(1) or 0xff
            if key2 == ord('p'):
                # print(refPt_nextframe)
                # print(distance)
                print(min(distance))
                print(distance.index(min(distance)))
                t=list(refPt_nextframe_stack[distance.index(min(distance))])
                print(t)

                # print(refPt_nextframe_stack)
 
                print("area:",area_detected)
                cv2.imshow("Re-Identification", clone)
                cv2.waitKey(0)

                break

    cv2.imshow('Hashing', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()