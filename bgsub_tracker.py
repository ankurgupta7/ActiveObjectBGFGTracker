from __future__ import print_function
import numpy as np
import cv2
import math

#global fg-bg subtractor based on Mixture of 2 Gaussians for modelling fg-bg clusters
fgbg = cv2.createBackgroundSubtractorMOG2(history=50)

def init_xywh_kalman():
    #creating kalman with states = [x,y,w,h,vx,vy,vw,vh]
    #measurement = [x,y,w,h]
    kalman = cv2.KalmanFilter(8, 4)  

    # velocity of x&y is twice that w&h to model larger deviations prior over translation than scale
    kalman.transitionMatrix = np.array([
        [1.,0.,0.,0., .01,0.,0.,0.],
        [0.,1.,0.,0., 0.,.01,0.,0.],
        [0.,0.,1.,0., 0.,0.,.005,0.],
        [0.,0.,0.,1., 0.,0.,0.,.005],
        # 
        [0.,0.,0.,0., 1.,0.,0.,0.],
        [0.,0.,0.,0., 0.,1.,0.,0.],
        [0.,0.,0.,0., 0.,0.,1.,0.],
        [0.,0.,0.,0., 0.,0.,0.,1.],
        ],dtype=np.float32)

    kalman.measurementMatrix = np.array([
        [1.,0.,0.,0., 0.,0.,0.,0.],
        [0.,1.,0.,0., 0.,0.,0.,0.],
        [0.,0.,1.,0., 0.,0.,0.,0.],
        [0.,0.,0.,1., 0.,0.,0.,0.],
        ], dtype=np.float32)

    #noise in process is less than measurements
    kalman.processNoiseCov = 2e-4 * np.eye(8,dtype=np.float32)
    kalman.measurementNoiseCov = 1e-4 * np.ones((4,4),dtype=np.float32)

    #noise in measurement of w &h is considerably higher in my measurements
    kalman.measurementNoiseCov[2,2] = 3 
    kalman.measurementNoiseCov[3,3] = 3 

    kalman.errorCovPost = 1. * np.ones((8,8),dtype=np.float32)
    kalman.statePost = 0.1 * np.random.randn(8,1).astype('float32')
    kalman.statePre = np.array([0.,0.,0.,0.,0.,0.,0.,0.],dtype=np.float32).transpose()
    
    return kalman

def bound_subt_pts(pts):
    pts = np.array(pts)
    
    #median would filter noise in considering the representing pt of diff image
    centroid = np.median(pts, axis = 0)

    #getting inliers based on L2 norm of pts from temporaty centroid above
    dists = np.linalg.norm(pts-centroid, axis = 1)
    median_dist =  np.mean(dists)
    inliers_idx = np.where(dists < median_dist*3)
    inliers = pts[inliers_idx]

    #statistics of inliers only
    y,x = np.mean(inliers, axis = 0)
    h,w = np.max(inliers,axis=0)- np.min(inliers,axis=0)
    
    return (x,y,w,h)

def get_foreground_mask(frame):
    
    fgmask = fgbg.apply(frame)
    
    #removing eroding and dilating to remove noise (not necessarily outliers)
    kernel = np.ones((5,5), dtype='uint8')
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    #indices of pts that still remain bright
    z = (np.where(fgmask == 255))
    subt_pts = (zip(z[0],z[1]))
    

    return subt_pts

def main():
    cap = cv2.VideoCapture(0)

    kalman = init_xywh_kalman()
    x_pred = 0.
    y_pred = 0.
    w_pred = 0.
    h_pred = 0.

    x_new = 0.
    y_new = 0.
    w_new = 0.
    h_new = 0.

    while(1):
        #read frame
        ret, frame = cap.read()

        subt_pts = get_foreground_mask(frame)
        
        #emperical lower limit for getting a good measurement
        if(len(subt_pts)>30):
            x_new, y_new, w_new, h_new = bound_subt_pts(subt_pts)
            kalman.correct(np.array([x_new, y_new, w_new, h_new],dtype=np.float32))

        prediction = kalman.predict()
        x_pred = prediction[0]
        y_pred = prediction[1]
        w_pred = prediction[2]
        h_pred = prediction[3]
        # print(prediction)

        cv2.rectangle(frame,
                (int(math.ceil(x_pred - w_pred/2)) , int(math.ceil(y_pred - h_pred/2))),
                (int(math.floor(x_pred + w_pred/2)) , int(math.floor(y_pred + h_pred/2))),
                (0,0,255), 3)
        
        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()