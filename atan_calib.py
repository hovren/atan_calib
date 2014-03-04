# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:38:36 2014

@author: hannes
"""

import os
import glob
import argparse
import sys
import operator

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import h5py

sys.path.append("/home/hannes/Code/rspy/build/lib.linux-x86_64-2.7/")
import rspy

def lensdist_inv(X,wc,lgamma):
    """
    Apply inverse atan-model lens distorsion

    Parameters
    ------------------------
    X : array Coordinate list 2xN
    wc: array Distorsion center (2x1 or 2x0)
    lgamma: float Lens distorsion parameter 
    """
    if X.ndim == 1:
        X = X.reshape((X.size, 1))
    # Switch to polar coordinates
    rn=np.sqrt((X[0,:]-wc[0])**2 + (X[1,:]-wc[1])**2)
    phi=np.arctan2(X[1,:]-wc[1],X[0,:]-wc[0])

    # 'atan' method
    r=np.tan(rn*lgamma)/lgamma;

    # Switch back to rectangular coordinates
    Y = np.ones(X.shape)
    Y[0,:]=wc[0]+r*np.cos(phi)
    Y[1,:]=wc[1]+r*np.sin(phi)
    return Y

def lensdist(X,wc,lgamma):
    """
    Apply atan-model lens distorsion

    Parameters
    ------------------------
    X : array Coordinate list 2xN
    wc: array Distorsion center (2x1 or 2x0)
    lgamma: float Lens distorsion parameter 
    """
    if X.ndim == 1:
        X = X.reshape((X.size, 1))
    # Switch to polar coordinates
    rn=np.sqrt((X[0,:]-wc[0])**2 + (X[1,:]-wc[1])**2)
    phi=np.arctan2(X[1,:]-wc[1],X[0,:]-wc[0])

    r=np.arctan(rn*lgamma)/lgamma

    # Switch back to rectangular coordinates
    Y = np.ones(X.shape)
    Y[0,:]=wc[0]+r*np.cos(phi)
    Y[1,:]=wc[1]+r*np.sin(phi)
    
    return Y

def build_corners(image_dir, chessboard_size):
    image_files = glob.glob(os.path.join(image_dir, '*.png'))    
    f = h5py.File(os.path.join(image_dir, 'corners.hdf'), 'w')
    grp = f.create_group("images") 
    
    for fname in image_files:
        im = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if im is None:
            print "Skipping {0}".format(fname)
            continue
        
        has_board, corners = cv2.findChessboardCorners(im, chessboard_size)
        if has_board:
            term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
            cv2.cornerSubPix(im, corners, (5, 5), (-1, -1), term)
            fname_base = os.path.basename(fname)
            imgrp = grp.create_group(fname_base)
            imgrp.create_dataset("image", data=im)
            imgrp.create_dataset("corners", data=corners)
    
    f.close()

def save_directory_hdf(corners, images, args):
    outfile = os.path.join(args.input, 'corners.hdf')
    f = h5py.File(outfile, 'w')
    grp = f.create_group("images")
    zpad = int(np.ceil(np.log10(len(images)))) + 1
    for i, (c, im) in enumerate(zip(corners, images)):
        imgrp = grp.create_group("{0:{zpad}d}".format(i, zpad=zpad))
        imgrp.create_dataset("image", data=im)
        imgrp.create_dataset("corners", data=c)
    f.close()
    
def optfunc_atan_lines(x, lines):
        wc = x[:2]        
        lgamma = x[2]
        
        residual = []
        for l in lines:
            lu = lensdist_inv(l.T, wc, lgamma).T
            #print luatan-reproj-K
            a = lu[0] # Line endpoint
            n = lu[-1]-lu[0] # Direction vector
            n /= np.sqrt(np.inner(n, n))
            for x in lu:
                d = (a - x) - np.dot((a - x), n)*n
                d = np.sum(d**2)
                residual.append(d)
        return residual


def lines_from_corners(corners, chessboard_size):
    cw, ch = chessboard_size
    vlines = [corners[cw*i:cw*(i+1)].reshape(-1,2) for i in range(ch)]
    hlines = [corners[i::cw].reshape(-1,2) for i in range(cw)]
    
    return vlines, hlines
    
def load_corners(input_file):
    f = h5py.File(input_file, 'r')
    corner_list = []
    images = []
    grp = f["images"]
    for key in sorted(grp.keys()):
        imgrp = grp[key]
        corner_list.append(imgrp["corners"].value)
        images.append(imgrp["image"].value)
    f.close()
    return corner_list, images

def chessboard_to_world(chessboard_size):
    pattern_points = np.zeros( (np.prod(chessboard_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(chessboard_size).T.reshape(-1, 2)
    return pattern_points

def calibrate_opencv(corners, images, args):
    h, w = images[0].shape[:2]
    
    object_points = [chessboard_to_world(args.chessboard),]*len(corners)
    flags = cv2.cv.CV_CALIB_FIX_K1 | cv2.cv.CV_CALIB_FIX_K2 | cv2.cv.CV_CALIB_FIX_K3 | \
            cv2.cv.CV_CALIB_ZERO_TANGENT_DIST
    print "Running OpenCV calibrator on {:d} image points".format(len(corners))
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(object_points, corners, (w, h), flags=flags)
    print "RMS", rms
    return camera_matrix, rvecs, tvecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir")
    parser.add_argument("--chessboard", type=int, nargs=2, default=(6,8))
    parser.add_argument("--plot", action="store_true")     
    parser.add_argument("--mode", choices=("atan-lines", "atan-reproj-K"), default="atan-lines")
    parser.add_argument("--recalc-opencv", action="store_true")
    args = parser.parse_args()

    
    corners_fname = os.path.join(args.inputdir, "corners.hdf")

    if not os.path.exists(corners_fname):
        print "Creating {}".format(corners_fname)
        build_corners(args.inputdir, args.chessboard)

    if args.mode == 'atan-reproj-K':
        if "opencv_calibration" in hdf_file.keys() and not args.recalc_opencv:
            print "Loading old calibration values"
            opencvgrp = hdf_file["opencv_calibration"]
            K = opencvgrp["camera_matrix"].value
            imlistgrp = opencvgrp["images"]
            R_list = []
            t_list = []
            for key in sorted(imlistgrp.keys()):
                imgrp = imlistgrp[key]
                R = imgrp["R"].value
                t = imgrp["t"].value
                R_list.append(R)
                t_list.append(t)
        else:
            if "opencv_calibration" in hdf_file.keys():
                del hdf_file["opencv_calibration"]
                hdf_file.flush()
            K, R_list, t_list = calibrate_opencv(corners, images, args)
            opencvgrp = hdf_file.create_group("opencv_calibration")
            opencvgrp.create_dataset("camera_matrix", data=K)            
            immetagrp = opencvgrp.create_group("images")
            zpad = int(np.ceil(np.log10(len(R_list)))) + 1
            for i, (R, t) in enumerate(zip(R_list, t_list)):
                imgrpstr = "{0:0{zpad}d}".format(i, zpad=zpad)
                imgrp = immetagrp.create_group(imgrpstr)
                imgrp.create_dataset("R", data=R)
                imgrp.create_dataset("t", data=t)
                #imgrp["image"] = hdf_file["/images/{}".format(imgrpstr)] #Link to data
            hdf_file.flush()
        
        # Now we have a guess of the intrinsic camera matrix
        # and the camera rotation and translation vectors
        print "Camera matrix according to OpenCV"
        print K
        
        

    elif args.mode == "atan-lines":
        corners, images = load_corners(corners_fname)
        x0 = np.array([1920/2, 1080/2, 0.0001])
        lines = []
        for vl, hl in [lines_from_corners(c, args.chessboard) for c in corners]:
            lines.extend(vl)
            lines.extend(hl)
        
        print "Running atan-lines optimizer on {:d} line segments".format(len(lines))
        x, covx, infodict, mesg, ier = scipy.optimize.leastsq(optfunc_atan_lines, x0, (lines, ), diag=(1000,1000,0.001), full_output=True, xtol=1e-10)
        print "X=",x
        print mesg
        print infodict['nfev'], "evaluations"
        
        wc = x[:2]
        lgamma = x[2]
        
        xmap, ymap = np.meshgrid(range(1920), range(1080))
        m = np.dstack((xmap, ymap))
        p = m.T.reshape(2,-1)
        print p[:, :5]
        pu = lensdist_inv(p, wc, lgamma)
        print pu[:,:5]
        print pu.min(), pu.max(), pu.mean()
        
        mu = pu.reshape(m.shape[::-1]).T
        xmapu = mu[:,:,0]
        ymapu= mu[:,:,1]
        
        im = images[np.random.randint(0,len(images)-1)]
        rectim = rspy.forwardinterp.forwardinterp(im.astype('float64'), xmapu, ymapu)
        plt.subplot(2,1,1)
        plt.imshow(im)
        plt.subplot(2,1,2)
        plt.imshow(rectim)
        plt.show()
    
    
    