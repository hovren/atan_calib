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
        im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print("Skipping {0}".format(fname))
            continue
        
        has_board, corners = cv2.findChessboardCorners(im, tuple(chessboard_size))
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
    image_keys = []
    grp = f["images"]
    for key in sorted(grp.keys()):
        imgrp = grp[key]
        corner_list.append(imgrp["corners"].value)
        images.append(imgrp["image"].value)
        image_keys.append(key)
    f.close()
    return corner_list, images, image_keys

def chessboard_to_world(chessboard_size):
    pattern_points = np.zeros( (np.prod(chessboard_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(chessboard_size).T.reshape(-1, 2)
    return pattern_points

def calibrate_opencv(corners, images, chessboard_size):
    h, w = images[0].shape[:2]
    
    object_points = [chessboard_to_world(chessboard_size),]*len(corners)
    flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | \
            cv2.CALIB_ZERO_TANGENT_DIST
    print("Running OpenCV calibrator on {:d} image points".format(len(corners)))
    K0 = np.eye(3)
    d0 = np.zeros(8)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(object_points, corners, (w, h), K0, d0, flags=flags)
    print("RMS", rms)
    return camera_matrix, rvecs, tvecs

def build_opencv_calibration(image_dir, chessboard_size):
    corners_fname = os.path.join(image_dir, 'corners.hdf')
    print("Loading from", corners_fname)
    corners, images, image_keys = load_corners(corners_fname)
    K, rvecs, tvecs = calibrate_opencv(corners, images, chessboard_size)
    
    ocvcal_fname = os.path.join(image_dir, 'ocvcal.hdf')
    f = h5py.File(ocvcal_fname, 'w')
    ocvgrp = f.create_group("opencv_calibration")
    ocvgrp.create_dataset("K", data=K)
    imlistgrp = ocvgrp.create_group("images")
    for r, t, key in zip(rvecs, tvecs, image_keys):
        imgrp = imlistgrp.create_group(key)
        imgrp.create_dataset("R", data=r)
        imgrp.create_dataset("t", data=t)
        imgrp["corners"] = h5py.ExternalLink("corners.hdf", "images/{}/corners".format(key))
        imgrp["image"] = h5py.ExternalLink("corners.hdf", "images/{}/image".format(key))
    f.close()
    

def load_ocvcal(ocvcal_fname):
    images = []
    rvecs = []
    tvecs = []
    corners = []
    image_keys = []
    
    f = h5py.File(ocvcal_fname, 'r')
    ocvgrp = f["opencv_calibration"]
    K = ocvgrp["K"].value
    imlistgrp = ocvgrp["images"]
    for key in sorted(imlistgrp.keys()):
        imgrp = imlistgrp[key]
        corners.append(imgrp["corners"].value)
        images.append(imgrp["image"].value)
        rvecs.append(imgrp["R"].value)
        tvecs.append(imgrp["t"].value)
        image_keys.append(key)
    f.close()
    return K, rvecs, tvecs, corners, images, image_keys


def optfunc_reproj(x, corner_list, object_points):
    fx, fy, cx, cy, wx, wy, lgamma = x[:7]
    wc = np.array([wx, wy])
    K = np.array([[ fx, 0,  cx],
                  [ 0,  fy, cy],
                  [ 0,  0,  1]])
    residuals = []
    for i, corners in enumerate(corner_list):
        offset = 7+6*i
        r = x[offset:offset+3]
        t = x[offset+3:offset+6].reshape(3,1)
        R = cv2.Rodrigues(r)[0]
        u = R.dot(object_points.T) + t
        u /= np.tile(u[2], (3,1))
        xnd = lensdist(u, wc, float(lgamma))        
        xhat = K.dot(xnd)
        xhat /= np.tile(xhat[2], (3,1))
        d = corners.T.reshape(2,-1) - xhat[:2]
        residuals.extend(d.flatten())
    return residuals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir")
    parser.add_argument("--chessboard", type=int, nargs=2, default=(6,8))
    parser.add_argument("--plot", action="store_true")     
    parser.add_argument("--mode", choices=("atan-lines", "atan-reproj-K"), default="atan-reproj-K")
    parser.add_argument("--recalc-opencv", action="store_true")
    parser.add_argument("--view", type=int, default=None, nargs='+')
    parser.add_argument("--view-mode", choices=("opt-result","rectify"), default="opt-result")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    print(args)

    # Only viewing?
    if args.view is not None:
        if args.mode == 'atan-reproj-K':
            f = h5py.File(os.path.join(args.inputdir, 'atanprojres.hdf'))
            imlistgrp = f["images"]
            
            num_images = len(imlistgrp.keys())
            if len(args.view)==1 and args.view[0] == -1:
                indices = range(num_images)
            elif np.any(np.array(args.view) >= num_images) or np.any(np.array(args.view) < 0):
                raise Exception("Bad view parameter")
            else:
                indices = args.view
            
            x_sol = f["x"].value
            plt.hist(f["residuals"])
            plt.show()
            fx, fy, cx, cy, wx, wy, lgamma = x_sol[:7]
            wc = np.array([wx, wy])
            K = np.array([[ fx, 0,  cx],
                          [ 0,  fy, cy],            
                          [ 0,  0,  1]])
            
            object_points = chessboard_to_world(args.chessboard)
            plt.figure()
            for idx in indices:
                fname = sorted(imlistgrp.keys())[idx]
                print("Looking at", fname)
                imgrp = imlistgrp[fname]
                R = imgrp["R"].value
                t = imgrp["t"].value.reshape(3,1)
                im = imgrp["image"].value
                corners = imgrp["corners"].value.T.reshape(2,-1)
                
                if args.view_mode == "opt-result":
                    plt.clf()
                    gs = plt.GridSpec(5,5)
                    imax = plt.subplot(gs[:4,:])
                    resax = plt.subplot(gs[4,:])
                    imax.clear()
                    resax.clear()
                    
                    plt.gray()
                    imax.imshow(im)
                    plt.title(fname)
                    x = corners[0]
                    y = corners[1]
                    imax.scatter(x,y, color='g')
                    
                    u = R.dot(object_points.T) + t
                    u /= np.tile(u[2], (3,1))
                    xnd = lensdist(u, wc, float(lgamma))        
                    xhat = K.dot(xnd)
                    xhat /= np.tile(xhat[2], (3,1))
                    
                    imax.scatter(xhat[0], xhat[1], color='r', marker='x')
                    
                    d = corners - xhat[:2]
                    resax.hist(d.flatten())
                else:
                    h, w = im.shape[:2]
                    xmap, ymap = np.meshgrid(range(w), range(h))
                    m = np.dstack((xmap, ymap))
                    ptmp = m.T.reshape(2,-1)
                    p = np.ones((3, ptmp.shape[1]))
                    p[:2] = ptmp
                    
                    
                    P = np.linalg.inv(K).dot(p)
                    P /= np.tile(P[2], (3,1))
                    Pu = lensdist_inv(P, wc, lgamma)
                    pu = K.dot(Pu)
                    pu /= np.tile(pu[2], (3,1))
                    pu2d = pu[:2]

                    mu = pu2d.reshape(m.shape[::-1]).T
                    xmapu = mu[:,:,0]
                    ymapu= mu[:,:,1]
                    rectim = rspy.forwardinterp.forwardinterp(im.astype('float64'), xmapu, ymapu)
                    
                    # Aaand back
                    # Nice to have to remember order of things for projections
                    if False:
                        pu3d = np.linalg.inv(K).dot(pu)
                        pu3d_2 = lensdist(pu3d, wc, lgamma)
                        pu3d_2d = K.dot(pu3d_2)
                        pu3d_2d /= np.tile(pu3d_2d[2], (3,1))
                        
                        print(pu3d_2d)
                        print(p)
                        dtmp = np.sum((pu3d_2d - p)**2, axis=0)
                        print(dtmp)
                        print(dtmp.min(), dtmp.max())
                    
                    plt.subplot(2,1,1)
                    plt.imshow(im)
                    plt.subplot(2,1,2)
                    plt.imshow(rectim)                    
                    
                plt.tight_layout()
                plt.draw()
                plt.waitforbuttonpress()
            
        sys.exit(0)
    
    # Build corners from input directory
    corners_fname = os.path.join(args.inputdir, "corners.hdf")
    if not os.path.exists(corners_fname):
        print("Creating {}".format(corners_fname))
        build_corners(args.inputdir, args.chessboard)

    if args.mode == 'atan-reproj-K':
        # Is there an existing opencv calibration?
        ocvcal_fname = os.path.join(args.inputdir, 'ocvcal.hdf')
        if not os.path.exists(ocvcal_fname) or args.recalc_opencv:
            build_opencv_calibration(args.inputdir, args.chessboard)

        K, rvecs, tvecs, corner_list, images, image_keys = load_ocvcal(ocvcal_fname)
        
        if args.max is not None:
            indices = range(len(image_keys))
            np.random.shuffle(indices)
            indices = indices[:args.max]
            rvecs = [rvecs[i] for i in indices]
            tvecs = [tvecs[i] for i in indices]
            corner_list = [corner_list[i] for i in indices]
            images = [images[i] for i in indices]
            image_keys = [image_keys[i] for i in indices]
                
        
        # Now we have a guess of the intrinsic camera matrix
        # and the camera rotation and translation vectors
        print("Camera matrix according to OpenCV")
        print(K)
        

        # Sanity check, just to look at the reprojection errors
        if False:
            object_points = chessboard_to_world(args.chessboard)        
            for i, (r, t, im, corn) in enumerate(zip(rvecs, tvecs, images, corner_list)):
                print(i)
                R = cv2.Rodrigues(r)[0]
                x = K.dot(R.dot(object_points.T) + t)
                x /= np.tile(x[2], (3,1))
                print(x[:, 5:])
                print(corn.T.reshape(2,-1)[:,5:])
                print("---")
                if i == 1:
                    break
        
        
        h, w = images[0].shape[:2]

        # OpenCV projection is x ~ K(RX + t)
        # Let the optimization parameters, in order, be
        # fx, fy, cx, cy of camera matrix K (5 parameters)
        # wx, wy, lgamma of atan model (3 parameters)
        # (rvecs_k, tvecs_k) ((3+3)*N parameters, where N is number of images)
        # Total: 8 + 6N parameters
        num_params = 4 + 3 + 6 * len(images)
        x0 = np.zeros((num_params,))
        x0[0] = K[0,0] # fx
        x0[1] = K[1,1] # fy
        x0[2] = K[0,2] # cx
        x0[3] = K[1,2] # cy
        x0[4] = 0 #w/2 # wx
        x0[5] = 0 #h/2 # wy
        x0[6] = 0.1 # lgamma
        for i, (r, t) in enumerate(zip(rvecs, tvecs)):
            offset = 7+6*i
            x0[offset:offset+3] = r.flatten()
            x0[offset+3:offset+6] = t.flatten()
        
        #print x0
        object_points = chessboard_to_world(args.chessboard)
        
        residuals = optfunc_reproj(x0, corner_list, object_points)
        #print residuals
        #print np.mean(residuals), np.std(residuals)        
        
        print("Running atan-reproj-K optimizer on {:d} points".format(len(corner_list)*np.prod(args.chessboard)))
        #x, covx, infodict, mesg, ier = scipy.optimize.leastsq(optfunc_reproj_late, x0, (corner_list, object_points), full_output=True)
        scaling = np.ones_like(x0)
        scaling[:4] = 100.0
        scaling[5:6] = 1000.0
        scaling[7] = 0.001
        x, covx, infodict, mesg, ier = scipy.optimize.leastsq(optfunc_reproj, x0, (corner_list, object_points), diag=scaling, full_output=True)
        print("X[:7] = ",x[:7])
        print(mesg)
        print(infodict['nfev'], "evaluations")

        # Store results        
        atanproj_fname = os.path.join(args.inputdir, 'atanprojres.hdf')
        f = h5py.File(atanproj_fname, 'w')
        f["residuals"] = infodict['fvec']
        f["nfev"] = infodict['nfev']
        f["x"] = x
        f["mesg"] = mesg
        f["ier"] = ier
        imlistgrp = f.create_group("images")
        for i, key in enumerate(image_keys):
            imgrp = imlistgrp.create_group(key)
            offset = 7+6*i
            r = x[offset:offset+3]
            t = x[offset+3:offset+6]
            imgrp["image"] = h5py.ExternalLink("corners.hdf", "images/{}/image".format(key))
            imgrp["corners"] = h5py.ExternalLink("corners.hdf", "images/{}/corners".format(key))
            imgrp["R"] = cv2.Rodrigues(r)[0]
            imgrp["t"] = t
            imgrp["local_index"] = i
        f.close()
        
        # Store short results
        f2 = h5py.File(os.path.join(args.inputdir, 'atan_calibration.hdf'), 'w')
        fx, fy, cx, cy, wx, wy, lgamma = x[:7]
        wc = np.array([wx, wy])
        K = np.array([[ fx, 0,  cx],
                      [ 0,  fy, cy],
                      [ 0,  0,  1]])
        f2["K"] = K
        f2["wc"] = wc
        f2["lgamma"] = lgamma
        f2["opt_residual_mean"] = np.mean(infodict['fvec'])
        f2["opt_residual_std"] = np.std(infodict['fvec'])
        f2.close()
            

    elif args.mode == "atan-lines":
        corners, images, image_keys = load_corners(corners_fname)
        x0 = np.array([1920/2, 1080/2, 0.0001])
        lines = []
        for vl, hl in [lines_from_corners(c, args.chessboard) for c in corners]:
            lines.extend(vl)
            lines.extend(hl)
    
        print("Running atan-lines optimizer on {:d} line segments".format(len(lines)))
        x, covx, infodict, mesg, ier = scipy.optimize.leastsq(optfunc_atan_lines, x0, (lines, ), diag=(1000,1000,0.001), full_output=True, xtol=1e-10)
        print("X=",x)
        print(mesg)
        print(infodict['nfev'], "evaluations")
        
        wc = x[:2]
        lgamma = x[2]
        
        xmap, ymap = np.meshgrid(range(1920), range(1080))
        m = np.dstack((xmap, ymap))
        p = m.T.reshape(2,-1)
        print(p[:, :5])
        pu = lensdist_inv(p, wc, lgamma)
        print(pu[:,:5])
        print(pu.min(), pu.max(), pu.mean())
        
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
    
    
    
