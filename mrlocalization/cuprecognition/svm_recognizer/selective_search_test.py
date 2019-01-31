#!/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''
import os
import sys
import cv2

import numpy as np

if __name__ == '__main__':

    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    import pcl
    relative_utils_path = '../../utils'
    utils_path = os.path.join(os.path.dirname(__file__), relative_utils_path)
    sys.path.append(utils_path)
    from pcl_helper import float_to_rgb

    relative_pointcloud_path = '../../../data/pointclouds5/table9_1516127870160776.pcd'
    pointcloud_path = os.path.join(os.path.dirname(__file__), relative_pointcloud_path)
    pointcloud = pcl.load_XYZRGB(pointcloud_path)
    pointcloud_np = pointcloud.to_array()

    # Compute RGB image
    rgb_image = []
    for i in range(0, pointcloud_np.shape[0]):
        rgb_image.append(float_to_rgb(pointcloud_np[i, 3]))
    rgb_image_np = np.asarray(rgb_image, dtype=np.uint8)
    rgb_image_np = np.reshape(rgb_image_np, (480, 640, 3))

    # read image
    im = rgb_image_np
    im = np.flip(im, axis = 2)


    # resize image
    newHeight = 200
    newWidth = int(im.shape[1]*200/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    fast = True
    if (fast):
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 50

    while True:
        # create a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Output", imOut)

        # record key press
        k = cv2.waitKey(0) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
