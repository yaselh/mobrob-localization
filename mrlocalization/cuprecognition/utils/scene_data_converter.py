import numpy as np

def bgr2rgb(img):
    b_channel = img[:,:,0]
    g_channel = img[:,:,1]
    r_channel = img[:,:,2]
    img[:,:,0] = r_channel
    img[:,:,1] = g_channel
    img[:,:,2] = b_channel
    return img

def float2int(img):
    img = img.astype(np.uint8)
    return img
