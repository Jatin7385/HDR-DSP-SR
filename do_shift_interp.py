''' This function is for bicubic or any interpolation. '''

import cv2
import matplotlib.pyplot as plt
def do_shift_interp(inp, shift, arg, L):
    '''

    %
    % out = do_shift_interp( in, shift, arg, L );
    %
    % Performs a shift on image image (in) using the
    % interpolation method specified in (arg).
    %
    % REQUIRED INPUTS
    % in        Input image or images
    % shift     [Horizontal shift, Vertical shift in input pixel spacings]
    % arg       Interpolation method 'nea','bil','bic','spline'
    % L         Upsampling factor for interpolation
    %
    % OUTPUTS
    % out       Output warped image(s)
    %
    % Author: Dr. Russell Hardie
    % 9/6/2009
    %
    % COPYRIGHT © 2009 RUSSELL C. HARDIE.  ALL RIGHTS RESERVED.

    '''

    resized_img = cv2.resize(inp, dsize=(inp.shape[1]*L, inp.shape[0]*L), interpolation=cv2.INTER_CUBIC)

    # plt.imshow(inp, cmap="gray")
    # plt.title("Interpolated Image")
    # plt.show()

    return resized_img
