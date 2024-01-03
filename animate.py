'''
Animation class. Create video with effects on a still image calculated by
frequency domain changes.

Overall TODOs:
    - screen pulse
    - image channel support
    - stereo sound support
    - video input support
    - working argument parser
    - comfy UI for editing    
    OR
    - rewrite this all as sony vegas plugin
'''


import argparse
import cv2
import scipy
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

MAX_ENERGY = 5000000
FPS = 60
HIST_LEN = 3

OUTPUT_NAME='output.mp4'

def brighten(image, factor):
    # factor range -0.5, 0.5
    new_image = np.copy(image)
    new_image = new_image * factor
    new_image = new_image / new_image.max()

    return new_image

def adjust_contrast(image, factor, mid):
    new_image = (image - mid) * factor + mid
    new_image[new_image<0] = 0
    new_image = new_image / new_image.max()
    return new_image

def color_effect(image, carrousel, factor):
    hsv_im = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_im[:,:,0] = hsv_im[:,:,0] + carrousel + factor*2
    return cv2.cvtColor(hsv_im, cv2.COLOR_HSV2RGB)

def saturation_effect(image, factor):
    hsv_im = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_im[:,:,1] = hsv_im[:,:,1] + (factor*4)
    return cv2.cvtColor(hsv_im, cv2.COLOR_HSV2RGB)

def blur(image, kernel_size):
    new_image = np.copy(image)
    blurred = gaussian_filter(new_image, sigma=kernel_size)
    return blurred

def apply_kernel(image, kernel):
    new_image = np.copy(image)
    for _ in range(2):
        new_image = np.concatenate([new_image, np.zeros((1, image.shape[1], image.shape[2]))])
    for _ in range(2):
        new_image = np.concatenate([new_image, np.zeros((new_image.shape[0], 1, new_image.shape[2]))], 1)
    for i in range(3):
        new_image[:,:,i] = convolve2d(image[:,:,i], kernel)
    
    return new_image[1:new_image.shape[0]-1, 1:new_image.shape[1]-1]

def combine_images(image1, image2):
    new_image = np.add(image1, image2)
    return new_image

def multiply_image(image, mask):
    mask = mask / mask.max()
    return np.multiply(image, mask)

def split_image(image, mask):
    masked = np.copy(image)
    for i in range(image.shape[2]):
        masked[:,:,i] = multiply_image(image[:,:,i], mask)
    leftovers = image - masked
    return (masked, leftovers)

def get_low_intensity(energy):
    if energy < 500000:
        return 0
    elif energy < 1000000:
        return 2
    elif energy < 1500000:
        return 3
    else:
        return 3
    
def get_midlow_intensity(energy):
    if energy < 100000:
        return 0
    elif energy < 200000:
        return 2
    elif energy < 300000:
        return 3
    else:
        return 3
    
def get_midmid_intensity(energy):
    if energy < 50000:
        return 0
    elif energy < 100000:
        return 2
    elif energy < 150000:
        return 3
    else:
        return 3
    
def get_midhigh_intensity(energy):
    if energy < 50000:
        return 0
    elif energy < 70000:
        return 2
    elif energy < 90000:
        return 3
    else:
        return 3
    
def get_high_intensity(energy):
    if energy < 1500:
        return 0
    elif energy < 3000:
        return 2
    elif energy < 4000:
        return 3
    else:
        return 3

def animate(spec, im, step_size, seed):
    i = 0
    
    spinner = True
    carrousel = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(OUTPUT_NAME, fourcc, FPS, im.shape[:2])

    with tqdm(total=spec.shape[1]) as pbar:
        while i < spec.shape[1]:
            i = min(spec.shape[1], i + step_size)
            pbar.update(step_size)
            window = spec[:, floor(i - step_size):floor(i)]
    
            spinner = not spinner
            
            carrousel = carrousel + 1
            if carrousel > 255:
                carrousel = 0
    
            low_intensity = get_low_intensity( window[0:1,:].mean() )
            midlow_intensity= get_midlow_intensity( window[1:3].mean() )
            midmid_intensity= get_midmid_intensity( window[3:6].mean() ) 
            midhigh_intensity= get_midhigh_intensity( window[6:10].mean() )
            high_intensity= get_high_intensity( window[10:,:].mean() )
    
            hsv_im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV )
            sobel_x = apply_kernel(hsv_im, np.array([[1, 2, 1], 
                                                  [0, 0, 0],
                                                  [-1, -2, -1]]))
            sobel_y = apply_kernel(hsv_im, np.array([[1, 0, -1],
                                                  [2, 0, -2],
                                                  [1, 0, -1]]))
            sobel = combine_images(sobel_x, sobel_y)
            hue_sobel = sobel[:,:,0]
            hue_sobel[hue_sobel<0] = 0
            hue_sobel = hue_sobel / hue_sobel.mean()
            hue_sobel = blur(hue_sobel, midmid_intensity)
            sat_sobel = sobel[:,:,1]
            sat_sobel[sat_sobel<0] = 0
            sat_sobel = sat_sobel / sat_sobel.mean()
            sat_sobel = blur(sat_sobel, midhigh_intensity)
            val_sobel = sobel[:,:,2]
            val_sobel[val_sobel<0] = 0
            val_sobel = val_sobel / val_sobel.mean()
            val_sobel = blur(val_sobel, high_intensity)
    
            sat_effects, rest = split_image(im, sat_sobel)
            val_effects, rest = split_image(rest, val_sobel)
            hue_effects, rest = split_image(rest, hue_sobel)
    
            colored = color_effect(hue_effects, carrousel, midlow_intensity)
            satured = saturation_effect(colored, midlow_intensity)
            hue_mask = satured / 255
    
            brightened = color_effect(sat_effects, carrousel, midlow_intensity)
            satured = saturation_effect(brightened, midlow_intensity)
            sat_mask = satured / 255
    
            contrasted = color_effect(val_effects, carrousel, midlow_intensity)
            satured = saturation_effect(contrasted, midlow_intensity)
            val_mask = satured / 255
    
            mask = combine_images(hue_mask, sat_mask)
            mask = combine_images(mask, val_mask)
            
            rest = blur(rest, low_intensity)
            rest = rest / 255
    
            added = combine_images(rest, mask)
            added[added>1] = 1
            # added[added<0] = 0
            added = added * 255
            added = added.astype('uint8')
            added = cv2.cvtColor(added, cv2.COLOR_RGB2BGR)
            video.write(added)

    video.release()
    

def main(args):
    samplerate, data = scipy.io.wavfile.read('./aani.wav')
    # data = data[2000000:4000000]

    n_frames = (data.shape[0] / samplerate) * FPS

    #TODO: split into two channels
    spec = plt.specgram(data[:,0], Fs=samplerate)[0]
    
    step_size = spec.shape[1] / n_frames
    
    im = np.array(Image.open('./harem.jpg'))[:,:,:3]
    # im = np.array(Image.open('./kuva.png'))[:,:,:3]
    im = cv2.resize(im, (480, 480))
    plt.imshow(im)

    animate(spec, im, step_size, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create visualized animation")
    parser.add_argument("--music", type=str, help="Music file")
    parser.add_argument("--image", type=str, help="Image file")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    args = parser.parse_args()
    main(args)