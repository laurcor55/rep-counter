import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy
from tqdm import tqdm

def sample_video(video_path, desired_fps):
  cap = cv.VideoCapture(video_path)
  frame_count_total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

  frame_skip_interval = int(np.round(cap.get(cv.CAP_PROP_FPS) / desired_fps))
  imgs = []
  frame_count = 1
  while frame_count < frame_count_total:
      ret, frame = cap.read()
      if (np.mod(frame_count, frame_skip_interval) == 0):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        imgs.append(frame)
      frame_count += 1
  cap.release()
  return imgs

def calculate_background(imgs):
  im = np.mean(np.array(imgs), axis=0)
  im = resize(im)
  im = auto_rotate(im)
  return im


def auto_rotate(im):
  if im.shape[1]>im.shape[0]:
    im = np.flipud(im.T)
  return im

def resize(im):
  return cv.resize(im, (int(im.shape[1]/4), int(im.shape[0]/4)))
  
def remove_background(im, im_background):
  im = im -im_background
  im = (im - np.min(im)) / (np.max(im) - np.min(im))
  im = np.uint8(im * 255)
  return im

def plot_frame_minimal(im, correlation, bar_ind_y_top, bar_ind_y_bottom, bar_ind_x):
  plt.figure(figsize=(8, 4))

  ax1 = plt.subplot(1, 2, 1)
  plt.imshow(im)

  ax2 = plt.subplot(1, 2, 2)  
  plt.scatter(np.arange(len(correlation)), correlation)

  plt.sca(ax1)
  plt.scatter(bar_ind_x, bar_ind_y_top)
  plt.scatter(bar_ind_x, bar_ind_y_bottom)



## Antiquated. Keep for reference:
def plot_frame(im, im2, slice, bar_patch, bar_ind_top, bar_ind_bottom, bar_ind_x, bar_height):
  plt.figure(figsize=(12, 4))

  ax1 = plt.subplot(1, 4, 1)
  plt.imshow(im)
  ax2 = plt.subplot(1, 4, 2)
  plt.imshow(im2)

  ax3 = plt.subplot(1, 4, 3)  

  plt.scatter(np.arange(len(slice)), slice)

  plt.subplot(1, 4, 4)
  plt.imshow(bar_patch)
  plt.sca(ax1)
  plt.scatter(bar_ind_x, bar_ind_top)
  plt.scatter(bar_ind_x, bar_ind_bottom)

def get_features(im, bar_height):

  bar = np.ones((2*bar_height, 16))
  bar[int(bar_height/2):int(3*bar_height/2), :] = -1
  im2 = scipy.signal.convolve2d(im, bar, 'same')
  
  bar_ind_x = int(im.shape[1]/4)
  slice = im2[:, bar_ind_x]
  bar_ind_y = np.argmax(slice)

  bar_patch = im[bar_ind_y-bar_height: bar_ind_y+bar_height,bar_ind_x-2: bar_ind_x+2]
  edges = cv.Canny(bar_patch,100, 255)
  edges = np.sum(edges, axis=1)
  edges_top = edges[:bar_height] 
  edges_bottom = edges[bar_height:] 
  bar_ind_top = np.argmax(edges_top)+bar_ind_y -len(edges_bottom)
  bar_ind_bottom = np.argmax(edges_bottom)+bar_ind_y
  bar_height = bar_ind_bottom - bar_ind_top

  return im, im2, slice, bar_patch, bar_ind_top, bar_ind_bottom, bar_ind_x, bar_height