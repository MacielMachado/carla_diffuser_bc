import h5py
import numpy as np
from PIL import Image


maps_h5_path = 'carla_gym/core/obs_manager/birdview/maps/Town02.h5'
h5_file = h5py.File(maps_h5_path, 'r', libver='latest', swmr=True)
road = np.array(h5_file['road'], dtype=np.uint8)
lane_marking_all = np.array(h5_file['lane_marking_all'], dtype=np.uint8)
lane_marking_white_broken = np.array(h5_file['lane_marking_white_broken'], dtype=np.uint8)
print(road.shape)
print(lane_marking_all.shape)
print(lane_marking_white_broken.shape)
image_array = np.zeros((1600, 1600, 3), dtype=np.uint8)
image_array[:, :, 0] = road[203:1803, 203:1803]
image_array[:, :, 1] = road[203:1803, 203:1803]
image_array[:, :, 2] = road[203:1803, 203:1803]
image_array = np.rot90(image_array, 1, (0,1))
# image_array = np.transpose(image_array, [1, 2, 0]).astype(np.uint8)
Image.fromarray(image_array).save('road.png')
image_array[:, :, 0] = lane_marking_all[203:1803, 203:1803]
image_array[:, :, 1] = lane_marking_all[203:1803, 203:1803]
image_array[:, :, 2] = lane_marking_all[203:1803, 203:1803]
image_array = np.rot90(image_array, 1, (0,1))
Image.fromarray(image_array).save('lane_marking_all.png')
image_array[:, :, 0] = lane_marking_white_broken[203:1803, 203:1803]
image_array[:, :, 1] = lane_marking_white_broken[203:1803, 203:1803]
image_array[:, :, 2] = lane_marking_white_broken[203:1803, 203:1803]
image_array = np.rot90(image_array, 1, (0,1))
Image.fromarray(image_array).save('lane_marking_white_broken.png')