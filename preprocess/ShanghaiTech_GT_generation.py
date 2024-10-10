import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy
import threading


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)  # nonzero value represent people in labels
    if gt_count == 0:  # gt_count is the amount of people
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))  # human label position
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density

def worker(thread_i):
    for img_path in img_paths[int(thread_i)]:
        # for every image
        print(f'image path: {img_path}')
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density(k)
        # save the Density Maps GT as h5 format
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:
            hf['density'] = k


root = 'F:\\Notes\\CV\\SKT\\ShanghaiTech-test'
print(root)

# now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final\\train_data','images')
part_A_test = os.path.join(root,'part_A_final\\test_data','images')
part_B_train = os.path.join(root,'part_B_final\\train_data','images')
part_B_test = os.path.join(root,'part_B_final\\test_data','images')
path_sets = [part_A_train,part_A_test,part_B_train,part_B_test]

img_paths = [[], [], [] ,[],]
for i in range(len(path_sets)):
    for img_path in glob.glob(os.path.join(path_sets[i], '*.jpg')):
        img_paths[i].append(img_path)

threads = []
for i in range(4):
    thread = threading.Thread(target=worker, args=(f"{i}"))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()