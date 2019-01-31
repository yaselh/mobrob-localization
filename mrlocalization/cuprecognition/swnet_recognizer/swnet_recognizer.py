import os
from keras.models import load_model
import skimage.transform

predictor_relative_path = '../../deepnn/models/swnet'
import sys
predictor_path = os.path.join(os.path.dirname(__file__), predictor_relative_path)
sys.path.append(predictor_path)
from swnet_predictor import SWNetPredictor
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

from sklearn.cluster import DBSCAN

class SWNetRecognizer:



    def initialize_recognizer(self):
        self.predictor = SWNetPredictor()
        self.predictor.initialize_predictor()

    def resize_image(self, image, max_side_length):
        image_height = image.shape[0]
        image_width = image.shape[1]
        ratio = min(max_side_length / image_height, max_side_length / image_width);
        image = skimage.transform.resize(image, (int(image_height * ratio), int(image_width * ratio)))
        return image

    def get_sliding_window_patches(self, image, height=50, width=50, stride=5, start_row=0, start_col=0, end_row=-1, end_col=-1):
        row_count = image.shape[0]
        col_count = image.shape[1]
        patches = [] # Format (row, col, height, width)
        if end_row  == -1:
            end_row = row_count
        if end_col == -1:
            end_col = col_count

        for col in range(start_col, (end_col - width) + 1, stride):
            for row in range(start_row, (end_row - height) + 1, stride):
                patch = image[row:row+height, col:col+width, :]
                patches.append((row, col, height, width, patch))
        return patches

    def recognize_mock(self, image, pointcloud):
        bboxes = [(150,280,80,80)]
        return bboxes, None

    def recognize(self, image, pointcloud):
        original_image_height = image.shape[0]
        original_image_width = image.shape[1]
        image_levels = [600]
        image_level_imgs = []

        overall_results = []
        overall_bboxes = []
        for i, image_level in enumerate(image_levels):
            image_level_imgs.append(self.resize_image(image, image_level))
            if image_level_imgs[i].shape[0] < 60 or image_level_imgs[i].shape[1] < 60:
                break
            patches_with_pos = self.get_sliding_window_patches(image_level_imgs[i], 60, 60, 10)
            patches = []
            # Prepare patch data for prediction
            for patch in patches_with_pos:
                patches.append(patch[4])
            result = self.predictor.predict(patches)
            # Append results for this level to overall result array
            overall_results.extend(result.tolist())
            # Append bounding box positions and sizes to overall bbox array
            current_image_height = image_level_imgs[i].shape[0]
            height_ratio = original_image_height / current_image_height
            current_image_width = image_level_imgs[i].shape[1]
            width_ratio = original_image_width / current_image_width
            for patch in patches_with_pos:
                original_patch = (int(patch[0] * height_ratio), int(patch[1] * width_ratio), int(patch[2] * height_ratio), int(patch[3] * width_ratio))
                overall_bboxes.append(original_patch)

        overall_results, overall_bboxes = zip(*sorted(zip(overall_results, overall_bboxes), reverse=True))
        print(overall_results[0])


        best_x_bboxes_count = 100
        best_x_bboxes_threshold = 0.80
        is_low_probability = False
        best_x_bboxes = []
        best_x_bbox_centers = []
        for i, bbox in enumerate(overall_bboxes):
            if overall_results[i] >= best_x_bboxes_threshold:
                best_x_bboxes.append(bbox)
                best_x_bbox_centers.append([int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)])
            else:
                is_low_probability = True
            if i >= best_x_bboxes_count or is_low_probability:
                break
        best_x_bboxes_np = np.asarray(best_x_bboxes)
        best_x_bbox_centers_np = np.asarray(best_x_bbox_centers)


        db = DBSCAN(eps=30.0, min_samples=3).fit(best_x_bbox_centers_np)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        combined_bboxes = []
        for i in range(0, n_clusters):
            current_label_bboxes = best_x_bboxes_np[labels==i]
            averages = np.average(current_label_bboxes, axis=0)
            average_row = averages[0]
            average_col = averages[1]
            average_height = averages[2]
            average_width = averages[3]
            combined_bboxes.append((int(average_row), int(average_col), int(average_height), int(average_width)))



        #Result evaluation
        fig,ax = plt.subplots(1)
        # Display the image
        ax.imshow(image)
        for label, bbox_center in zip(labels, best_x_bbox_centers_np):
                ax.scatter(x=[bbox_center[1]], y=[bbox_center[0]], color='r')

        for bbox in combined_bboxes:
            rect = matplotlib.patches.Rectangle((bbox[1],bbox[0]),bbox[3],bbox[2],linewidth=1,edgecolor='b',facecolor='none')
                # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()



        print('n_clusters: {}'.format(n_clusters))

        # Result evaluation
        #fig,ax = plt.subplots(1)
        # Display the image
        #ax.imshow(image)
        # Create rectangle patches
        #i=0
        #for result, bbox in zip(overall_results, overall_bboxes):
        #    rect = matplotlib.patches.Rectangle((bbox[1],bbox[0]),bbox[3],bbox[2],linewidth=1,edgecolor=cm.viridis(result),facecolor='none')
            # Add the patch to the Axes
        #    ax.add_patch(rect)
        #    if i > 50:
        #        break
        #    i = i + 1
        #plt.show()

        return None, None
