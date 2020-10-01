__author__ = 'Brian M Anderson'
# Created on 9/30/2020
import SimpleITK as sitk
import numpy as np
import copy
from enum import Enum
import matplotlib.pyplot as plt


class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)


class SurfaceDistanceMeasures(Enum):
    mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(4)


def identify_overlap_metrics(prediction_handle, truth_handle, perform_distance_measures=False):
    out_dict = {}
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(truth_handle, prediction_handle)
    out_dict[OverlapMeasures.jaccard.name] = overlap_measures_filter.GetJaccardCoefficient()
    out_dict[OverlapMeasures.dice.name] = overlap_measures_filter.GetDiceCoefficient()
    out_dict[OverlapMeasures.volume_similarity.name] = overlap_measures_filter.GetVolumeSimilarity()
    out_dict[OverlapMeasures.false_negative.name] = overlap_measures_filter.GetFalseNegativeError()
    out_dict[OverlapMeasures.false_positive.name] = overlap_measures_filter.GetFalsePositiveError()
    if perform_distance_measures:
        statistics_image_filter = sitk.StatisticsImageFilter()
        reference_surface = sitk.LabelContour(truth_handle)
        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(truth_handle, squaredDistance=False,
                                                                       useImageSpacing=True))

        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())
        segmented_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(prediction_handle, squaredDistance=False, useImageSpacing=True))
        segmented_surface = sitk.LabelContour(prediction_handle)

        # Multiply the binary surface segmentations with the distance maps. The resulting distance
        # maps contain non-zero values only on the surface (they can also contain zero on the surface)
        seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
        ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

        # Get the number of pixels in the reference surface by counting all pixels that are 1.
        statistics_image_filter.Execute(segmented_surface)
        num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

        # Get all non-zero distances and then add zero distances if required.
        seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
        seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
        seg2ref_distances = seg2ref_distances + \
                            list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
        ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
        ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
        ref2seg_distances = ref2seg_distances + \
                            list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

        all_surface_distances = seg2ref_distances + ref2seg_distances

        # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
        # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
        # segmentations, though in our case it is. More on this below.

        out_dict[SurfaceDistanceMeasures.mean_surface_distance.name] = np.mean(
            all_surface_distances)

        out_dict[SurfaceDistanceMeasures.median_surface_distance.name] = np.median(
            all_surface_distances)
        out_dict[SurfaceDistanceMeasures.std_surface_distance.name] = np.std(
            all_surface_distances)
        out_dict[SurfaceDistanceMeasures.max_surface_distance.name] = np.max(
            all_surface_distances)
    return out_dict


def determine_false_positive_rate_and_false_volume(prediction_handle, truth_handle):
    '''
    :param prediction_handle:
    :param truth_handle:
    :return: a dictionary with False Positive Volume (cc)
    False Predictions Volume (cc), this is the volume not connected to any truth prediction
    Over Segmentation Volume (cc), this is the volume over-segmented on ground truth
    '''
    prediction = sitk.GetArrayFromImage(prediction_handle)
    truth = sitk.GetArrayFromImage(truth_handle)

    out_dict = {}
    spacing = np.prod(truth_handle.GetSpacing())
    volume = np.sum(sitk.GetArrayFromImage(truth_handle)) * spacing / 1000
    out_dict['Volume (cc)'] = volume
    '''
    False Positive Volume (cc) is the easiest one, just subtract them
    '''
    total_difference = np.sum(prediction - truth > 0) * spacing / 1000
    false_volume = total_difference
    out_dict['False Positive Volume (cc)'] = false_volume
    '''
    Next, we want to grow the prediction volume that touches ground truth, so multiple the prediction and ground truth
    '''
    overlap = prediction * truth
    '''
    See if there is any overlap at all first
    '''
    stats = sitk.LabelShapeStatisticsImageFilter()
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    if np.max(overlap) == 0:
        labeled_difference = Connected_Component_Filter.Execute(prediction_handle)
        stats.Execute(labeled_difference)
        difference_labels = stats.GetLabels()
        difference = prediction
    else:
        seeds = np.transpose(np.asarray(np.where(overlap > 0)))[..., ::-1]
        seeds = [[int(i) for i in j] for j in seeds]
        stats = sitk.LabelShapeStatisticsImageFilter()
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        '''
        The seeds represent the starting points of the ground truth, we'll use these to determine truth from false
        '''
        Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        Connected_Threshold.SetUpper(2)
        Connected_Threshold.SetLower(1)
        Connected_Threshold.SetSeedList(seeds)
        seed_grown_pred = sitk.GetArrayFromImage(Connected_Threshold.Execute(prediction_handle))
        difference = prediction - seed_grown_pred  # Now we have our prediction, subtracting those that include truth
        labeled_difference = Connected_Component_Filter.Execute(sitk.GetImageFromArray(difference.astype('int')))
        stats.Execute(labeled_difference)
        difference_labels = stats.GetLabels()
    out_dict['Number False Positives'] = len(difference_labels)
    difference -= truth  # Subtract truth in case seed didn't land
    false_prediction_volume = np.sum(difference > 0) * spacing / 1000
    out_dict['False Prediction Volume (cc)'] = false_prediction_volume
    out_dict['Over Segmentation Volume (cc)'] = false_volume - false_prediction_volume
    return out_dict


def determine_sensitivity(prediction_handle, truth_handle):
    out_dict = {'Site_Number': [], '% Covered': [], 'Volume (cc)': []}
    prediction = sitk.GetArrayFromImage(prediction_handle)
    stats = sitk.LabelShapeStatisticsImageFilter()
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    labeled_truth = Connected_Component_Filter.Execute(truth_handle)
    stats.Execute(labeled_truth)
    tumor_labels = stats.GetLabels()
    spacing = np.prod(truth_handle.GetSpacing())
    for tumor_label in tumor_labels:
        single_site = sitk.GetArrayFromImage(labeled_truth == tumor_label).astype('int')
        total = np.sum(single_site)
        difference = (single_site - prediction) > 0
        remainder = np.sum(difference)
        covered = (total - remainder) / total * 100
        out_dict['Site_Number'].append(tumor_label)
        out_dict['% Covered'].append(covered)
        out_dict['Volume (cc)'].append(total * spacing / 1000)  # Record in cc
    return out_dict


def plot_scroll_Image(x):
    '''
    :param x: input to view of form [rows, columns, # images]
    :return:
    '''
    if x.dtype not in ['float32','float64']:
        x = copy.deepcopy(x).astype('float32')
    if len(x.shape) > 3:
        x = np.squeeze(x)
    if len(x.shape) == 3:
        if x.shape[0] != x.shape[1]:
            x = np.transpose(x,[1,2,0])
        elif x.shape[0] == x.shape[2]:
            x = np.transpose(x, [1, 2, 0])
    fig, ax = plt.subplots(1, 1)
    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=-1)
    tracker = IndexTracker(ax, x)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig,tracker
    #Image is input in the form of [#images,512,512,#channels]


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = np.where((np.min(self.X,axis=(0,1))!= np.max(self.X,axis=(0,1))))[-1]
        if len(self.ind) > 0:
            self.ind = self.ind[len(self.ind)//2]
        else:
            self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


if __name__ == '__main__':
    pass
