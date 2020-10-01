__author__ = 'Brian M Anderson'
# Created on 9/30/2020
import SimpleITK as sitk
import numpy as np
from enum import Enum


class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)


class SurfaceDistanceMeasures(Enum):
    mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(4)


def identify_overlap_metrics(prediction_handle, ground_truth_handle, perform_distance_measures=False):
    out_dict = {}
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(ground_truth_handle, prediction_handle)
    out_dict[OverlapMeasures.jaccard.name] = overlap_measures_filter.GetJaccardCoefficient()
    out_dict[OverlapMeasures.dice.name] = overlap_measures_filter.GetDiceCoefficient()
    out_dict[OverlapMeasures.volume_similarity.name] = overlap_measures_filter.GetVolumeSimilarity()
    out_dict[OverlapMeasures.false_negative.name] = overlap_measures_filter.GetFalseNegativeError()
    out_dict[OverlapMeasures.false_positive.name] = overlap_measures_filter.GetFalsePositiveError()
    if perform_distance_measures:
        statistics_image_filter = sitk.StatisticsImageFilter()
        reference_surface = sitk.LabelContour(ground_truth_handle)
        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ground_truth_handle, squaredDistance=False,
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


if __name__ == '__main__':
    pass
