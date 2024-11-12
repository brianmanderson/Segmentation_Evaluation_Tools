# Segmentation_Evaluation_Tools
There are a set of tools for creating quantitative comparison metrics based on ground truth and prediction SITK Image handles

## Installation
    pip install SegmentationEvaluationTools

### Usage
    from SegmentationEvaluationTools.SIKOverlapTools import calculate_overlap_measures, determine_sensitivity, 
    determine_false_positive_rate_and_false_volume, sitk
    
    truth_handle_base = sitk.ReadImage(image_path)
    prediction_handle_base = sitk.ReadImage(prediction_path)
    
    overlap_measures = calculate_overlap_measures(prediction_handle_base, truth_handle_base, measure_as_multiple_sites=False, perform_distance_measures=False)
    
    fp_measures = determine_false_positive_rate_and_false_volume(prediction_handle_base, truth_handle_base)
    
    sensitivity_measures = determine_sensitivity(prediction_handle=prediction_handle_base, truth_handle_base)
