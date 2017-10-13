#include "VPCalibration.h"
#include "VPDetection.h"

using cv::Mat;

void VPCalibration(const Mat& image, Mat& K) {
	// Detect and cluster lines
	VPDetection::LineClusters lineClusters = VPDetection::lineClustering(image);
}
