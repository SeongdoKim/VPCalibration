#include "VPCalibration.h"
#include "VPDetection.h"

using cv::Mat;
using std::vector;

void VPCalibration(const Mat& image, Mat& K, bool draw_output) {
	// Detect and cluster lines
	VPDetection::LineClusters lineClusters = VPDetection::lineClustering(image, 1000, 2, 2.5, -1.f, false, draw_output);

	// Optimize the line clusters so that their vanishing directions are mutually perpendicular
	float lineThreshold = 2.f;
	vector<int> selectedLineClusters;
	Mat VD = VPDetection::detectVP(lineClusters, selectedLineClusters, K, true, lineThreshold, true, image);

	if (draw_output) {
		Mat debugImage = VPDetection::drawLineClusters(image, lineClusters, 2, true);
		debugImage = VPDetection::drawVanishingDirections(debugImage, VD, K);
		cv::imwrite("line_clustering.png", debugImage);
	}
}
