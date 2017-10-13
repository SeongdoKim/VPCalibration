#include "main.h"
#include "VPCalibration.h"

using cv::Mat;

int main() {
	Mat image = cv::imread("*.png");
	Mat K;

	// Calibrate the image with mutually orthogonal vanishing points
	VPCalibration(image, K);
}
