#include "main.h"
#include "VPCalibration.h"

using cv::Mat;

int main() {
	Mat image = cv::imread("../image/test1.png");
	Mat K;

	if (image.empty()) {
		std::cout << "Failed to read image file." << std::endl;
		return 0;
	}

	// Calibrate the image with mutually orthogonal vanishing points
	VPCalibration(image, K);
}
