#pragma once

#include <opencv2\opencv.hpp>

void VPCalibration(const cv::Mat& image, cv::Mat& K, bool draw_output = true);
