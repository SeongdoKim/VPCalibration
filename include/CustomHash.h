#pragma once

#include <opencv2\opencv.hpp>
#include <boost\functional\hash.hpp>

/**
* Hash seed of the Line class to be used with STL containers.
*/
class CustomHash {
public:
public:
	std::size_t operator()(const std::pair<cv::Point2f, cv::Point2f>& lhs) const {
		std::size_t seed = 0;
		boost::hash_combine(seed, lhs.first.x);
		boost::hash_combine(seed, lhs.first.y);
		boost::hash_combine(seed, lhs.second.x);
		boost::hash_combine(seed, lhs.second.y);

		return seed;
	}

	std::size_t operator()(const cv::Point2f& lhs) const {
		std::size_t seed = 0;
		boost::hash_combine(seed, lhs.x);
		boost::hash_combine(seed, lhs.y);

		return seed;
	}

	std::size_t operator()(const cv::Mat& lhs) const {
		std::size_t seed = 0;
		for (int r = 0; r < cv::min(lhs.rows, 4); r++) {
			for (int c = 0; c < cv::min(lhs.cols, 4); c++) {
				boost::hash_combine(seed, lhs.at<float>(r, c));
			}
		}

		return seed;
	}
};
