#pragma once

#include <vector>

#include <opencv2\opencv.hpp>
#include <opencv2\line_descriptor.hpp>

#include <boost\functional\hash.hpp>

#include "LineClusters.h"

namespace VPDetection  {
	class IntPairHash {
	public:
		std::size_t operator()(const std::pair<int, int>& lhs) const {
			std::size_t seed = 0;
			boost::hash_combine(seed, lhs.first);
			boost::hash_combine(seed, lhs.second);
			return seed;
		}
	};

	/**
	 * Generate a mask of empty space, where the empty space is represented as a zero-intensity.
	 */
	cv::Mat generateEmptySpaceMask(const cv::Mat &grayImage, int kernelSize = 3);

	/**
	 * Line merging algorithm presented in the paper of "Manuel, A New Approach for Merging Edge Line
	 * Segments, JMRS 1995".
	 * @returns Distance between the projected extrem points of two line segment onto the merged line -
	 *  total length of projected two line segment onto the merged line. If the value is negative,
	 *  then the two line segments were overlapped each other.
	 */
	float mergeLines(const cv::line_descriptor::KeyLine &line1,
		const cv::line_descriptor::KeyLine &line2, cv::line_descriptor::KeyLine &newLine);

	/**
	* Merge lines if a pair of lines have similar line property.
	*/
	void mergeLines(std::vector<cv::line_descriptor::KeyLine> &lines,
		float lineGapThreshold = 30.f, float lineDistThreshold = 1.0f);

	/**
	 * Detect lines from given image
	 */
	std::vector<PointPair> detectLines(const cv::Mat &image, float lineLengthThreshold = 50.f);

	/**
	 * Generate models of vanishing points by randomly selecting two lines.
	 */
	std::vector<cv::Point3f> generateVPModels(const std::vector<PointPair> &lines, int numOfModels, uint64 seed = 0xFFFFFFFF);

	/**
	 * Generate clusters of lines with the result of T-linkage
	 */
	LineClusters generateClusters(const std::vector<PointPair>& lines,
		const std::vector<cv::Point3f>& models, const float *prefMat, const std::vector<std::pair<int, int>> &T);

	/**
	 * Cluster lines detected from given image.
	 */
	LineClusters lineClustering(cv::Mat image, int numModels = 1000, int minCardinarity = 2, float distThreshold = 2.5f,
		float lengthThreshold = -1, bool allowDuplicates = false);

	inline double LineDistance(const cv::Point2f &point, const float *line);

	inline cv::Point3f computeIntersection(const PointPair &line1, const PointPair &line2);
};
