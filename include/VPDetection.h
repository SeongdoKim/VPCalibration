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

	//----------------------------------
	// Cluster optimization
	//----------------------------------

	/**
	 * Choose three clusters with mutually orthogonal vanishing directions.
	 * The initial calibration matrix K does not need to be accurate, and will be refined.
	 * Returns a 3x3 matrix where each column represents vanishing directions.
	 * The first column aways represents the vertical vanishing direction.
	 */
	cv::Mat detectVP(const LineClusters &lineClusters, std::vector<int> &selectedIndices, cv::Mat &K,
		bool refineK, float threshold, bool extendInliers, const cv::Mat &debug_image = cv::Mat());

	/**
	 * Find clusters with mutually orthogonal vanishing directions. Vanishing directions will be computed using
	 * given calibration matrix. You can expect good selection of clusters even if the calibration matrix is not accurate.
	 * threshold: inlier threshold. Typically this threshold is set to around 2.0.
	 */
	std::vector<int> findOrthogonalClusters(const LineClusters &lineClusters, const cv::Mat &K, float threshold, cv::Mat &VD = cv::Mat());

	/**
	 * Refine given calibration matrix and vanishing directions of input clusters.
	 * The vanishing directions of input cluster must be perpendicular each other.
	 * It will return the matrix of refined vanishing directions.
	 */
	cv::Mat refineCalibration(const LineClusters &lineClusters, const cv::Mat &VD, cv::Mat &K, float inlierThreshold, bool refineK);

	/**
	 * Cluster lines detected from given image.
	 */
	LineClusters lineClustering(cv::Mat image, int numModels = 1000, int minCardinarity = 2, float distThreshold = 2.5f,
		float lengthThreshold = -1, bool allowDuplicates = false, bool draw_output = true);

	inline double LineDistance(const cv::Point2f &point, const float *line);

	inline cv::Point3f computeIntersection(const PointPair &line1, const PointPair &line2);

	/**
	 * QR decomposition based on the House Holder Algorithm (https://en.wikipedia.org/wiki/QR_decomposition).
	 */
	void HouseHolderQR(const cv::Mat &A, cv::Mat &Q, cv::Mat &R);

	/**
	 * Draw line clusters on a clone of the given image, and return it.
	 */
	cv::Mat drawLineClusters(const cv::Mat &image, const LineClusters &lineClusters, int lineWidth, bool convertGray = false);

	/**
	 * Draw vanishing directions on a clone of the given image, and return it.
	 */
	cv::Mat drawVanishingDirections(const cv::Mat &image, const cv::Mat &VDs, const cv::Mat &K);
};
