#pragma once

#include <opencv2\opencv.hpp>

namespace VPDetection {
	typedef std::pair<cv::Point2f, cv::Point2f> PointPair;
	
	class Line {
	public:
		/**
		 * Start point of the line (read-only)
		 */
		cv::Point2f &StartPoint;

		/**
		 * End point of the line (read-only)
		 */
		cv::Point2f &EndPoint;

		/**
		 * Vector representation of the line (read-only)
		 */
		cv::Mat &LineVector;

		/**
		 * Copy constructor
		 */
		Line(const Line &l);

		/**
		* Basic constructor with its endpoints
		*/
		Line(const std::pair<cv::Point2f, cv::Point2f> &endPoints);

		/**
		 * Basic constructor with its endpoints
		 */
		Line(const cv::Point2f &startPoint, const cv::Point2f &endPoint);

		/**
		 * Compute length of the line
		 */
		float length() const;

		/**
		 * Compute mid-point
		 */
		cv::Point2f getMidPoint() const;

		/**
		 * Set endpoints of the line
		 */
		void setPoints(const cv::Point2f &startPoint, const cv::Point2f &endPoint);

		/**
		 * Swap start and end points
		 */
		void swapPoints();

		/**
		 * Compute distance between this line and given point. If computeSign is set to true,
		 * distance will be computed with sign.
		 */
		float computeDistance(const cv::Point2f &point, bool computeSign = false) const;

		/**
		 * Compute sign with respect to this line.
		 */
		float computeSign(const cv::Point2f &point) const;

		/**
		 * Compute an intersection point of the line and given line
		 */
		cv::Point2f getIntersection(const Line &line) const;

		/**
		 * Compute an intersection point of the line and given line, which is expressed as its normal.
		 */
		cv::Point2f getIntersection(const cv::Mat &normal) const;

		/**
		 * Return either StartPoint or EndPoint that is further from given point.
		 */
		cv::Point2f getFurtherPointFrom(const cv::Point2f &fromPoint) const;

		/**
		 * If given point is on the line, return true.
		 */
		bool Line::isPointOnLine(cv::Point2f testPoint) const;

		/**
		 * Assign a line
		 */
		Line& operator=(const Line &l);

		/**
		 * Compare with the other line
		 */
		bool operator== (const Line &l) const;

		/**
		 * Compute normal of a line connecting pt1 and pt2.
		 */
		static cv::Mat computeLineNormal(const cv::Point2f &pt1, const cv::Point2f &pt2);

		/**
		 * Compute the distance between given model and line. It first calculates the line
		 * connecting the mid-point of given line sample (line) and vanishing point (model).
		 * Then, residual of the starting point of given line for the computed line will be
		 * defined as the distance.
		 */
		static float LineDistance(const cv::Point2f &model, const Line &line);

	private:
		/**
		 * Start point of the line
		 */
		cv::Point2f m_startPoint;

		/**
		 * End point of the line
		 */
		cv::Point2f m_endPoint;

		/**
		 * Vector representation of the line
		 */
		cv::Mat m_lineVector;
	};

	/**
	 * Compute cross-product of given 3D vectors
	 */
	inline void vec_cross(float a1, float b1, float c1,
		float a2, float b2, float c2,
		float& a3, float& b3, float& c3);

	/**
	 * Normalize the given 3D vector
	 */
	inline void vec_norm(float& a, float& b, float& c);

	/**
	 * Hash seed of the Line class to be used with STL containers.
	 */
	class LineHash {
	public:
		std::size_t operator()(const Line &l) const;
	};
}