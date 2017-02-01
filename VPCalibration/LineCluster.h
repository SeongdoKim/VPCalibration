#pragma once

#include "Line.h"

namespace VPDetection  {
	class LineCluster {
	public:
		/**
		 * Read-only array of lines
		 */
		const std::vector<Line> &Lines;

		/**
		 * Copy constructor
		 */
		LineCluster(const LineCluster& lineCluster);

		/**
		 * Construct a line cluster with vector of lines
		 */
		LineCluster(const std::vector<Line> &lines);

		/**
		 * Add a line to the cluster
		 */
		void add(const Line& line);

		/**
		 * Returns true if the cluster contains vertical lines
		 */
		bool isVertical() const;

		/**
		 * Get vanishing point of the cluster
		 */
		cv::Point2f getVanishingPoint() const;

		/**
		* Not always, but sometimes we need to use a model used to build this cluster.
		*/
		cv::Mat getModel() const;

		/**
		 * Get the number of lines in this cluster
		 */
		size_t size() const;

		/**
		 * Overloaded assign operator
		 */
		LineCluster& operator=(const LineCluster &lineCluster);

		/**
		 * Direct access to each line
		 */
		Line &operator[](const int index);

		/**
		 * Direct access to each line for read-only
		 */
		const Line &operator[](const int index) const;

		/**
		 * Compute the vanishing point with given lines
		 */
		static cv::Point2f computeVanishingPoint(const std::vector<Line> &lines);

	private:
		/**
		 * Indicates if the cluster contains vertical lines
		 */
		bool m_isVertical;

		/**
		 * Vanishing point
		 */
		cv::Point2f m_vanishingPoint;

		/**
		 * A model used to build this cluster
		 */
		cv::Mat m_model;

		/**
		 * Actual array of lines
		 */
		std::vector<Line> m_lines;

		/**
		 * Compute intersection point of lines
		 */
		void computeVanishingPoint();

		/**
		 * Makes the StartPoint to be closer to the vanishing point than the EndPoint.
		 */
		void reorderEndpoints();
	};
};
