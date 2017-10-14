#pragma once

#include "Line.h"

namespace VPDetection  {
	class LineCluster {
	public:
		/**
		 * Type definition of iterator
		 */
		typedef std::vector<Line>::iterator iterator;

		/**
		 * Type definition of constant iterator
		 */
		typedef std::vector<Line>::const_iterator const_iterator;

		/**
		 * Read-only array of lines
		 */
		const std::vector<Line> &Lines;

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

	public:
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
		 * Re-compute the vanishing point of this cluster
		 */
		void resetVanishingPoint(float threshold);

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

		//-------------------------------
		// Functions for the standard access
		//-------------------------------

		/**
		 * The beginning iterator of the line cluster
		 */
		iterator begin();

		/**
		 * The beginning constant iterator of the line cluster
		 */
		const_iterator begin() const;

		/**
		 * The last iterator of the line cluster
		 */
		iterator end();

		/**
		 * The last constant iterator of the line cluster
		 */
		const_iterator end() const;

	private:

		/**
		 * Compute intersection point of lines
		 */
		void computeVanishingPoint();

		/**
		 * Makes the StartPoint to be closer to the vanishing point than the EndPoint.
		 */
		void reorderEndpoints();

	public:

		/**
		 * Compute the vanishing point with given lines
		 */
		static cv::Point2f computeVanishingPoint(const std::vector<Line> &lines);
	};
};
