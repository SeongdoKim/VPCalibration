#pragma once

#include "LineCluster.h"

#include <unordered_map>

namespace VPDetection  {
	class LineClusters {
	public:
		/**
		 * Type definition of iterator
		 */
		typedef std::vector<LineCluster>::iterator iterator;

		/**
		 * Type definition of constant iterator
		 */
		typedef std::vector<LineCluster>::const_iterator const_iterator;

		/**
		 * Read-only cluster reference
		 */
		const std::vector<LineCluster> &Clusters;

	private:
		/**
		 * Vector of lines that are not include to the cluster
		 */
		std::vector<Line> m_outlierLines;

		/**
		 * Vector of line clusters
		 */
		std::vector<LineCluster> m_clusters;

		/**
		 * Indexer of a line. Note that each line can be added to more than one cluster.
		 */
		std::unordered_map<Line, int, LineHash> m_lineIndexer;

	public:
		/**
		 * Basic empty constructor
		 */
		LineClusters();

		/**
		 * Copy constructor
		 */
		LineClusters(const LineClusters &lineClusters);

		/**
		 * Initialize with given vector of line clusters
		 */
		LineClusters(const std::vector<LineCluster> &lineClusters, const std::vector<PointPair> &lines);

		/**
		 * Add one LineCluster
		 */
		void addLineCluster(const LineCluster &lineCluster);

		/**
		 * Add a line as an outlier.
		 */
		void addLine(const Line& l);

		/**
		 * Add lines as outliers. If a line already exists, it will not be added.
		 */
		void addLines(const std::vector<Line> &lines);

		/**
		 * Clear line clusters
		 */
		void clear();

		/**
		 * Sort clusters by their cardinality
		 */
		void sort();

		/**
		 * Get the number of clusters
		 */
		size_t size() const;

		/**
		 * Compute cardinality of given vanishing point
		 */
		int computeCardinality(cv::Point2f vanishingPoint, float threshold, bool fromOutliers = true) const;

		/**
		 * Collect inliers from outlier lines
		 */
		void collectInliers(float threshold, bool recomputeVP = false);

		/**
		 * Get sub-clusters of this clusters. If threshold is set to greater than zero, 
		 * it will recompute inliers of each cluster in the sub-clusters.
		 */
		LineClusters subCluster(const std::vector<int> &selectedIndices, const float threshold = 0.f) const;

		/**
		 * Direct access to each cluster
		 */
		LineCluster &operator[](const int index);

		/**
		 * Direct access to each cluster for read-only
		 */
		const LineCluster &operator[](const int index) const;

		/**
		 * Overloaded assign operator
		 */
		LineClusters& operator=(const LineClusters &lineClusters);

		/**
		 * The beginning iterator of the line clusters
		 */
		iterator begin();

		/**
		 * The beginning constant iterator of the line clusters
		 */
		const_iterator begin() const;

		/**
		 * The last iterator of the line clusters
		 */
		iterator end();

		/**
		 * The last constant iterator of the line clusters
		 */
		const_iterator end() const;

	private:
		/**
		 * Construct the line indexer.
		 */
		void constructLineIndexer();

	public:
		/**
		 * Defines the index of outliers cluster
		 */
		const static int OUTLIER_CLUSTER_INDEX = -1;
	};

	bool sortClusterByLines(const LineCluster &left, const LineCluster &right);
};
