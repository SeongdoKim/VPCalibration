#pragma once

#include "LineCluster.h"

#include <unordered_map>

namespace VPDetection  {
	class LineClusters {
	public:
		/**
		 * Read-only cluster reference
		 */
		const std::vector<LineCluster> &Clusters;
		
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
		 * Defines the index of outliers cluster
		 */
		const static int OUTLIER_CLUSTER_INDEX = -1;

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

		/**
		 * Construct the line indexer.
		 */
		void constructLineIndexer();
	};

	bool sortClusterByLines(const LineCluster &left, const LineCluster &right);
};
