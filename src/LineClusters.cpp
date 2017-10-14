#include "LineClusters.h"

namespace VPDetection  {
	using namespace std;
	using namespace cv;

	LineClusters::LineClusters() : Clusters(m_clusters) {
	}

	LineClusters::LineClusters(const LineClusters &lineClusters) :
		Clusters(m_clusters) {
		*this = lineClusters;
	}

	LineClusters::LineClusters(const vector<LineCluster> &lineClusters, const vector<PointPair> &lines) :
		m_clusters(lineClusters), Clusters(m_clusters) {
		constructLineIndexer();

		for (const PointPair &pointPair : lines) {
			Line line(pointPair);
			if (m_lineIndexer.find(line) == m_lineIndexer.end()) {
				m_lineIndexer[line] = OUTLIER_CLUSTER_INDEX;
				m_outlierLines.push_back(line);
			}
		}
	}

	void LineClusters::addLineCluster(const LineCluster &lineCluster) {
		int clusterIndex = (int)m_clusters.size();

		m_clusters.push_back(lineCluster);

		for (int li = 0; li < (int)lineCluster.size(); li++) {
			m_lineIndexer[lineCluster[li]] = clusterIndex;
		}
	}

	void LineClusters::addLine(const Line& l) {
		if (m_lineIndexer.find(l) == m_lineIndexer.end()) {
			m_outlierLines.push_back(l);
			m_lineIndexer[l] = -1;
		}
	}

	void LineClusters::addLines(const vector<Line> &lines) {
		for (const Line &l : lines) {
			if (m_lineIndexer.find(l) == m_lineIndexer.end()) {
				m_outlierLines.push_back(l);
				m_lineIndexer[l] = -1;
			}
		}
	}

	void LineClusters::clear() {
		m_clusters.clear();
		m_lineIndexer.clear();
	}

	void LineClusters::sort() {
		std::sort(m_clusters.begin(), m_clusters.end(), sortClusterByLines);
		constructLineIndexer();
	}

	size_t LineClusters::size() const {
		return m_clusters.size();
	}

	int LineClusters::computeCardinality(cv::Point2f vanishingPoint, float threshold, bool fromOutliers) const {
		int numOfInliers = 0;

		for (unordered_map<Line, int, LineHash>::const_iterator citer = m_lineIndexer.cbegin();
			citer != m_lineIndexer.cend(); citer++) {
			if (citer->second < 0) {
				float dist = Line::LineDistance(vanishingPoint, citer->first);

				if (dist <= threshold) {
					numOfInliers++;
				}
			}
		}

		return numOfInliers;
	}

	void LineClusters::collectInliers(float threshold, bool recomputeVP) {
		vector<Point2f> VPs;
		vector<int> bestIndices(m_outlierLines.size(), -1);

		// Get vanishing points;
		for (auto& cluster : m_clusters) {
			if (recomputeVP) {
				cluster.resetVanishingPoint(threshold);
			}

			VPs.push_back(cluster.getVanishingPoint());
		}

		for (uint i = 0; i < m_outlierLines.size(); i++) {
			float minError = threshold;
			float error;

			for (uint vi = 0; vi < VPs.size(); vi++) {
				if ((error = Line::LineDistance(VPs[vi], m_outlierLines[i])) < minError) {
					minError = error;
					bestIndices[i] = vi;
				}
			}
		}

		auto iter = m_outlierLines.begin();
		for (uint i = 0; i < bestIndices.size(); i++) {
			if (bestIndices[i] >= 0) {
				m_clusters[bestIndices[i]].add(*iter);
				m_lineIndexer[*iter] = bestIndices[i];
				iter = m_outlierLines.erase(iter);
			}
			else {
				iter++;
			}
		}
	}

	LineClusters LineClusters::subCluster(const vector<int> &selectedIndices, const float threshold) const {
		LineClusters lineClusters;

		// Validations
		if (selectedIndices.size() == 0) {
			return lineClusters;
		}

		if (threshold > 0.f) {
			vector<Point2f> VPs;
			vector<vector<Line>> vecLineCluster;
			vector<Line> outliers;

			for (int ci = 0; ci < (int)selectedIndices.size(); ci++) {
				VPs.push_back(LineCluster::computeVanishingPoint(m_clusters[selectedIndices[ci]].Lines));
			}

			vecLineCluster.resize(VPs.size());

			// For each line, choose one model with minimum error
			for (unordered_map<Line, int, LineHash>::const_iterator citer = m_lineIndexer.cbegin();
				citer != m_lineIndexer.cend(); citer++) {
				int bestIndex = -1;
				float minDist = threshold;

				for (int vpi = 0; vpi < (int)VPs.size(); vpi++) {
					float dist = Line::LineDistance(VPs[vpi], citer->first);
					if (dist < minDist) {
						minDist = dist;
						bestIndex = vpi;
					}
				}

				if (bestIndex >= 0) {
					vecLineCluster[bestIndex].push_back(citer->first);
				}
				else {
					outliers.push_back(citer->first);
				}
			}

			for (const vector<Line> &lineCluster : vecLineCluster) {
				lineClusters.addLineCluster(lineCluster);
			}

			lineClusters.addLines(outliers);
		}
		else {
			for (const int idx : selectedIndices) {
				lineClusters.addLineCluster(m_clusters[idx]);
			}
		}

		return lineClusters;
	}

	LineCluster &LineClusters::operator[](const int index) {
		return m_clusters[index];
	}

	const LineCluster &LineClusters::operator[](const int index) const {
		return m_clusters[index];
	}

	LineClusters& LineClusters::operator = (const LineClusters &lineClusters) {
		m_clusters = lineClusters.Clusters;
		m_lineIndexer = lineClusters.m_lineIndexer;
		m_outlierLines = lineClusters.m_outlierLines;

		return *this;
	}

	LineClusters::iterator LineClusters::begin() {
		return m_clusters.begin();
	}

	LineClusters::const_iterator LineClusters::begin() const {
		return m_clusters.cbegin();
	}

	LineClusters::iterator LineClusters::end() {
		return m_clusters.end();
	}

	LineClusters::const_iterator LineClusters::end() const {
		return m_clusters.cend();
	}

	void LineClusters::constructLineIndexer() {
		m_lineIndexer.clear();

		for (int ci = 0; ci < (int)m_clusters.size(); ci++) {
			for (int li = 0; li < m_clusters[ci].size(); li++) {
				m_lineIndexer[m_clusters[ci][li]] = ci;
			}
		}

		for (const Line& l : m_outlierLines) {
			m_lineIndexer[l] = -1;
		}
	}

	bool sortClusterByLines(const LineCluster &left, const LineCluster &right) {
		return left.Lines.size() > right.Lines.size();
	}
};
