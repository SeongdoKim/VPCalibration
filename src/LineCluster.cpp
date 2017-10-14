#include "LineCluster.h"

namespace VPDetection  {
	using namespace std;
	using namespace cv;

	LineCluster::LineCluster(const LineCluster& lineCluster) :
		Lines(m_lines) {
		*this = lineCluster;
	}

	LineCluster::LineCluster(const vector<Line> &lines) :
		m_lines(lines), Lines(m_lines) {
		computeVanishingPoint();
		reorderEndpoints();
	}

	void LineCluster::add(const Line& line) {
		m_lines.push_back(line);

		float distStart = norm(m_vanishingPoint - line.StartPoint);
		float distEnd = norm(m_vanishingPoint - line.EndPoint);

		if (distStart > distEnd) {
			m_lines.back().swapPoints();
		}
	}

	bool LineCluster::isVertical() const {
		return m_isVertical;
	}

	Point2f LineCluster::getVanishingPoint() const {
		return m_vanishingPoint;
	}

	void LineCluster::resetVanishingPoint(float threshold) {
		if (threshold <= 0.f) {
			Mat lineMat(m_lines.size(), 3, CV_32F);
			Mat solution;

			for (int row = 0; row < m_lines.size(); row++) {
				lineMat.row(row) = m_lines[row].LineVector.t();
				/*vec_cross(m_lines[row].StartPoint.x, m_lines[row].StartPoint.y, 1.f,
				m_lines[row].EndPoint.x, m_lines[row].EndPoint.y, 1.f,
				lineMat.at<float>(row, 0), lineMat.at<float>(row, 1), lineMat.at<float>(row, 2));*/
			}

			cv::SVD::solveZ(lineMat, solution);

			m_vanishingPoint.x = solution.at<float>(0) / solution.at<float>(2);
			m_vanishingPoint.y = solution.at<float>(1) / solution.at<float>(2);
		}
		else {
			map<float, Line, greater<float>> orderedLines;
			multimap<int, pair<float, Point2f>, greater<int>> orderedModelsByCard;
			multimap<float, Point2f> orderedModelsByError;
			vector<Line> candidateLines;

			// Insert and order lines by their length
			for (const Line &l : m_lines) {
				orderedLines.insert({ l.length(), l });
			}

			// Pick a part of lines by order, compute a VP, and check how much the lines are aligned with the VP
			for (auto iter = orderedLines.begin(); iter != orderedLines.end(); iter++) {
				if (candidateLines.size() >= 2) {
					Point2f vp = LineCluster::computeVanishingPoint(candidateLines);
					int numInliers = 0;
					float sumError = 0.f;

					for (const Line &l : m_lines) {
						float error = Line::LineDistance(vp, l);
						if (error < threshold) {
							numInliers++;
						}
						sumError += error;
					}

					orderedModelsByCard.insert({ numInliers,{ sumError, vp } });
				}

				candidateLines.push_back(iter->second);
			}

			// Select a part of candidating models with the largest cardinality
			for (auto iter = orderedModelsByCard.begin(); iter != orderedModelsByCard.end();) {
				orderedModelsByError.insert(iter->second);
				int num_crt = iter->first;

				if (++iter == orderedModelsByCard.end() || num_crt != iter->first) {
					break;
				}
			}

			m_vanishingPoint = orderedModelsByError.begin()->second;
		}

		m_isVertical = (abs(m_vanishingPoint.y) / abs(m_vanishingPoint.x) > 5.f);
	}

	Mat LineCluster::getModel() const{
		return m_model;
	}

	size_t LineCluster::size() const {
		return m_lines.size();
	}

	LineCluster& LineCluster::operator=(const LineCluster &lineCluster) {
		m_isVertical = lineCluster.isVertical();
		m_lines = lineCluster.Lines;
		m_vanishingPoint = lineCluster.getVanishingPoint();
		m_model = lineCluster.getModel();

		return *this;
	}

	Line &LineCluster::operator[](const int index) {
		return m_lines[index];
	}

	const Line &LineCluster::operator[](const int index) const {
		return m_lines[index];
	}

	LineCluster::iterator LineCluster::begin() {
		return m_lines.begin();
	}

	LineCluster::const_iterator LineCluster::begin() const {
		return m_lines.cbegin();
	}

	LineCluster::iterator LineCluster::end() {
		return m_lines.end();
	}

	LineCluster::const_iterator LineCluster::end() const {
		return m_lines.cend();
	}

	void LineCluster::computeVanishingPoint() {
		Mat lineMat(m_lines.size(), 3, CV_32F);
		Mat solution;

		for (int row = 0; row < m_lines.size(); row++) {
			lineMat.row(row) = m_lines[row].LineVector.t();
			/*vec_cross(m_lines[row].StartPoint.x, m_lines[row].StartPoint.y, 1.f,
			m_lines[row].EndPoint.x, m_lines[row].EndPoint.y, 1.f,
			lineMat.at<float>(row, 0), lineMat.at<float>(row, 1), lineMat.at<float>(row, 2));*/
		}

		cv::SVD::solveZ(lineMat, solution);

		m_vanishingPoint.x = solution.at<float>(0) / solution.at<float>(2);
		m_vanishingPoint.y = solution.at<float>(1) / solution.at<float>(2);

		m_isVertical = (abs(m_vanishingPoint.y) / abs(m_vanishingPoint.x) > 5.f);
	}

	void LineCluster::reorderEndpoints() {
		for (Line& l : m_lines) {
			float distStart = norm(m_vanishingPoint - l.StartPoint);
			float distEnd = norm(m_vanishingPoint - l.EndPoint);

			if (distStart > distEnd) {
				m_lines.back().swapPoints();
			}
		}
	}

	Point2f LineCluster::computeVanishingPoint(const vector<Line> &lines) {
		Mat lineMat(lines.size(), 3, CV_32F);
		Mat solution;

		assert(lines.size() >= 2);

		for (int row = 0; row < lines.size(); row++) {
			lineMat.row(row) = lines[row].LineVector.t();
			/*vec_cross(m_lines[row].StartPoint.x, m_lines[row].StartPoint.y, 1.f,
			m_lines[row].EndPoint.x, m_lines[row].EndPoint.y, 1.f,
			lineMat.at<float>(row, 0), lineMat.at<float>(row, 1), lineMat.at<float>(row, 2));*/
		}

		cv::SVD::solveZ(lineMat, solution);

		return Point2f(solution.at<float>(0) / solution.at<float>(2), solution.at<float>(1) / solution.at<float>(2));
	}
};
