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

	void LineCluster::reorderEndpoints() {
		for (Line& l : m_lines) {
			float distStart = norm(m_vanishingPoint - l.StartPoint);
			float distEnd = norm(m_vanishingPoint - l.EndPoint);

			if (distStart > distEnd) {
				m_lines.back().swapPoints();
			}
		}
	}
};
