#include "Line.h"
#include <boost\functional\hash.hpp>

namespace VPDetection {
	using namespace std;
	using namespace cv;

	Line::Line(const Line &l) : m_startPoint(l.StartPoint), m_endPoint(l.EndPoint),
		StartPoint(m_startPoint), EndPoint(m_endPoint), LineVector(m_lineVector) {
		m_lineVector = l.LineVector.clone();
	}

	Line::Line(const pair<Point2f, Point2f> &endPoints)
		: StartPoint(m_startPoint), EndPoint(m_endPoint), LineVector(m_lineVector) {
		setPoints(endPoints.first, endPoints.second);
	}

	Line::Line(const Point2f &startPoint, const Point2f &endPoint)
		: StartPoint(m_startPoint), EndPoint(m_endPoint), LineVector(m_lineVector) {
		setPoints(startPoint, endPoint);
	}

	float Line::length() const {
		return sqrt(pow(m_startPoint.x - m_endPoint.x, 2) + pow(m_startPoint.y - m_endPoint.y, 2));
	}

	Point2f Line::getMidPoint() const {
		return Point2f((m_startPoint.x + m_endPoint.x) / 2.f, (m_startPoint.y + m_endPoint.y) / 2.f);
	}

	void Line::swapPoints() {
		Point2f tempPoint = m_startPoint;
		m_startPoint = m_endPoint;
		m_endPoint = tempPoint;
	}

	void Line::setPoints(const Point2f &startPoint, const Point2f &endPoint) {
		m_startPoint = startPoint;
		m_endPoint = endPoint;
		m_lineVector = computeLineNormal(m_startPoint, m_endPoint);
	}

	float Line::computeDistance(const cv::Point2f &point, bool computeSign) const {
		Mat p = (Mat_<float>(3, 1) << point.x, point.y, 1.f);
		float sign = 1.f;

		// TODO: Check if the following method work for groups of non-vertical lines
		if (computeSign) {
			sign = this->computeSign(point);
		}

		return abs(m_lineVector.at<float>(0) * point.x + m_lineVector.at<float>(1) * point.y + m_lineVector.at<float>(2)) * sign;
	}

	float Line::computeSign(const cv::Point2f &point) const {
		if ((m_endPoint.x - m_startPoint.x)*(point.y - m_startPoint.y) - (m_endPoint.y - m_startPoint.y)*(point.x - m_startPoint.x) < 0) {
			return -1.f;
		}
		else {
			return 1.f;
		}
	}

	Point2f Line::getIntersection(const Line &line) const {
		Mat intersectPoint = m_lineVector.cross(line.LineVector);
		return Point2f(intersectPoint.at<float>(0) / intersectPoint.at<float>(2),
			intersectPoint.at<float>(1) / intersectPoint.at<float>(2));
	}

	Point2f Line::getIntersection(const Mat &normal) const {
		Mat intersectPoint = m_lineVector.cross(normal);
		return Point2f(intersectPoint.at<float>(0) / intersectPoint.at<float>(2),
			intersectPoint.at<float>(1) / intersectPoint.at<float>(2));
	}

	Point2f Line::getFurtherPointFrom(const Point2f &fromPoint) const {
		if (norm(fromPoint - StartPoint) > norm(fromPoint - EndPoint)) {
			return StartPoint;
		}
		else {
			return EndPoint;
		}
	}

	bool Line::isPointOnLine(Point2f testPoint) const {
		if (testPoint.x >= min(StartPoint.x, EndPoint.x) &&
			testPoint.x <= max(StartPoint.x, EndPoint.x) &&
			testPoint.y >= min(StartPoint.y, EndPoint.y) &&
			testPoint.y <= max(StartPoint.y, EndPoint.y)) {
			return true;
		}

		return false;
	}

	Line& Line::operator = (const Line &l) {
		m_startPoint = l.StartPoint;
		m_endPoint = l.EndPoint;
		m_lineVector = l.LineVector.clone();

		return *this;
	}

	bool Line::operator== (const Line &l) const {
		return l.StartPoint == StartPoint && l.EndPoint == EndPoint;
	}

	Mat Line::computeLineNormal(const Point2f &pt1, const Point2f &pt2) {
		Mat normal(3, 1, CV_32F);

		vec_cross(pt1.x, pt1.y, 1.f,
			pt2.x, pt2.y, 1.f,
			normal.at<float>(0, 0), normal.at<float>(1, 0), normal.at<float>(2, 0));

		vec_norm(normal.at<float>(0, 0), normal.at<float>(1, 0), normal.at<float>(2, 0));

		return normal;
	}

	float Line::LineDistance(const cv::Point2f &model, const Line &line) {
		float l[3];
		float midPoint[3] = { (line.StartPoint.x + line.EndPoint.x) / 2.0, (line.StartPoint.y + line.EndPoint.y) / 2.0, 1 };

		vec_cross(midPoint[0], midPoint[1], midPoint[2],
			model.x, model.y, 1.f,
			l[0], l[1], l[2]);

		return fabs(l[0] * line.StartPoint.x + l[1] * line.StartPoint.y + l[2]) / sqrt(l[0] * l[0] + l[1] * l[1]);
	}

	inline void vec_cross(float a1, float b1, float c1,
		float a2, float b2, float c2,
		float& a3, float& b3, float& c3) {
		a3 = b1*c2 - c1*b2;
		b3 = -(a1*c2 - c1*a2);
		c3 = a1*b2 - b1*a2;
	}

	inline void vec_norm(float& a, float& b, float& c) {
		float len = sqrt(a*a + b*b + c*c);
		a /= len; b /= len; c /= len;
	}

	size_t LineHash::operator()(const Line &l) const {
		size_t seed = 0;

		boost::hash_combine(seed, l.StartPoint.x);
		boost::hash_combine(seed, l.StartPoint.y);
		boost::hash_combine(seed, l.EndPoint.x);
		boost::hash_combine(seed, l.EndPoint.y);

		return seed;
	}
};