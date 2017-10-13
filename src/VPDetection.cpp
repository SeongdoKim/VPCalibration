#include "VPDetection.h"
#include "Line.h"
#include "TLinkageGPU.h"
#include "CustomHash.h"

#include <unordered_set>
#include <unordered_map>

namespace VPDetection {
	using namespace std;
	using namespace cv;

	using cv::line_descriptor::KeyLine;

	Mat generateEmptySpaceMask(const Mat &grayImage, int kernelSize) {
		Mat mask = grayImage * 255;
		Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * kernelSize + 1, 2 * kernelSize + 1),
			Point(kernelSize, kernelSize));

		// Apply the erosion operation
		erode(mask, mask, element);

		return mask;
	}

	float mergeLines(const KeyLine &line1, const KeyLine &line2, KeyLine &newLine) {
		// Compute length of lines from the origin
		float len1 = sqrt(line1.pt.x * line1.pt.x + line1.pt.y * line1.pt.y);
		float len2 = sqrt(line2.pt.x * line2.pt.x + line2.pt.y * line2.pt.y);

		float w1 = len1 / (len1 + len2);
		float w2 = len2 / (len1 + len2);

		Point2f midPoint;
		midPoint.x = (w1*(line1.startPointX + line1.endPointX) + w2*(line2.startPointX + line2.endPointX)) * 0.5;
		midPoint.y = (w1*(line1.startPointY + line1.endPointY) + w2*(line2.startPointY + line2.endPointY)) * 0.5;

		// Define the orientation of the merged line as the weighted sum of the orientations of the given segments
		// adjust line directions
		float radian;
		if (line1.pt.x * line2.pt.x + line1.pt.y * line2.pt.y < 0) {
			radian = w1 * atan2(line1.pt.y, line1.pt.x) + w2 * atan2(-line2.pt.y, -line2.pt.x);
		}
		else {
			radian = w1 * atan2(line1.pt.y, line1.pt.x) + w2 * atan2(line2.pt.y, line2.pt.x);
		}
		Point2f lineDir(cos(radian), sin(radian));

		vector<float> p;
		p.push_back((line1.startPointX - midPoint.x) * lineDir.x + (line1.startPointY - midPoint.y) * lineDir.y);
		p.push_back((line1.endPointX - midPoint.x) * lineDir.x + (line1.endPointY - midPoint.y) * lineDir.y);
		p.push_back((line2.startPointX - midPoint.x) * lineDir.x + (line2.startPointY - midPoint.y) * lineDir.y);
		p.push_back((line2.endPointX - midPoint.x) * lineDir.x + (line2.endPointY - midPoint.y) * lineDir.y);

		vector<int> idx;
		sortIdx(p, idx, SORT_ASCENDING);
		vector<Point2f> points = { line1.getStartPoint(), line1.getEndPoint(), line2.getStartPoint(), line2.getEndPoint() };

		newLine.startPointX = points[idx[0]].x;
		newLine.startPointY = points[idx[0]].y;
		newLine.endPointX = points[idx[3]].x;
		newLine.endPointY = points[idx[3]].y;

		return (p[idx[3]] - p[idx[0]]) - (fabs(p[0] - p[1]) + fabs(p[2] - p[3]));
	}

	void mergeLines(vector<KeyLine> &lines, float lineGapThreshold, float lineDistThreshold) {
		bool updated = true;
		vector<bool> merged(lines.size(), false);   // which line will be deleted.

		while (updated) {
			updated = false;

			for (unsigned int ind1 = 0; ind1 < lines.size() - 1; ind1++) {
				if (merged[ind1]) {
					continue;
				}

				for (unsigned int ind2 = ind1 + 1; ind2 < lines.size(); ind2++) {
					if (merged[ind2]) {
						continue;
					}

					KeyLine newLine;
					float lineGap = mergeLines(lines[ind1], lines[ind2], newLine);

					// If gap between two lines is too large, do not merge the lines.
					if (lineGap > lineGapThreshold) {
						continue;
					}

					// Compute line coefficient of merged line
					float newLineCoefficient[3];
					vec_cross(newLine.startPointX, newLine.startPointY, 1.f,
						newLine.endPointX, newLine.endPointY, 1.f,
						newLineCoefficient[0], newLineCoefficient[1], newLineCoefficient[2]);

					float dist = (LineDistance(lines[ind1].getStartPoint(), newLineCoefficient) +
						LineDistance(lines[ind1].getEndPoint(), newLineCoefficient) +
						LineDistance(lines[ind2].getStartPoint(), newLineCoefficient) +
						LineDistance(lines[ind2].getEndPoint(), newLineCoefficient)) * 0.25f;

					if (dist < lineDistThreshold) {
						lines[ind1] = newLine;
						merged[ind2] = true;
					}
				}
			}
		}

		int lineIndex = 0;
		for (auto iter = lines.begin(); iter != lines.end(); lineIndex++) {
			if (merged[lineIndex]) {
				iter = lines.erase(iter);
			}
			else {
				iter++;
			}
		}
	}

	vector<PointPair> detectLines(const Mat &image, float lineLengthThreshold) {
		vector<PointPair> outputLines;
		vector<KeyLine> lines;
		Ptr<line_descriptor::LSDDetector> lsddetector = line_descriptor::LSDDetector::createLSDDetector();
		Mat grayImage;

		cvtColor(image, grayImage, COLOR_BGR2GRAY);

		lsddetector->detect(grayImage, lines, 2, 1, generateEmptySpaceMask(grayImage));

		// Remove a line if its length is smaller than given threshold
		for (auto iter = lines.begin(); iter != lines.end();) {
			if (iter->lineLength < lineLengthThreshold) {
				if (false) {
					if (iter->lineLength > lineLengthThreshold / 2) {
						Point2f centerPoint((iter->startPointX + iter->endPointX) / 2.f,
							(iter->startPointY + iter->endPointY) / 2.f);

						iter->startPointX = iter->startPointX + (iter->startPointX - iter->endPointX) / iter->lineLength * (lineLengthThreshold / 2.f);
						iter->endPointX = iter->endPointX + (iter->endPointX - iter->startPointX) / iter->lineLength * (lineLengthThreshold / 2.f);

						iter->startPointY = iter->startPointY + (iter->startPointY - iter->endPointY) / iter->lineLength * (lineLengthThreshold / 2.f);
						iter->endPointY = iter->endPointY + (iter->endPointY - iter->startPointY) / iter->lineLength * (lineLengthThreshold / 2.f);

						iter++;
					}
					else {
						iter = lines.erase(iter);
					}
				}
				else {
					iter = lines.erase(iter);
				}
			}
			else {
				iter++;
			}
		}

		mergeLines(lines, 30.f);

		for (const KeyLine &l : lines) {
			outputLines.push_back({ l.getStartPoint(), l.getEndPoint() });
		}

		return outputLines;
	}

	vector<Point3f> generateVPModels(const vector<PointPair> &lines, int numOfModels, uint64 seed) {
		RNG rng(0xFFFFFFFF);
		unordered_set<pair<int, int>, IntPairHash> checker;
		int numOfLines = int(lines.size());
		vector<Point3f> models;

		for (int i = 0; i < numOfModels; i++) {
			int idx1 = rng.uniform(0, numOfLines);
			int idx2 = idx1;
			while (idx1 == idx2) {
				idx2 = rng.uniform(0, numOfLines);

				if (checker.find({ idx1, idx2 }) == checker.end() &&
					checker.find({ idx2, idx1 }) == checker.end()) {
					checker.insert({ idx1, idx2 });
					break;
				}
			}

			models.push_back(computeIntersection(lines[idx1], lines[idx2]));
		}

		return models;
	}

	LineClusters generateClusters(const vector<PointPair>& lines,
		const vector<Point3f>& models, const float *prefMat, const vector<pair<int, int>> &T) {
		// Counts
		int numOfClusters = 0;
		size_t numOfMatches = lines.size();
		size_t numOfModels = models.size();

		// Construct clusters
		vector<int> C;	// matches point index to row of preference matrix
		for (int i = 0; i < numOfMatches; i++) {
			C.push_back(i);
		}

		for (int i = 0; i < T.size(); i++) {
			int a = T[i].first;
			int b = T[i].second;
			for (int j = 0; j < C.size(); j++) {
				if (C[j] == a || C[j] == b) {
					C[j] = numOfMatches + i;
				}
			}
		}

		unordered_map<int, int> indexer;	// matches row of preference matrix to the actual cluster index
		for (int i = 0; i < numOfMatches; i++) {
			if (indexer.find(C[i]) == indexer.end()) {
				indexer[C[i]] = numOfClusters++;
			}
		}

		// Form clusters
		vector<vector<Line>> arrayOfLines(numOfClusters);
		for (int i = 0; i < numOfMatches; i++) {
			arrayOfLines[indexer[C[i]]].emplace_back(lines[i]);
		}

		// Analize the preference matrix
		vector<LineCluster> lineClusters;
		for (const auto &elem : arrayOfLines) {
			if (arrayOfLines.size() > 2) {
				lineClusters.push_back(LineCluster(elem));
			}
		}

		return LineClusters(lineClusters, lines);
	}

	LineClusters lineClustering(Mat image, int numModels, int minCardinarity, float distThreshold,
		float lengthThreshold, bool allowDuplicates) {
		vector<PointPair> lines;
		vector<PointPair> longLines;
		vector<Point3f> models;
		Mat output;
		double t = (double)getTickCount();
		float longLineThr = sqrt(image.cols * image.cols + image.rows * image.rows) / 30.f;
		float shortLineThr = sqrt(image.cols * image.cols + image.rows * image.rows) / 50.f;

		if (lengthThreshold < 0) {
			shortLineThr = sqrt(image.cols * image.cols + image.rows * image.rows) / 50.f;
			longLineThr = sqrt(image.cols * image.cols + image.rows * image.rows) / 30.f;
		}
		else {
			shortLineThr = lengthThreshold;
			longLineThr = lengthThreshold * 2.f;
		}

		cout << "Detect lines..." << endl;
		lines = detectLines(image, lengthThreshold);

		for (const PointPair& line : lines) {
			float length = sqrt(pow((line.first.x - line.second.x), 2) + pow((line.first.y - line.second.y), 2));
			if (length >= longLineThr) {
				longLines.push_back(line);
			}
		}

		cout << "Generate random models of vanishing points..." << endl;
		models = generateVPModels(longLines, numModels);

		// Copy lines to linear memory
		float *arrLines = new float[sizeof(float) * lines.size() * 4];
		for (size_t idx_org = 0, idx_tar = 0; idx_org < lines.size(); idx_org++) {
			arrLines[idx_tar++] = lines[idx_org].first.x;
			arrLines[idx_tar++] = lines[idx_org].first.y;
			arrLines[idx_tar++] = lines[idx_org].second.x;
			arrLines[idx_tar++] = lines[idx_org].second.y;
		}

		// Copy homography data to temporary array
		float *arrModels = new float[sizeof(float) * models.size() * 3];
		for (int idx_org = 0, idx_tar = 0; idx_org < models.size(); idx_org++) {
			arrModels[idx_tar++] = models[idx_org].x;
			arrModels[idx_tar++] = models[idx_org].y;
			arrModels[idx_tar++] = models[idx_org].z;
		}

		vector<pair<int, int>> T;
		float *prefMatArray = new float[lines.size() * models.size() * 2];

		cout << "Cluster lines with T-linkage..." << endl;
		TLinkageGPU(arrLines, int(lines.size()), arrModels, int(models.size()), distThreshold, prefMatArray, T);

		LineClusters lineClusters = generateClusters(lines, models, prefMatArray, T);
		
		cout << "Sort the clustered lines by their cardinality..." << endl;
		lineClusters.sort();

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "*** Line clustering took " << t << " seconds. " << endl;

		// Remove temporary data
		delete[] arrLines;
		delete[] arrModels;
		delete[] prefMatArray;

		return lineClusters;
	}

	inline double LineDistance(const Point2f &point, const float *line) {
		float absquare = sqrt(line[0] * line[0] + line[1] * line[1]);
		return fabs(line[0] * point.x + line[1] * point.y + line[2]) / absquare;
	}

	inline Point3f computeIntersection(const PointPair &line1, const PointPair &line2) {
		Point3f intersection;

		float xs0 = line1.first.x;
		float ys0 = line1.first.y;
		float xe0 = line1.second.x;
		float ye0 = line1.second.y;
		float xs1 = line2.first.x;
		float ys1 = line2.first.y;
		float xe1 = line2.second.x;
		float ye1 = line2.second.y;

		float l0[3], l1[3], v[3];
		vec_cross(xs0, ys0, 1,
			xe0, ye0, 1,
			l0[0], l0[1], l0[2]);
		vec_cross(xs1, ys1, 1,
			xe1, ye1, 1,
			l1[0], l1[1], l1[2]);
		vec_cross(l0[0], l0[1], l0[2],
			l1[0], l1[1], l1[2],
			v[0], v[1], v[2]);
		vec_norm(v[0], v[1], v[2]);

		intersection.x = v[0];
		intersection.y = v[1];
		intersection.z = v[2];

		return intersection;
	}

};
