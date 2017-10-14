#include "VPDetection.h"
#include "Line.h"
#include "TLinkageGPU.h"
#include "CustomHash.h"

#include <unordered_set>
#include <unordered_map>

#include <ceres\ceres.h>

#include "VDResidual.h"

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
		float lengthThreshold, bool allowDuplicates, bool draw_output) {
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

		if (draw_output) {
			cv::imwrite("line_clusters_org.png", drawLineClusters(image, lineClusters, 2, true));
		}

		// Remove temporary data
		delete[] arrLines;
		delete[] arrModels;
		delete[] prefMatArray;

		return lineClusters;
	}

	//----------------------------------
	// Cluster optimization
	//----------------------------------

	Mat detectVP(const LineClusters &lineClusters, vector<int> &selectedIndices, Mat &K, bool refineK,
		float threshold, bool extendInliers, const Mat &debug_image) {
		Mat VD;

		selectedIndices = findOrthogonalClusters(lineClusters, K, threshold, VD);
		LineClusters selectedClusters = lineClusters.subCluster(selectedIndices);

		//cv::imwrite("Results\\lineClustersSelected.png", drawLineClusters(image, selectedClusters, 2, true));

		for (LineCluster &lineCluster : selectedClusters) {
			lineCluster.resetVanishingPoint(threshold);
		}

		if (extendInliers) {
			selectedClusters.collectInliers(threshold);
		}

		//cv::imwrite("Results\\lineClustersRecomputed.png", drawLineClusters(image, selectedClusters, 2, true));

		// Orthogonalize the vanishing directions
		cout << "VD: \n" << VD << endl;
		HouseHolderQR(VD.clone(), VD, Mat());

		cout << "=== Initial matrices ===" << endl;
		cout << "K: \n" << K << endl;
		cout << "VD: \n" << VD << endl;
		cout << "========================" << endl;

		// TODO: check if refine calibration has a problem
		VD = refineCalibration(selectedClusters, VD, K, threshold, refineK);

		cout << "====== Optimized =======" << endl;
		cout << "K: \n" << K << endl;
		cout << "VD: \n" << VD << endl;
		cout << "========================" << endl;

		return VD;
	}

	vector<int> findOrthogonalClusters(const LineClusters &lineClusters, const Mat &K, float threshold, Mat &VD) {
		float perpCosThr = cos(75.0 * CV_PI / 180.0);
		float paraCosThr = cos(30.0 * CV_PI / 180.0);
		RNG rng(time(NULL));
		const int numOfClusters = (int)lineClusters.size();
		vector<Vec3f> VanishingDirections;
		vector<float> gravityConsistency;
		vector<int> gravityConsistencyIdx;
		Vec3f yDir(0.f, 1.f, 0.f);
		int maxNumOfInliers = 0;
		Mat Kinv = K.inv();
		vector<int> selectedIndices(3, -1);

		VD = Mat::zeros(3, 3, CV_32F);

		// Compute initial vanishing directions
		for (int ci = 0; ci < numOfClusters; ci++) {
			Point2f VP = lineClusters[ci].getVanishingPoint();
			Vec3f VD;

			VD[0] = Kinv.at<float>(0, 0) * VP.x + Kinv.at<float>(0, 1) * VP.y + Kinv.at<float>(0, 2) * 1.f;
			VD[1] = Kinv.at<float>(1, 0) * VP.x + Kinv.at<float>(1, 1) * VP.y + Kinv.at<float>(1, 2) * 1.f;
			VD[2] = Kinv.at<float>(2, 0) * VP.x + Kinv.at<float>(2, 1) * VP.y + Kinv.at<float>(2, 2) * 1.f;

			VD = normalize(VD);

			VanishingDirections.push_back(VD);
			gravityConsistency.push_back(abs(VD.dot(yDir)));
		}

		sortIdx(gravityConsistency, gravityConsistencyIdx, SORT_DESCENDING);

		for (int vdi = 0; vdi < numOfClusters; vdi++) {
			int y_idx = gravityConsistencyIdx[vdi];
			int numOfLines1 = lineClusters[y_idx].size();

			if (gravityConsistency[y_idx] > paraCosThr) {
				Vec3f &VD_y = VanishingDirections[y_idx];

				for (int vdj = 0; vdj < numOfClusters; vdj++) {
					int numOfLines2 = lineClusters[vdj].size();

					// Two selected vanishing directions must be as perpendicular as possible.
					if (vdj != y_idx && abs(VD_y.dot(VanishingDirections[vdj])) < perpCosThr) {
						for (int vdk = 0; vdk < numOfClusters; vdk++) {
							if (vdk != y_idx && vdk != vdj) {
								// Three selected vanishing directions must be as perpendicular as possible.
								if (abs(VD_y.dot(VanishingDirections[vdk])) < perpCosThr &&
									abs(VanishingDirections[vdj].dot(VanishingDirections[vdk])) < perpCosThr) {

									// Compute the number of inliers that are aligned with current setting
									int numOfInliers = numOfLines1 + numOfLines2 + lineClusters[vdk].size();

									if (numOfInliers > maxNumOfInliers) {
										maxNumOfInliers = numOfInliers;
										selectedIndices[0] = y_idx;
										selectedIndices[1] = vdj;
										selectedIndices[2] = vdk;
										Mat(VD_y).copyTo(VD.col(0));
										Mat(VanishingDirections[vdj]).copyTo(VD.col(1));
										Mat(VanishingDirections[vdk]).copyTo(VD.col(2));
									}
								}
								else {
									// Compute vanishing direction, which is perpendicular to the selected VDs.
									Vec3f vanishingDirection = VD_y.cross(VanishingDirections[vdj]);
									float normalizer = K.at<float>(2, 0) * vanishingDirection[0] +
										K.at<float>(2, 1) * vanishingDirection[1] +
										K.at<float>(2, 2) * vanishingDirection[2];
									Point2f vanishingPoint((K.at<float>(0, 0) * vanishingDirection[0] +
										K.at<float>(0, 1) * vanishingDirection[1] +
										K.at<float>(0, 2) * vanishingDirection[2]) / normalizer,
										(K.at<float>(1, 0) * vanishingDirection[0] +
											K.at<float>(1, 1) * vanishingDirection[1] +
											K.at<float>(1, 2) * vanishingDirection[2]) / normalizer);

									// Compute cardinality of computed vanishing point
									int numOfInliers = numOfLines1 + numOfLines2 +
										lineClusters.computeCardinality(vanishingPoint, threshold);

									if (numOfInliers > maxNumOfInliers) {
										maxNumOfInliers = numOfInliers;
										selectedIndices[0] = y_idx;
										selectedIndices[1] = vdj;
										selectedIndices[2] = -1;	// virtual cluster, might need to be computed
										Mat(VD_y).copyTo(VD.col(0));
										Mat(VanishingDirections[vdj]).copyTo(VD.col(1));
										Mat(vanishingDirection).copyTo(VD.col(2));
									}
								}
							}	// end_if (vdk != y_idx && vdk != vdj)
						}	// end_for (int vdk = 0; vdk < numOfClusters; vdk++)
					}	// end_if (vdj != y_idx && VD_y.dot(VanishingDirections[vdj]) < diffThr)
				}	// end_for (int vdj = 0; vdj < numOfClusters; vdj++)
			}	// end_if (gravityConsistency[y_idx] > equalThr)
			else {
				break;
			}
		}

		if (maxNumOfInliers == 0) {
			cout << "Failed to find orthogonal clusters. Find orthogonal clusters based on their cardinality" << endl;

			vector<int> numInliers;
			vector<int> sortedIndices;

			for (int i = 0; i < numOfClusters; i++) {
				numInliers.push_back((int)lineClusters.size());
			}

			sortIdx(numInliers, sortedIndices, SORT_DESCENDING);
			selectedIndices[0] = sortedIndices[0];
			selectedIndices[1] = sortedIndices[1];
			Vec3f VDTest = VanishingDirections[selectedIndices[0]].cross(VanishingDirections[selectedIndices[1]]);

			// Find the last cluster
			int bestIndex = -1;
			float maxCosine = 0., cosineValue;
			for (int i = 0; i < numOfClusters; i++) {
				if (i == selectedIndices[0] || i == selectedIndices[1]) {
					continue;
				}

				cosineValue = abs(VDTest.dot(VanishingDirections[i]));
				if (cosineValue > maxCosine) {
					maxCosine = cosineValue;
					bestIndex = i;
				}
			}

			selectedIndices[2] = bestIndex;

			Mat(VanishingDirections[selectedIndices[0]]).copyTo(VD.col(0));
			Mat(VanishingDirections[selectedIndices[1]]).copyTo(VD.col(1));
			Mat(VanishingDirections[selectedIndices[2]]).copyTo(VD.col(2));
		}
		else if (selectedIndices[2] == -1) {
			cout << "Selected from computed third VD" << endl;
			size_t maxInliers = 0;
			int bestIndex = -1;
			for (int i = 0; i < lineClusters.size(); i++) {
				if (i == selectedIndices[0] || i == selectedIndices[1]) {
					continue;
				}

				if (lineClusters.size() > maxInliers) {
					maxInliers = lineClusters.size();
					bestIndex = i;
				}
			}
			selectedIndices[2] = bestIndex;
		}

		return selectedIndices;
	}

	Mat refineCalibration(const LineClusters &lineClusters, const Mat &VD, Mat &K, float threshold, bool refineK) {
		Mat rotation;
		Mat rotationComps;
		ceres::Problem problem;

		Rodrigues(VD, rotationComps);

		double x[3] = { rotationComps.at<float>(0), rotationComps.at<float>(1), rotationComps.at<float>(2) };
		K.convertTo(K, CV_64F);
		double *focalLength = K.ptr<double>(0, 0);

		for (int ci = 0; ci < 3; ci++) {
			for (const Line& l : lineClusters[ci]) {
				const Mat &lineVector = l.LineVector;

				problem.AddResidualBlock(
					VDResidual::Create(lineVector.at<float>(0), lineVector.at<float>(1), lineVector.at<float>(2),
						K.at<double>(0, 2), K.at<double>(1, 2), ci),
					new ceres::CauchyLoss(0.5), x, focalLength);
			}
		}

		ceres::Solver::Options options;
		ceres::Solver::Summary summary;

		// Preventing from the focal length would have weild value
		problem.SetParameterLowerBound(focalLength, 0, *focalLength * 0.8);
		problem.SetParameterUpperBound(focalLength, 0, *focalLength * 1.2);

		options.linear_solver_type = ceres::DENSE_SCHUR; // Either DENSE_SCHUR or DENSE_QR
		options.minimizer_progress_to_stdout = false;
		options.use_explicit_schur_complement = true;
		ceres::Solve(options, &problem, &summary);

		//cout << summary.FullReport() << endl;

		rotationComps.at<float>(0) = x[0];
		rotationComps.at<float>(1) = x[1];
		rotationComps.at<float>(2) = x[2];

		Rodrigues(rotationComps, rotation);
		if (refineK) {
			K.at<float>(0, 0) = *focalLength;
			K.at<float>(1, 1) = *focalLength;
		}
		K.convertTo(K, CV_32F);
		K.at<float>(1, 1) = K.at<float>(0, 0);

		if (K.at<float>(0, 0) < 0.f) {
			VD.row(0) *= -1.f;
			VD.row(1) *= -1.f;
			K.at<float>(0, 0) *= -1.f;
			K.at<float>(1, 1) *= -1.f;
		}

		return rotation;
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

	void HouseHolderQR(const cv::Mat &A, cv::Mat &Q, cv::Mat &R)
	{
		assert(A.channels() == 1);
		assert(A.rows >= A.cols);
		auto sign = [](float value) { return value >= 0 ? 1 : -1; };
		const auto totalRows = A.rows;
		const auto totalCols = A.cols;
		R = A.clone();
		Q = cv::Mat::eye(totalRows, totalRows, A.type());
		for (int col = 0; col < A.cols; ++col)
		{
			cv::Mat matAROI = cv::Mat(R, cv::Range(col, totalRows), cv::Range(col, totalCols));
			cv::Mat y = matAROI.col(0);
			auto yNorm = norm(y);
			cv::Mat e1 = cv::Mat::eye(y.rows, 1, A.type());
			cv::Mat w = y + sign(y.at<float>(0, 0)) *  yNorm * e1;
			cv::Mat v = w / norm(w);
			cv::Mat vT; cv::transpose(v, vT);
			cv::Mat I = cv::Mat::eye(matAROI.rows, matAROI.rows, A.type());
			cv::Mat I_2VVT = I - 2 * v * vT;
			cv::Mat matH = cv::Mat::eye(totalRows, totalRows, A.type());
			cv::Mat matHROI = cv::Mat(matH, cv::Range(col, totalRows), cv::Range(col, totalRows));
			I_2VVT.copyTo(matHROI);
			R = matH * R;
			Q = Q * matH;
		}
	}

	Mat drawLineClusters(const Mat &image, const LineClusters &lineClusters, int lineWidth, bool convertGray) {
		Mat output;
		RNG rng(0xFFFFFFFF);

		if (convertGray) {
			if (image.channels() == 3) {
				cvtColor(image, output, CV_BGR2GRAY);
				cvtColor(output, output, CV_GRAY2BGR);
			}
			else if (image.channels() == 1) {
				cvtColor(image, output, CV_GRAY2BGR);
			}
		}
		else {
			output = image.clone();
		}

		for (const auto &cluster : lineClusters) {
			int icolor = (unsigned)rng;
			Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);

			for (const auto &l : cluster.Lines) {
				circle(output, l.StartPoint, 2, CV_RGB(255, 255, 255));
				circle(output, l.EndPoint, 2, CV_RGB(255, 255, 255));
				line(output, l.StartPoint, l.EndPoint, color, lineWidth);
			}
		}

		return output;
	}

	Mat drawVanishingDirections(const Mat &image, const Mat &VDs, const Mat &K) {
		Mat result = image.clone();
		float length = (float)sqrt(image.cols * image.cols + image.rows * image.rows) / 50.f;
		float focal = K.at<float>(0, 0);
		Point s(K.at<float>(0, 2), K.at<float>(1, 2));
		RNG rng(0xFFFFFFFF);

		for (int i = 0; i < VDs.cols; i++) {
			VDs.col(i) * length;

			Mat S = (Mat_<float>(3, 1) << 0, 0, focal * length);
			Mat E = (Mat_<float>(3, 1) <<
				S.at<float>(0) + focal * VDs.at<float>(0, i),
				S.at<float>(1) + focal * VDs.at<float>(1, i),
				S.at<float>(2) + focal * VDs.at<float>(2, i));
			Mat e = K * E;

			int icolor = (unsigned)rng;
			Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
			line(result, s, Point(e.at<float>(0) / e.at<float>(2), e.at<float>(1) / e.at<float>(2)), color, 3);
		}

		return result;
	}
};
