////
//  LineCluster.cpp
//  Error Analysis
//
//  Created by Alexxxxx on 2024/7/14.
//
#include "LineCluster.h"
#include "BasicMath.h"
#include "Parameters.h"

// Compute cross product of two vectors
cv::Vec3d crossProduct(const cv::Vec3d& v1, const cv::Vec3d& v2) {
	return v1.cross(v2);
}

// Compute the distance between two points
double getDist(const cv::Vec3d& point1, const cv::Vec3d& point2) {
	return cv::norm(point1 - point2);
}

// Compute the line equation parameters
void LineEquation(const cv::Vec3d& V1, const cv::Vec3d& V2, cv::Vec3d& abc) {
	double x1 = V1[0], y1 = V1[1], z1 = V1[2];
	double x2 = V2[0], y2 = V2[1], z2 = V2[2];

	if (x1 == x2 && y1 == y2 && z1 == z2) {
		abc = cv::Vec3d(0, 0, 0);
	}
	else {
		double denom = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
		abc = cv::Vec3d((x2 - x1) / denom, (y2 - y1) / denom, (z2 - z1) / denom);
	}
}

// Output to OBJ file
void outObj(std::string filePath, cv::Mat lines) {
	// Output OBJ file
	std::ofstream outfile;
	outfile.open(filePath);

	// Write vertices
	for (int i = 0; i < lines.rows; i++) {
		outfile << "v " << lines.at<float>(i, 0) << " " << lines.at<float>(i, 1) << " " << lines.at<float>(i, 2) << "\n";
		outfile << "v " << lines.at<float>(i, 3) << " " << lines.at<float>(i, 4) << " " << lines.at<float>(i, 5) << "\n";
	}

	// Write line segments
	for (int i = 0; i < lines.rows; i++) {
		outfile << "l " << i * 2 + 1 << " " << i * 2 + 2 << "\n";
	}

	outfile.close();
}

// Compute the distance between two lines
void line2lineDist(const cv::Mat& l3d, const cv::Mat& cen, const cv::Mat& cen2, cv::Vec3d& O1, cv::Vec3d& O2, double& dist) {
	// Extract two points on each line
	cv::Vec3d P1 = l3d.row(0).colRange(0, 3);
	cv::Vec3d P2 = l3d.row(0).colRange(3, 6);
	cv::Vec3d Q1 = cen.row(0);
	cv::Vec3d Q2 = cen2.row(0);

	// Compute line equation parameters
	cv::Vec3d abc1, abc2;
	LineEquation(P1, P2, abc1);
	LineEquation(Q1, Q2, abc2);

	// Line vectors
	cv::Vec3d v1 = P2 - P1;
	cv::Vec3d v2 = Q2 - Q1;

	// Compute the plane equation for the line_1 and perpendicular line
	cv::Vec3d n1 = crossProduct(crossProduct(v1, v2), v1);
	double A1 = n1[0], B1 = n1[1], C1 = n1[2];
	double D1 = -A1 * P1[0] - B1 * P1[1] - C1 * P1[2];

	// Compute the plane equation for the line_2 and perpendicular line
	cv::Vec3d n2 = crossProduct(crossProduct(v1, v2), v2);
	double A2 = n2[0], B2 = n2[1], C2 = n2[2];
	double D2 = -A2 * Q1[0] - B2 * Q1[1] - C2 * Q1[2];

	// Find the closest point O1 on line_1 to line_2
	double t1 = -(A2 * P1[0] + B2 * P1[1] + C2 * P1[2] + D2) / (A2 * abc1[0] + B2 * abc1[1] + C2 * abc1[2]);
	O1 = P1 + t1 * abc1;

	// Find the closest point O2 on line_2 to line_1
	double t2 = -(A1 * Q1[0] + B1 * Q1[1] + C1 * Q1[2] + D1) / (A1 * abc2[0] + B1 * abc2[1] + C1 * abc2[2]);
	O2 = Q1 + t2 * abc2;

	// Compute the distance between the two points
	dist = getDist(O1, O2);
}

// Calculate the angle between two vectors
double calculateAngle(const cv::Vec3d& vec1, const cv::Vec3d& vec2) {
	// Compute dot product
	double dotProduct = vec1.dot(vec2);

	// Compute the norms of the vectors
	double normVec1 = cv::norm(vec1);
	double normVec2 = cv::norm(vec2);

	// Compute the cosine of the angle
	double cosAngle = dotProduct / (normVec1 * normVec2);

	// Compute and return the angle
	double angle = acos(cosAngle);

	return angle;
}




// Function to split a string by a delimiter
std::vector<std::string> split(const std::string& str, char delimiter) {
	std::vector<std::string> tokens;
	std::string token;
	for (char ch : str) {
		if (ch == delimiter) {
			if (!token.empty()) {
				tokens.push_back(token);
				token.clear();
			}
		}
		else {
			token += ch;
		}
	}
	if (!token.empty()) {
		tokens.push_back(token);
	}
	return tokens;
}

// Calculate the distance from a point to a line
cv::Mat pt2LineDis(cv::Mat P, cv::Mat A, cv::Mat B) {
	cv::Mat AB = B - A;
	cv::Mat AP = P - A;

	double dotAPAB = AP.dot(AB);
	double dotABAB = AB.dot(AB);

	cv::Mat projP = A + (dotAPAB / dotABAB) * AB;
	cv::Mat err3 = projP - P;

	return err3;
}

// Calculate the multivariate normal cumulative distribution function
double mvncdf(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
	// Compute the Cholesky decomposition of the covariance matrix
	Eigen::LLT<Eigen::MatrixXd> llt(cov);
	Eigen::MatrixXd L = llt.matrixL();

	// Standardize the input vector
	Eigen::VectorXd z = L.triangularView<Eigen::Lower>().solve(x - mean);

	// Use the one-dimensional normal distribution to calculate the CDF
	boost::math::normal_distribution<> standard_normal(0, 1);
	double p = 1.0;
	for (int i = 0; i < z.size(); ++i) {
		p *= boost::math::cdf(standard_normal, z(i));
	}

	return p;
}



// Triangulate points in front of the camera
void triangulate(cv::Mat left_cam, cv::Mat right_cam, cv::Mat left_pts, cv::Mat right_pts, cv::Mat& space_pts) {
	cv::Mat re_space_pts;
	cv::triangulatePoints(left_cam, right_cam, left_pts.t(), right_pts.t(), re_space_pts);
	space_pts = cv::Mat(left_pts.rows, 3, CV_32FC1);

	space_pts.col(0) = (re_space_pts.row(0) / re_space_pts.row(3)).t();
	space_pts.col(1) = (re_space_pts.row(1) / re_space_pts.row(3)).t();
	space_pts.col(2) = (re_space_pts.row(2) / re_space_pts.row(3)).t();
}

// Update the mean and standard deviation
void getMeanStd(double mean, double stdv, double n, double new_d, double& new_mean, double& new_std) {
	new_mean = (n * mean + new_d) / (n + 1.0);
	new_std = std::sqrt(((n - 1.0) * stdv * stdv + n * mean * mean + new_d * new_d - (n + 1.0) * new_mean * new_mean) / n);
}

// Function to calculate the median
double median(std::vector<double>& vec) {
	std::sort(vec.begin(), vec.end());
	size_t size = vec.size();
	if (size % 2 == 0) {
		return (vec[size / 2 - 1] + vec[size / 2]) / 2.0;
	}
	else {
		return vec[size / 2];
	}
}

// medianStds function
double medianStds(const cv::Mat& stdArr) {
	// Reshape the matrix to N x 3
	cv::Mat stdA = stdArr.reshape(1, stdArr.rows * stdArr.cols);

	// Convert to std::vector and remove rows containing zero
	std::vector<cv::Vec3f> vecA;
	for (int i = 0; i < stdA.rows; ++i) {
		cv::Vec3f row(stdA.at<float>(i, 0), stdA.at<float>(i, 1), stdA.at<float>(i, 2));
		if (row[0] != 0 && row[1] != 0 && row[2] != 0) {
			vecA.push_back(row);
		}
	}

	// If filtered vector is empty, return 0
	if (vecA.empty()) {
		return 0;
	}

	// Store each column's values
	std::vector<double> col1, col2, col3;
	for (const auto& row : vecA) {
		col1.push_back(row[0]);
		col2.push_back(row[1]);
		col3.push_back(row[2]);
	}

	// Calculate the median of each column
	double stx = median(col1);
	double sty = median(col2);
	double stz = median(col3);

	// Calculate the distance
	double distmean = std::sqrt(std::pow(2 * stx, 2) + std::pow(2 * sty, 2) + std::pow(2 * stz, 2));

	return distmean;
}


//直接使用内存中的multiPoints
void normalBuild(std::vector<cv::Mat> multiPoints, SFM_INFO& sfmInfo, IMG_INFO& imgInfo, ARR_INFO& arrInfo) {
	std::cout << "Start Norm Build:..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	cv::Mat imageConnect = cv::Mat::zeros(imgInfo.cameras.size(), imgInfo.cameras.size(), CV_16SC1);
	for (int i = 0; i < sfmInfo.pairID.rows; i++) {
		int row = sfmInfo.pairID.at<float>(i, 0) - 1;
		int col = sfmInfo.pairID.at<float>(i, 1) - 1;
		if (row == -1 || col == -1)	continue;
		imageConnect.at<short>(row, col) = i + 1;
		imageConnect.at<short>(col, row) = i + 1;
	}

	arrInfo.meanArr = cv::Mat::zeros(sfmInfo.pairID.rows, imgInfo.cameras.size(), CV_32FC3);
	arrInfo.stdArr = cv::Mat::zeros(sfmInfo.pairID.rows, imgInfo.cameras.size(), CV_32FC3);
	cv::Mat counterArr = cv::Mat::zeros(sfmInfo.pairID.rows, imgInfo.cameras.size(), CV_32FC1);

	//std::vector<std::string> allLinesFromMP;
	//int allPtsCount = readLinesFromFile(outfolder + "\\multiPoints.txt", allLinesFromMP);
	for (int i = 0; i < multiPoints.size(); i++) {
		//std::vector<std::string> sLine = split(allLinesFromMP[i], ',');
		//if (sLine.size() < 3) continue;
		if (multiPoints[i].rows < 3) continue;
		cv::Mat subpoints = multiPoints[i];
		//for (int j = 0; j < sLine.size(); j++) {
		//	std::stringstream ss(sLine[j]);
		//	std::string tmp_s;
		//	ss >> tmp_s; subpoints.at<float>(j, 0) = atof(tmp_s.c_str());
		//	ss >> tmp_s; subpoints.at<float>(j, 1) = atof(tmp_s.c_str());
		//	ss >> tmp_s; subpoints.at<float>(j, 2) = atof(tmp_s.c_str());
		//}

		for (int indx_i = 0; indx_i < subpoints.rows; indx_i++) {
			int idi = subpoints.at<float>(indx_i, 0);
			for (int indx_j = indx_i + 1; indx_j < subpoints.rows; indx_j++) {
				int idj = subpoints.at<float>(indx_j, 0);
				int pairID = imageConnect.at<short>(idi, idj) - 1;
				if (pairID == -1) continue;

				auto cam1 = imgInfo.cameras[idi];
				auto cam2 = imgInfo.cameras[idj];

				cv::Mat re_space_pts;
				triangulate(cam1, cam2, subpoints.row(indx_i).colRange(1, 3), subpoints.row(indx_j).colRange(1, 3), re_space_pts);

				for (int indx_k = 0; indx_k < subpoints.rows; indx_k++) {
					int idk = subpoints.at<float>(indx_k, 0);
					if (idi == idk || idj == idk) continue;
					auto cam3 = imgInfo.cameras[idk];
					auto cen3 = imgInfo.centers[idk];

					cv::Mat pt3 = (cv::Mat_<float>(3, 1) << subpoints.at<float>(indx_k, 1), subpoints.at<float>(indx_k, 2), 1.0);
					cv::Mat s_cam3 = cam3.rowRange(0, 3).colRange(0, 3);
					cv::Mat ray = s_cam3.inv() * pt3;

					cv::Mat pt2l = pt2LineDis(re_space_pts.t(), cen3.t(), cen3.t() + 2 * ray);

					counterArr.at<float>(pairID, idk) = counterArr.at<float>(pairID, idk) + 1;

					double m0, s0;
					getMeanStd(arrInfo.meanArr.at<cv::Vec3f>(pairID, idk)[0], arrInfo.stdArr.at<cv::Vec3f>(pairID, idk)[0],
						counterArr.at<float>(pairID, idk), pt2l.at<float>(0), m0, s0);
					arrInfo.meanArr.at<cv::Vec3f>(pairID, idk)[0] = m0;
					arrInfo.stdArr.at<cv::Vec3f>(pairID, idk)[0] = s0;

					double m1, s1;
					getMeanStd(arrInfo.meanArr.at<cv::Vec3f>(pairID, idk)[1], arrInfo.stdArr.at<cv::Vec3f>(pairID, idk)[1],
						counterArr.at<float>(pairID, idk), pt2l.at<float>(1), m1, s1);
					arrInfo.meanArr.at<cv::Vec3f>(pairID, idk)[1] = m1;
					arrInfo.stdArr.at<cv::Vec3f>(pairID, idk)[1] = s1;

					double m2, s2;
					getMeanStd(arrInfo.meanArr.at<cv::Vec3f>(pairID, idk)[2], arrInfo.stdArr.at<cv::Vec3f>(pairID, idk)[2],
						counterArr.at<float>(pairID, idk), pt2l.at<float>(2), m2, s2);
					arrInfo.meanArr.at<cv::Vec3f>(pairID, idk)[2] = m2;
					arrInfo.stdArr.at<cv::Vec3f>(pairID, idk)[2] = s2;
				}
			}
		}
	}
	arrInfo.distmean = medianStds(arrInfo.stdArr);
	std::cout << "distmean = " << arrInfo.distmean << std::endl;

	//saveMat(outfolder + "\\meanArr.m", arrInfo.meanArr);
	//saveMat(outfolder + "\\stdArr.m", arrInfo.stdArr);

	auto end = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "End Norm Build, time use: " << duration / 1000.0 << "s" << std::endl;
}


bool tringulate3Dline(
	float* cam2, float* M2, float* C2, float* line2,
	float* M1, float* C1, float* line1,
	float* pt3d1, float* pt3d2, float* maxcos, float* mindis)
{

	float* lpt1 = line1;
	float* lpt2 = line1 + 2;

	float* rpt1 = line2;
	float* rpt2 = line2 + 2;

	//plane
	float plane[4];
	float lf2[3];

	cross_v2(line2, lf2);
	line2_spaceplane(lf2, cam2, plane);

	float ray1[3], ray2[3];

	//ray1
	solverAxb(M1, lpt1, ray1);

	if (1)
	{
		float sina = vectorPlaneSin_cpu(ray1, plane);
		// printf("sina %f\n",sina);
		if (sina < LINE_plane_sin_min)
			return false;
	}

	//ray and plane intersection
	float planePoint[3] = { 0,0,-plane[3] / plane[2] };
	rayInterPlane(ray1, C1, plane, planePoint, pt3d1);

	//ray2
	solverAxb(M1, lpt2, ray2);

	if (1)
	{
		float sina = vectorPlaneSin_cpu(ray2, plane);
		// printf("sina %f\n",sina);
		if (sina < LINE_plane_sin_min)
			return false;
	}

	//ray and plane intersection
	rayInterPlane(ray2, C1, plane, planePoint, pt3d2);

	float lineray[3];
	lineray[0] = pt3d2[0] - pt3d1[0];
	lineray[1] = pt3d2[1] - pt3d1[1];
	lineray[2] = pt3d2[2] - pt3d1[2];
	float cos1 = cos_vec3(lineray, ray1);
	float cos2 = cos_vec3(lineray, ray2);

	if (cos1 < 0)
		cos1 = -cos1;
	if (cos2 < 0)
		cos2 = -cos2;


	// point 2 plane dis
	float vv[4];
	equation_plane(C1[0], C1[1], C1[2], C2[0], C2[1], C2[2], pt3d1[0], pt3d1[1], pt3d1[2], vv);
	float p2p1 = point_plane_dis3d(pt3d2, vv);
	equation_plane(C1[0], C1[1], C1[2], C2[0], C2[1], C2[2], pt3d2[0], pt3d2[1], pt3d2[2], vv);
	float p2p2 = point_plane_dis3d(pt3d1, vv);

	*maxcos = max_2(cos1, cos2);
	*mindis = min_2(p2p1, p2p2);


	return true;

}

cv::Mat triangulateLine(cv::Mat lineids, cv::Mat camids, std::vector<cv::Mat> cameras, std::vector<cv::Mat> centers, std::vector<cv::Mat> lines) {
	int camid1 = camids.at<float>(0, 0);
	int lineid1 = lineids.at<float>(0, 0);
	cv::Mat line1 = lines[camid1].row(lineid1).colRange(0, 4);
	cv::Mat cen1 = centers[camid1];
	cv::Mat Mt1 = cameras[camid1].rowRange(0, 3).colRange(0, 3).t();

	int camid2 = camids.at<float>(0, 1);
	int lineid2 = lineids.at<float>(0, 1);
	cv::Mat line2 = lines[camid2].row(lineid2).colRange(0, 4);
	cv::Mat cen2 = centers[camid2];
	cv::Mat Mt2 = cameras[camid2].rowRange(0, 3).colRange(0, 3).t();
	cv::Mat cam2 = cameras[camid2];

	float pt3d1[3] = { 0,0,0 };
	float pt3d2[3] = { 0,0,0 };
	float mindis = 0;
	float maxcos = 0;

	bool re = tringulate3Dline((float*)cam2.data, (float*)Mt2.data, (float*)cen2.data, (float*)line2.data,
		(float*)Mt1.data, (float*)cen1.data, (float*)line1.data, pt3d1, pt3d2, &maxcos, &mindis);

	cv::Mat l3d = (cv::Mat_<float>(1, 9) << pt3d1[0], pt3d1[1], pt3d1[2], pt3d2[0], pt3d2[1], pt3d2[2], re, maxcos, mindis);

	return l3d;
}

void adaptiveCluster(IMG_INFO imgInfo, cv::Mat camid, cv::Mat lineid,
	cv::Mat counters, cv::Mat meanArr, cv::Mat stdArr,
	cv::Mat& lines3D, cv::Mat& clusters, cv::Mat& meaningFulCluster, cv::Mat& scores) {
	size_t clusterSize = counters.rows;
	lines3D = cv::Mat::zeros(clusterSize, 9, CV_32FC1);
	scores = cv::Mat::ones(clusterSize, 1, CV_32FC1) * -999;
	clusters = cv::Mat::zeros(clusterSize, lineid.cols, CV_32FC1);
	clusters.colRange(0, 2) = 1.0;
	cv::Mat ps = cv::Mat::zeros(camid.cols, 1, CV_32FC1);

	meaningFulCluster = cv::Mat();
	clusters.copyTo(meaningFulCluster);

	cv::Mat subind = cv::Mat::zeros(1, lineid.cols, CV_16SC1);

	for (int i = 0; i < clusterSize; i++) {
		auto counter = counters.at<float>(i, 0);
		if (counter <= 2) continue;

		cv::Mat mainlineid = lineid.row(i).colRange(0, 2);
		cv::Mat maincamid = camid.row(i).colRange(0, 2);

		cv::Mat sublineid = lineid.row(i).colRange(2, counter);
		cv::Mat subcamid = camid.row(i).colRange(2, counter);

		cv::Mat l3d = triangulateLine(mainlineid, maincamid, imgInfo.cameras, imgInfo.centers, imgInfo.lines);

		l3d.copyTo(lines3D.row(i));
		if (l3d.at<float>(0, 6) == 0) continue;
		cv::Mat vec1 = l3d.colRange(0, 3) - l3d.colRange(3, 6);
		auto len = cv::norm(vec1);

		int cc = 0;
		for (int j = 0; j < sublineid.cols; j++) {
			cv::Mat mean3 = meanArr.row(subcamid.at<float>(0, j));
			cv::Mat std3 = stdArr.row(subcamid.at<float>(0, j));

			if (std3.at<float>(0, 0) == 0) continue;

			cv::Mat camj = imgInfo.cameras[subcamid.at<float>(0, j)];
			cv::Mat linej = imgInfo.lines[subcamid.at<float>(0, j)].row(sublineid.at<float>(0, j)).colRange(0, 4);
			cv::Mat cenj = imgInfo.centers[subcamid.at<float>(0, j)];

			cv::Mat pt1 = (cv::Mat_<float>(3, 1) << linej.at<float>(0, 0), linej.at<float>(0, 1), 1.0);
			cv::Mat pt2 = (cv::Mat_<float>(3, 1) << linej.at<float>(0, 2), linej.at<float>(0, 3), 1.0);
			cv::Mat ray1 = camj.rowRange(0, 3).colRange(0, 3).inv() * pt1;
			cv::Mat ray2 = camj.rowRange(0, 3).colRange(0, 3).inv() * pt2;

			cv::Vec3d O1;
			cv::Vec3d a2, b2;
			cv::Vec3d O22;
			double dist;
			line2lineDist(l3d, cenj, cenj + 10.0 * ray1.t(), O1, a2, dist);
			line2lineDist(l3d, cenj, cenj + 10.0 * ray2.t(), O1, b2, dist);
			auto vec2 = a2 - b2;
			double ang = calculateAngle(vec1, vec2);
			double maxang = atan(cv::norm(3 * std3) / len);
			if (ang > maxang || ang < 0) continue;

			cv::Mat aa2 = (cv::Mat_<float>(1, 3) << a2[0], a2[1], a2[2]);
			cv::Vec3f err1 = pt2LineDis(aa2, l3d.row(0).colRange(0, 3), l3d.row(0).colRange(3, 6));
			if (err1[0] > mean3.at<float>(0, 0) + 3 * std3.at<float>(0, 0) || err1[0] < mean3.at<float>(0, 0) - 3 * std3.at<float>(0, 0) ||
				err1[1] > mean3.at<float>(0, 1) + 3 * std3.at<float>(0, 1) || err1[1] < mean3.at<float>(0, 1) - 3 * std3.at<float>(0, 1) ||
				err1[2] > mean3.at<float>(0, 2) + 3 * std3.at<float>(0, 2) || err1[2] < mean3.at<float>(0, 2) - 3 * std3.at<float>(0, 2)
				)continue;

			cv::Mat bb2 = (cv::Mat_<float>(1, 3) << b2[0], b2[1], b2[2]);
			cv::Vec3f err2 = pt2LineDis(bb2, l3d.row(0).colRange(0, 3), l3d.row(0).colRange(3, 6));
			if (err2[0] > mean3.at<float>(0, 0) + 3 * std3.at<float>(0, 0) || err2[0] < mean3.at<float>(0, 0) - 3 * std3.at<float>(0, 0) ||
				err2[1] > mean3.at<float>(0, 1) + 3 * std3.at<float>(0, 1) || err2[1] < mean3.at<float>(0, 1) - 3 * std3.at<float>(0, 1) ||
				err2[2] > mean3.at<float>(0, 2) + 3 * std3.at<float>(0, 2) || err2[2] < mean3.at<float>(0, 2) - 3 * std3.at<float>(0, 2)
				)continue;

			Eigen::Vector3d err(std::max(std::abs(err1[0]), std::abs(err2[0])),
				std::max(std::abs(err1[1]), std::abs(err2[1])),
				std::max(std::abs(err1[2]), std::abs(err2[2])));

			float a = mean3.at<float>(0);
			float b = mean3.at<float>(1);
			float c = mean3.at<float>(2);
			Eigen::Vector3d mean(a, b, c);

			Eigen::Matrix3d std333(3, 3);
			std333 << std3.at<float>(0) * std3.at<float>(0), 0, 0,
				0, std3.at<float>(1)* std3.at<float>(1), 0,
				0, 0, std3.at<float>(2)* std3.at<float>(2);

			double p1 = mvncdf(err, mean, std333);
			double p2 = ang / maxang;
			double p = p1 * p2;

			ps.at<float>(cc) = p;
			clusters.at<float>(i, j + 2) = 1;
			subind.at<short>(cc) = j;
			cc += 1;
		}

		if (cc < 2) continue;
		// 提取ps的前cc个元素
		cv::Mat ps_subset;
		ps.rowRange(0, cc).copyTo(ps_subset);

		// 将ps_subset转换为std::vector
		std::vector<float> ps_vector;
		ps_vector.assign((float*)ps_subset.datastart, (float*)ps_subset.dataend);

		// 创建索引向量
		std::vector<int> indices(ps_vector.size());
		for (int i = 0; i < indices.size(); ++i) {
			indices[i] = i;
		}

		// 使用lambda表达式对索引进行排序
		std::sort(indices.begin(), indices.end(), [&ps_vector](int i1, int i2) {
			return ps_vector[i1] < ps_vector[i2];
			});

		// 创建排序后的向量
		std::vector<float> ps_sort(ps_vector.size());
		for (int i = 0; i < indices.size(); ++i) {
			ps_sort[i] = ps_vector[indices[i]];
		}

		double bestnfa = -999;
		double logn = 1;
		int bestj = 1;
		for (int j = 0; j < ps_sort.size(); j++) {
			float nfai = nfa(ps_sort.size(), j + 1, ps_sort[j], logn);
			if (nfai > bestnfa) {
				bestnfa = nfai;
				bestj = j;
			}
		}

		scores.at<float>(i, 0) = bestnfa;

		//std::vector<int> select_indices;
		for (int j = 0; j < bestj + 1; j++) {
			//select_indices.push_back(subind.at<short>(indices[j]) + 2);
			meaningFulCluster.at<float>(i, subind.at<short>(indices[j]) + 2) = 1;
		}

	}
}

// Function to concatenate a vector of cv::Mat vertically
cv::Mat vconcatMats(const std::vector<cv::Mat>& mats) {
	if (mats.empty()) return cv::Mat();
	cv::Mat result = mats[0].clone();
	for (size_t i = 1; i < mats.size(); ++i) {
		cv::vconcat(result, mats[i], result);
	}
	return result;
}

void greedAssign(std::vector<cv::Mat> lines3D_vec, std::vector<cv::Mat> clusters_vec,
	std::vector<cv::Mat> meaningFulCluster_vec, std::vector<cv::Mat> scores_vec,
	SFM_INFO sfmInfo, IMG_INFO imgInfo, SPACE_REC& spaceRec) {

	cv::Mat lines3D;
	cv::Mat clusters;
	cv::Mat scores;
	cv::Mat camid;
	cv::Mat lineid;
	cv::Mat counters;
	cv::Mat meaningFulCluster;

	for (int i = 0; i < scores_vec.size(); i++) {
		for (int j = 0; j < scores_vec[i].rows; j++) {
			if (scores_vec[i].at<float>(j) != -999) {
				lines3D.push_back(lines3D_vec[i].row(j));
				clusters.push_back(clusters_vec[i].row(j));
				scores.push_back(scores_vec[i].row(j));
				camid.push_back(sfmInfo.camID[i].row(j));
				lineid.push_back(sfmInfo.lineID[i].row(j));
				counters.push_back(sfmInfo.counters[i].row(j));
				meaningFulCluster.push_back(meaningFulCluster_vec[i].row(j));
			}
		}
	}

	cv::Mat inds;
	cv::sortIdx(scores, inds, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);

	cv::Mat valids = cv::Mat::zeros(inds.rows, 1, CV_8UC1);

	std::vector<cv::Mat> used_lines; used_lines.resize(imgInfo.lines.size());
	for (int i = 0; i < imgInfo.lines.size(); i++) {
		used_lines[i] = cv::Mat::zeros(imgInfo.lines[i].rows, 1, CV_8UC1);
	}

	for (int i = 0; i < inds.rows; i++) {
		int cid = inds.at<int>(i);
		auto nfaIDs = clusters.row(cid);
		auto subcam = camid.row(cid);
		auto subline = lineid.row(cid);

		bool isvalid = true;

		for (int j = 0; j < (int)counters.at<float>(cid); j++) {
			if (nfaIDs.at<float>(j) == 0) continue;
			if (used_lines[subcam.at<float>(j)].at<uchar>(subline.at<float>(j)) == 1) {
				isvalid = false;
				break;
			}
		}

		for (int j = 0; j < (int)counters.at<float>(cid); j++) {
			if (nfaIDs.at<float>(j) == 0) continue;
			used_lines[subcam.at<float>(j)].at<uchar>(subline.at<float>(j)) = 1;
		}
		valids.at<uchar>(cid) = isvalid;
	}

	for (int i = 0; i < valids.rows; i++) {
		if (valids.at<uchar>(i) == 1) {
			spaceRec.lines3D.push_back(lines3D.row(i));
			spaceRec.camid.push_back(camid.row(i));
			spaceRec.clusters.push_back(meaningFulCluster.row(i));
			spaceRec.counters.push_back(counters.row(i));
			spaceRec.lineid.push_back(lineid.row(i));
		}
	}

}

void callAdaptiveLineCluster(SFM_INFO& sfmInfo, IMG_INFO& imgInfo, ARR_INFO& arrInfo, SPACE_REC& spaceRec) {
	std::cout << std::endl;
	std::cout << "Start Adaptive Line Cluster:..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<cv::Mat> line3Ds; line3Ds.resize(sfmInfo.counters.size());
	std::vector<cv::Mat> clusters; clusters.resize(sfmInfo.counters.size());
	std::vector<cv::Mat> meaningFulClusters; meaningFulClusters.resize(sfmInfo.counters.size());
	std::vector<cv::Mat> scores; scores.resize(sfmInfo.counters.size());
	for (int i = 0; i < sfmInfo.counters.size(); i++) {
		if (sfmInfo.counters[i].empty())continue;
		cv::Mat meanA = arrInfo.meanArr.row(i);
		meanA = meanA.reshape(1, meanA.rows * meanA.cols);
		cv::Mat stdA = arrInfo.stdArr.row(i);
		stdA = stdA.reshape(1, stdA.rows * stdA.cols);

		adaptiveCluster(imgInfo, sfmInfo.camID[i], sfmInfo.lineID[i], sfmInfo.counters[i], meanA, stdA,
			line3Ds[i], clusters[i], meaningFulClusters[i], scores[i]);

	}
	greedAssign(line3Ds, clusters, meaningFulClusters, scores, sfmInfo, imgInfo, spaceRec);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "End Adaptive Line Cluster, time use: " << duration / 1000.0 << "s" << std::endl;

}

std::vector<cv::Mat> extractImageLines(IMG_INFO imgInfo, SPACE_REC spaceRec) {
	std::vector<cv::Mat> etlines;
	etlines.resize(imgInfo.cameras.size());
	for (int i = 0; i < spaceRec.camid.rows; i++) {
		for (int j = 0; j < spaceRec.counters.at<float>(i); j++) {
			if (spaceRec.clusters.at<float>(i, j) == 0) continue;
			int cid = spaceRec.camid.at<float>(i, j);
			//counters.at<float>(cid) = counters.at<float>(cid) + 1;
			cv::Mat rowLine = (cv::Mat_<float>(1, 2) << spaceRec.lineid.at<float>(i, j), i);
			etlines[cid].push_back(rowLine.row(0));
		}
	}
	return etlines;
}

// Function to normalize vectors
cv::Mat normalizeVectors(const cv::Mat& vecs) {
	cv::Mat normVecs = vecs.clone();
	for (int i = 0; i < vecs.rows; ++i) {
		cv::normalize(normVecs.row(i), normVecs.row(i));
	}
	return normVecs;
}

// Distance evaluation function
std::vector<bool> evalLineFcn(const cv::Mat& model, const cv::Mat& vecs, double maxDistance, double& accDis) {
	cv::Mat dotProducts = vecs * model.t();
	std::vector<bool> inliers(dotProducts.rows);
	accDis = 0.0;
	for (int i = 0; i < dotProducts.rows; ++i) {
		double val = std::abs(dotProducts.at<float>(i, 0));
		// Clamp value to [-1, 1] range
		//val = std::min(std::max(val, -1.0), 1.0);
		double distance = std::acos(val);
		//inliers[i] = distance < maxDistance;

		if (distance < maxDistance) {
			inliers[i] = true;
			accDis += distance;
		}
		else {
			inliers[i] = false;
			accDis += maxDistance;
		}
	}
	return inliers;
}

cv::Mat fitLineFunction(const cv::Mat samples) {
	return samples;
}

int computeLoopNumber(int sampleSize, double confidence, int pointNum, int inlierNum) {
	double inlierProbability = std::pow(((double)inlierNum / (double)pointNum), sampleSize);
	double num = std::log10(1 - confidence);
	double den = std::log10(1 - inlierProbability);
	return (int)std::ceil(num / den);
}

// RANSAC function
bool ransac(const cv::Mat& vecs, cv::Mat& model, std::vector<bool>& inliers, int sampleSize, double maxDistance, double confidence, int in_maxIter) {

	int maxIter = in_maxIter;

	int bestInlierCount = 0;
	double bestInlierAccDis = maxDistance * (double)vecs.rows;
	cv::Mat bestModel;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, vecs.rows - 1);

	for (int iter = 0; iter < maxIter; ++iter) {
		// Randomly sample points
		std::vector<int> sampleIndices(sampleSize);
		for (int i = 0; i < sampleSize; ++i) {
			sampleIndices[i] = dis(gen);
		}

		// Fit model
		cv::Mat sampleVecs(sampleSize, vecs.cols, vecs.type());
		for (int i = 0; i < sampleSize; ++i) {
			vecs.row(sampleIndices[i]).copyTo(sampleVecs.row(i));
		}
		//cv::Mat sampleMean;
		//cv::reduce(sampleVecs, sampleMean, 0, cv::REDUCE_AVG);

		cv::Mat sampleMean = fitLineFunction(sampleVecs);

		// Evaluate model
		double accDis;
		std::vector<bool> currentInliers = evalLineFcn(sampleMean, vecs, maxDistance, accDis);
		int currentInlierCount = std::count(currentInliers.begin(), currentInliers.end(), true);

		// Update best model if current model is better
		//if (currentInlierCount > bestInlierCount) {
		if (accDis < bestInlierAccDis) {
			bestInlierCount = currentInlierCount;
			bestInlierAccDis = accDis;
			bestModel = sampleMean;
			inliers = currentInliers;
			maxIter = std::min(computeLoopNumber(sampleSize, confidence, vecs.rows, bestInlierCount), maxIter);//动态计算迭代次数
			if (currentInlierCount > vecs.rows * confidence) {
				break;
			}
		}
	}

	if (bestInlierCount > 0) {
		model = bestModel;
		return true;
	}
	return false;
}

void ransacLines(cv::Mat lines3d, PARAMS param, cv::Mat& inlierIdx, std::vector<cv::Mat>& models) {


	if (lines3d.rows < 5)
		return;
	

	cv::Mat vecs = lines3d.colRange(0, 3) - lines3d.colRange(3, 6);



	vecs = normalizeVectors(vecs);
	inlierIdx = cv::Mat::zeros(vecs.rows, 1, CV_32S);
	models.clear();

	int cc = 0;
	while (true) {
		std::vector<int> ids;
		for (int i = 0; i < vecs.rows; ++i) {
			if (inlierIdx.at<int>(i) == 0) {
				ids.push_back(i);
			}
		}

		if (ids.empty()) break;
		cv::Mat remainingVecs(ids.size(), vecs.cols, vecs.type());
		for (size_t i = 0; i < ids.size(); ++i) {
			vecs.row(ids[i]).copyTo(remainingVecs.row(i));
		}

		cv::Mat model;
		std::vector<bool> subInliers;
		bool found = ransac(remainingVecs, model, subInliers, 1, param.maxAng, 0.90, 1000);

		if (!found) {
			break;
		}

		std::vector<int> ids_sub;
		for (size_t i = 0; i < subInliers.size(); ++i) {
			if (subInliers[i]) {
				ids_sub.push_back(ids[i]);
			}
		}

		if (ids_sub.size() < param.colinearNum) {
			break;
		}

		cc++;
		for (int idx : ids_sub) {
			inlierIdx.at<int>(idx) = cc;
		}
		models.push_back(model);

		if (0) {
			break;
		}
	}
}

double validateCell(const cv::Mat& cami, const cv::Mat& linei, const cv::Point3d& p1, const cv::Point3d& p2, double thre) {
	double score = 0;

	// Convert p1 and p2 to homogeneous coordinates
	cv::Mat p1_hom = (cv::Mat_<float>(4, 1) << p1.x, p1.y, p1.z, 1);
	cv::Mat p2_hom = (cv::Mat_<float>(4, 1) << p2.x, p2.y, p2.z, 1);

	// Project p1 and p2 using cami (assuming cami is a 3x4 matrix)
	cv::Mat rp1 = cami * p1_hom;
	cv::Mat rp2 = cami * p2_hom;

	cv::Mat	lf1 = rp1.cross(rp2);
	double p2l1 = std::abs((linei.at<float>(0, 0) * lf1.at<float>(0) + linei.at<float>(0, 1) * lf1.at<float>(1) + lf1.at<float>(2))) / cv::norm(lf1.rowRange(0, 2));
	if (p2l1 > thre) {
		return 0;
	}

	double p2l2 = std::abs((linei.at<float>(0, 2) * lf1.at<float>(0) + linei.at<float>(0, 3) * lf1.at<float>(1) + lf1.at<float>(2))) / cv::norm(lf1.rowRange(0, 2));
	if (p2l2 > thre) {
		return 0;
	}

	score = thre - std::max(p2l1, p2l2);

	return score;
}

void validateInImage(cv::Point3d& p1, cv::Point3d& p2, std::vector<cv::Mat> cameras, std::vector<cv::Mat> lines,
	cv::Mat camids, cv::Mat lineids, cv::Mat clusters, int counter, double threshold, double& score) {
	score = 0;
	for (int i = 0; i < counter; i++) {
		if (clusters.at<float>(0, i) == 0) {
			continue;
		}

		int cid = camids.at<float>(0, i);
		int lid = lineids.at<float>(0, i);
		double scoreCell = validateCell(cameras[cid], lines[cid].row(lid), p1, p2, threshold);

		if (scoreCell == 0 && (i == 1 || i == 0)) return;
		if (i == 1 || i == 0) continue;
		score += scoreCell;
	}
}

#pragma region NLOPT_BLOCK

cv::Mat plucker_matrix(const cv::Mat& A, const cv::Mat& B) {
	return A * B.t() - B * A.t();
}

cv::Mat skewMat_inverse(const cv::Mat& M) {
	cv::Mat result = (cv::Mat_<float>(3, 1) << M.at<float>(2, 1), M.at<float>(0, 2), M.at<float>(1, 0));
	return result;
}

double validateInImage(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const std::vector<Eigen::MatrixXd>& cameras,
	const std::vector<Eigen::MatrixXd>& lines, const Eigen::Vector2i& camids, const Eigen::Vector2i& lineids, const Eigen::VectorXd& clusters,
	int counter, double projdis) {
	// Placeholder implementation for validation logic
	return 1.0; // Return a dummy score
}

struct OptimizationData {
	cv::Point3d cen1;
	cv::Point3d ray1;
	cv::Point3d ray2;
	cv::Point3d vecModel;
	IMG_INFO* imageInfo;
	SPACE_REC* spaceRec;
	cv::Vec2i camids;
	cv::Vec2i lineids;
	std::string cons;
};

double mmax_par(const std::vector<double>& x, std::vector<double>& grad, void* data) {
	OptimizationData* optData = static_cast<OptimizationData*>(data);
	cv::Point3d pt1 = optData->cen1 + x[0] * optData->ray1;
	cv::Point3d pt2 = optData->cen1 + x[1] * optData->ray2;
	cv::Point3d vec0 = pt1 - pt2;
	cv::Point3d crossv = vec0.cross(optData->vecModel);
	return crossv.dot(crossv);
}

double mmax_var(const std::vector<double>& x, std::vector<double>& grad, void* data) {
	OptimizationData* optData = static_cast<OptimizationData*>(data);
	cv::Point3d pt1 = optData->cen1 + x[0] * optData->ray1;
	cv::Point3d pt2 = optData->cen1 + x[1] * optData->ray2;
	cv::Point3d vec0 = pt1 - pt2;
	return pow(vec0.dot(optData->vecModel), 2);
}

void mcon(const std::vector<double>& x, std::vector<double>& constraints, std::vector<double>& ceq, void* data) {
	OptimizationData* optData = static_cast<OptimizationData*>(data);
	cv::Point3d pt1 = optData->cen1 + x[0] * optData->ray1;
	cv::Point3d pt2 = optData->cen1 + x[1] * optData->ray2;
	cv::Mat LM = plucker_matrix((cv::Mat_<float>(4, 1) << pt1.x, pt1.y, pt1.z, 1.0), (cv::Mat_<float>(4, 1) << pt2.x, pt2.y, pt2.z, 1.0));

	for (int j = 0; j < 2; ++j) {
		cv::Mat P = optData->imageInfo->cameras[optData->camids[j]];
		cv::Mat lx = P * LM * P.t();
		cv::Mat lx_inv = skewMat_inverse(lx);
		cv::Mat line = optData->imageInfo->lines[optData->camids[j]].row(optData->lineids[j]);
		constraints.push_back(pow(line.at<float>(0) * lx_inv.at<float>(0) + line.at<float>(1) * lx_inv.at<float>(1) + lx_inv.at<float>(2), 2) / (norm(lx_inv.rowRange(0, 2)) * norm(lx_inv.rowRange(0, 2))) - 4);
		constraints.push_back(pow(line.at<float>(2) * lx_inv.at<float>(0) + line.at<float>(3) * lx_inv.at<float>(1) + lx_inv.at<float>(2), 2) / (norm(lx_inv.rowRange(0, 2)) * norm(lx_inv.rowRange(0, 2))) - 4);
	}
}

double constraint_wrapper(const std::vector<double>& x, std::vector<double>& grad, void* data) {
	double* constraintData = static_cast<double*>(data);
	return *constraintData;
}

double nlopt_vecRefine3(IMG_INFO imgInfo, SPACE_REC spaceRec, int id, cv::Mat vecModel, PARAMS param, std::string cons,
	cv::Point3d& p1, cv::Point3d& p2) {

	auto cam1 = spaceRec.camid.at<float>(id, 0);
	auto cen1 = imgInfo.centers[cam1];

	OptimizationData optData;
	optData.cen1 = cv::Point3d(cen1.at<float>(0, 0), cen1.at<float>(0, 1), cen1.at<float>(0, 2));
	optData.ray1 = (cv::Point3d(spaceRec.lines3D.at<float>(id, 0), spaceRec.lines3D.at<float>(id, 1), spaceRec.lines3D.at<float>(id, 2))
		- optData.cen1);

	optData.ray2 = (cv::Point3d(spaceRec.lines3D.at<float>(id, 3), spaceRec.lines3D.at<float>(id, 4), spaceRec.lines3D.at<float>(id, 5))
		- optData.cen1);

	optData.ray1 *= (1.0 / norm(optData.ray1));
	optData.ray2 *= (1.0 / norm(optData.ray2));
	optData.vecModel = cv::Point3d(vecModel.at<float>(0, 0), vecModel.at<float>(0, 1), vecModel.at<float>(0, 2));

	optData.imageInfo = &imgInfo;
	optData.spaceRec = &spaceRec;

	optData.camids = cv::Vec2i(spaceRec.camid.at<float>(id, 0), spaceRec.camid.at<float>(id, 1));
	optData.lineids = cv::Vec2i(spaceRec.lineid.at<float>(id, 0), spaceRec.lineid.at<float>(id, 1));
	optData.cons = cons;

	std::vector<double> x0 = { cv::norm(cv::Point3d(spaceRec.lines3D.at<float>(id, 0), spaceRec.lines3D.at<float>(id, 1), spaceRec.lines3D.at<float>(id, 2)) - optData.cen1),
					   cv::norm(cv::Point3d(spaceRec.lines3D.at<float>(id, 3), spaceRec.lines3D.at<float>(id, 4), spaceRec.lines3D.at<float>(id, 5)) - optData.cen1) };

	nlopt::opt opt(nlopt::LD_SLSQP, 2);
	if (cons == "var") {
		opt.set_min_objective(mmax_var, &optData);
	}
	else {
		opt.set_min_objective(mmax_par, &optData);
	}

	std::vector<double> constraints;
	std::vector<double> ceq;
	mcon(x0, constraints, ceq, &optData);

	for (auto& c : constraints) {
		opt.add_inequality_constraint(constraint_wrapper, &c, 1e-8);
	}

	opt.set_xtol_rel(1e-4);
	opt.set_maxeval(5000); 
	
	try {
		double minf;
		nlopt::result result = opt.optimize(x0, minf);
		
	}
	catch (std::exception& e) {
		std::cout << "std::exception e" << std::endl;
		return -1;
	}

	double score = 0;
	p1 = cv::Point3d(0, 0, 0);
	p2 = cv::Point3d(0, 0, 0);
	p1 = optData.cen1 + x0[0] * optData.ray1;
	p2 = optData.cen1 + x0[1] * optData.ray2;
	validateInImage(p1, p2, imgInfo.cameras, imgInfo.lines, spaceRec.camid.row(id),
		spaceRec.lineid.row(id), spaceRec.clusters.row(id), spaceRec.counters.at<float>(id, 0), param.projdis,
		score);
	return score;

}

#pragma endregion

cv::Mat colinearRefine(IMG_INFO imgInfo, SPACE_REC spaceRec, PARAMS param) {
	std::cout << std::endl;
	std::cout << "Start Colinear Refine:..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<cv::Mat> etlines = extractImageLines(imgInfo, spaceRec);
	cv::Mat refLines = cv::Mat::zeros(spaceRec.lines3D.rows, 7, CV_32FC1);

	for (int i = 0; i < etlines.size(); i++) {
		cv::Mat sublines;
		for (int j = 0; j < etlines[i].rows; j++) {
			sublines.push_back(spaceRec.lines3D.row(etlines[i].at<float>(j, 1)));
		}

		cv::Mat inlierIdx;
		std::vector<cv::Mat> models;
		ransacLines(sublines, param, inlierIdx, models);

		for (int j = 0; j < models.size(); j++) {
			std::vector<int> id3s;
			for (int row = 0; row < inlierIdx.rows; row++) {
				if (inlierIdx.at<int>(row) == j + 1) {
					id3s.push_back(etlines[i].at<float>(row, 1));
				}
			}

			for (int k = 0; k < id3s.size(); k++) {
				int l3id = id3s[k];

				cv::Point3d p1, p2;
				double score = nlopt_vecRefine3(imgInfo, spaceRec, l3id, models[j], param, "par", p1, p2);
				//double score = 0.0;
				//vecRefine3(imgInfo, spaceRec, l3id, models[j], param, "par", score, p1, p2);
				if (refLines.at<float>(l3id, 0) < score) {
					cv::Mat l = (cv::Mat_<float>(1, 7) << score, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
					l.row(0).copyTo(refLines.row(l3id));
				}
			}
		}
	}

	//saveMat(R"(E:\Research\ReadSFM\i16\output\colinearLines.m)", refLines);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "End Colinear Refine, time use: " << duration / 1000.0 << "s" << std::endl;
	return refLines;

}

cv::Mat fitPlane(const cv::Mat& lines) {
	cv::Mat A = cv::Mat(4, 3, CV_32FC1);
	lines.row(0).colRange(0, 3).copyTo(A.row(0));
	lines.row(0).colRange(3, 6).copyTo(A.row(2));
	lines.row(1).colRange(0, 3).copyTo(A.row(1));
	lines.row(1).colRange(3, 6).copyTo(A.row(3));
	cv::Mat b = cv::Mat::ones(4, 1, CV_32FC1) * -1;
	cv::Mat model;
	cv::solve(A, b, model, cv::DECOMP_SVD);
	return model;
}

std::vector<bool> evalPlaneFcn(const cv::Mat& model, const cv::Mat& lines, double maxDis, double& accDis) {
	std::vector<bool> results;
	results.resize(lines.rows);
	double norm_model = cv::norm(model) * cv::norm(model);
	double model_0 = model.at<float>(0, 0);
	double model_1 = model.at<float>(1, 0);
	double model_2 = model.at<float>(2, 0);
	accDis = 0.0;
	for (int i = 0; i < lines.rows; i++) {
		double a1 = lines.at<float>(i, 0) * model_0 + lines.at<float>(i, 1) * model_1 + lines.at<float>(i, 2) * model_2 + 1.0;
		a1 = a1 * a1 / norm_model;
		double a2 = lines.at<float>(i, 3) * model_0 + lines.at<float>(i, 4) * model_1 + lines.at<float>(i, 5) * model_2 + 1.0;
		a2 = a2 * a2 / norm_model;
		if (std::max(a1, a2) < maxDis) {
			results[i] = true;
			accDis += std::max(a1, a2);
		}
		else {
			results[i] = false;
			accDis += maxDis;
		}
	}
	return results;
}

// RANSAC function
bool ransac2(const cv::Mat& vecs, cv::Mat& model, std::vector<bool>& inliers, int sampleSize, double maxDistance, double confidence, int in_maxIter) {
	int bestInlierCount = 0;
	double bestInlierAccDis = maxDistance * (double)vecs.rows;
	cv::Mat bestModel;
	int maxIter = in_maxIter;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, vecs.rows - 1);

	for (int iter = 0; iter < maxIter; ++iter) {
		// Randomly sample points
		std::vector<int> sampleIndices(sampleSize);
		for (int i = 0; i < sampleSize; ++i) {
			sampleIndices[i] = dis(gen);
		}

		// Fit model
		cv::Mat sampleVecs(sampleSize, vecs.cols, vecs.type());
		for (int i = 0; i < sampleSize; ++i) {
			vecs.row(sampleIndices[i]).copyTo(sampleVecs.row(i));
		}
		//cv::Mat sampleMean;
		//cv::reduce(sampleVecs, sampleMean, 0, cv::REDUCE_AVG);

		cv::Mat sampleMean = fitPlane(sampleVecs);

		// Evaluate model
		double accDis;
		std::vector<bool> currentInliers = evalPlaneFcn(sampleMean, vecs, maxDistance, accDis);
		int currentInlierCount = std::count(currentInliers.begin(), currentInliers.end(), true);

		// Update best model if current model is better
		if (accDis < bestInlierAccDis) {
			//std::cout << currentInlierCount << "\t" << sampleIndices[0] << "\t" << sampleIndices[1] << std::endl;
			bestInlierCount = currentInlierCount;
			bestInlierAccDis = accDis;
			bestModel = sampleMean;
			inliers = currentInliers;
			maxIter = std::min(computeLoopNumber(sampleSize, confidence, vecs.rows, bestInlierCount), maxIter);//动态计算迭代次数
			if (currentInlierCount > vecs.rows * confidence) {
				break;
			}
		}
	}

	if (bestInlierCount > 0) {
		model = cv::Mat::ones(4, 1, bestModel.type());
		bestModel.rowRange(0, 3).copyTo(model.rowRange(0, 3));
		return true;
	}
	return false;
}

void ransacPlanes(cv::Mat lines3d, PARAMS param, cv::Mat& inlierIdx, std::vector<cv::Mat>& models) {
	double maxdistance = param.dist3D * param.dist3D;

	cv::Mat vecs;
	lines3d.copyTo(vecs);

	inlierIdx = cv::Mat::zeros(vecs.rows, 1, CV_32S);
	models.clear();

	int cc = 0;
	while (true) {
		std::vector<int> ids;
		for (int i = 0; i < vecs.rows; ++i) {
			if (inlierIdx.at<int>(i) == 0) {
				ids.push_back(i);
			}
		}

		if (ids.empty()) break;
		cv::Mat remainingVecs(ids.size(), vecs.cols, vecs.type());
		for (size_t i = 0; i < ids.size(); ++i) {
			vecs.row(ids[i]).copyTo(remainingVecs.row(i));
		}

		cv::Mat model;
		std::vector<bool> subInliers;
		bool found = ransac2(remainingVecs, model, subInliers, 2, maxdistance, 0.90, 1000);

		if (!found) {
			break;
		}

		std::vector<int> ids_sub;
		for (size_t i = 0; i < subInliers.size(); ++i) {
			if (subInliers[i]) {
				ids_sub.push_back(ids[i]);
			}
		}

		if (ids_sub.size() < param.coplanarNum) {
			break;
		}

		cc++;
		for (int idx : ids_sub) {
			inlierIdx.at<int>(idx) = cc;
		}
		models.push_back(model);

		if (0) {
			break;
		}
	}
}

cv::Mat rayInterPlane(cv::Mat rayVector, cv::Mat rayPoint, cv::Mat planeNormal, cv::Mat planePoint) {
	cv::Mat diff3 = rayPoint - planePoint;
	double prod1 = diff3.dot(planeNormal.t());
	double prod2 = rayVector.dot(planeNormal.t());
	double prod3 = prod1 / prod2;
	cv::Mat interPoint = rayPoint - rayVector * prod3;
	return interPoint;
}

cv::Mat coplanarRef(IMG_INFO imgInfo, SPACE_REC spaceRec, PARAMS param) {
	std::cout << std::endl;
	std::cout << "Start Coplanar Refine:..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	std::vector<cv::Mat> etlines = extractImageLines(imgInfo, spaceRec);
	cv::Mat refLines = cv::Mat::zeros(spaceRec.lines3D.rows, 10, CV_32FC1);

	cv::Mat cen1, ray1, ray2, palnePoint, interPoint1, interPoint2, plane;

	for (int i = 0; i < etlines.size(); i++) {
		cv::Mat sublines;
		for (int j = 0; j < etlines[i].rows; j++) {
			sublines.push_back(spaceRec.lines3D.row(etlines[i].at<float>(j, 1)));
		}
		cv::Mat inlierIdx;
		std::vector<cv::Mat> models;
		ransacPlanes(sublines, param, inlierIdx, models);

		for (int j = 0; j < models.size(); j++) {
			std::vector<int> id3s;
			for (int row = 0; row < inlierIdx.rows; row++) {
				if (inlierIdx.at<int>(row) == j + 1) {
					id3s.push_back(etlines[i].at<float>(row, 1));
				}
			}

			plane = models[j];

			for (int k = 0; k < id3s.size(); k++) {
				int l3id = id3s[k];

				int cam1 = spaceRec.camid.at<float>(l3id, 0);
				cen1 = imgInfo.centers[cam1];

				ray1 = spaceRec.lines3D.row(l3id).colRange(0, 3) - cen1;
				ray2 = spaceRec.lines3D.row(l3id).colRange(3, 6) - cen1;

				palnePoint = (cv::Mat_<float>(1, 3) << 0, 0, plane.at<float>(3, 0) / plane.at<float>(2, 0) * -1);
				interPoint1 = rayInterPlane(ray1, cen1, plane.rowRange(0, 3), palnePoint);
				interPoint2 = rayInterPlane(ray2, cen1, plane.rowRange(0, 3), palnePoint);

				cv::Point3d p1(interPoint1.at<float>(0), interPoint1.at<float>(1), interPoint1.at<float>(2));
				cv::Point3d p2(interPoint2.at<float>(0), interPoint2.at<float>(1), interPoint2.at<float>(2));

				double score = 0;
				validateInImage(p1, p2, imgInfo.cameras, imgInfo.lines, spaceRec.camid.row(l3id),
					spaceRec.lineid.row(l3id), spaceRec.clusters.row(l3id), spaceRec.counters.at<float>(l3id), param.projdis,
					score);

				if (refLines.at<float>(l3id, 0) < score) {
					cv::Mat l = (cv::Mat_<float>(1, 10) << score, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, plane.at<float>(0), plane.at<float>(1), plane.at<float>(2));
					l.row(0).copyTo(refLines.row(l3id));
				}

			}

		}


	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "End Coplanar Refine, time use: " << duration / 1000.0 << "s" << std::endl;
	return refLines;
}

void divideSpaceLine(cv::Mat& r_control3D, cv::Mat& r_reviseID, cv::Mat colinearRef, cv::Mat planeRef) {
	std::cout << std::endl;
	std::cout << "Start Divide Space Line:..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	cv::Mat control3D = cv::Mat::zeros(colinearRef.rows, 7, CV_32FC1);
	cv::Mat reviseID = cv::Mat::zeros(colinearRef.rows, 1, CV_32FC1);
	int cc = 0;
	int rcc = 0;
	for (int i = 0; i < colinearRef.rows; i++) {
		double vs = 0;
		int id = -1;
		if (colinearRef.at<float>(i, 0) > planeRef.at<float>(i, 0)) {
			id = 0;
			vs = colinearRef.at<float>(i, 0);
		}
		else {
			id = 1;
			vs = planeRef.at<float>(i, 0);
		}

		if (vs == 0) {
			reviseID.at<float>(rcc, 0) = i;
			rcc++;
			continue;
		}

		if (id == 0) {
			control3D.at<float>(cc, 0) = i;
			colinearRef.row(i).colRange(1, colinearRef.cols).copyTo(control3D.row(cc).colRange(1, 7));
			cc++;
		}
		if (id == 1) {
			control3D.at<float>(cc, 0) = i;
			planeRef.row(i).colRange(1, 7).copyTo(control3D.row(cc).colRange(1, 7));
			cc++;
		}

	}
	control3D.rowRange(0, cc).copyTo(r_control3D);
	reviseID.rowRange(0, rcc).copyTo(r_reviseID);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "End Divide Space Line, time use: " << duration / 1000.0 << "s" << std::endl;
}

cv::Mat lineGrow1(IMG_INFO imgInfo, SPACE_REC spaceRec, cv::Mat control3D, cv::Mat reviseID, PARAMS param) {
	cv::Mat reviseline;

	for (int i = 0; i < reviseID.rows; i++) {
		reviseline.push_back(spaceRec.lines3D.row(reviseID.at<float>(i, 0)));
	}

	cv::Mat searchMat = (control3D.colRange(1, 4) + control3D.colRange(4, 7)) / 2;
	cv::Mat vecs = control3D.colRange(1, 4) - control3D.colRange(4, 7);
	cv::Mat queriesMat = (reviseline.colRange(0, 3) + reviseline.colRange(3, 6)) / 2;

	cv::Mat vecDist, vecIndx;
	cv::flann::KDTreeIndexParams  indexParams(6);
	cv::flann::Index kdtree(searchMat, indexParams);

	cv::flann::SearchParams knn_params(64);

	int knnum = 20;
	if (knnum >= reviseline.rows)
		knnum = reviseline.rows;

	kdtree.knnSearch(queriesMat, vecIndx, vecDist, knnum, knn_params);
	
	cv::Mat reviseRes = cv::Mat::zeros(spaceRec.lines3D.rows, 7, CV_32FC1);

	for (int i = 0; i < reviseID.rows; i++) {
		cv::Mat l3d1 = spaceRec.lines3D.row(reviseID.at<float>(i, 0));
		cv::Mat vec1 = l3d1.colRange(0, 3) - l3d1.colRange(3, 6);
		double maxscore = 0;
		cv::Point3d mp1, mp2;
		for (int j = 0; j < vecIndx.cols; j++) {
			int idbase = vecIndx.at<int>(i, j);
			cv::Mat vec2 = vecs.row(idbase);
			if (std::acos(std::abs(vec1.dot(vec2)) / cv::norm(vec1) / cv::norm(vec2)) < param.growAng) 
			{
				//double score = 0.0;
				cv::Point3d p1(0, 0, 0);
				cv::Point3d p2(0, 0, 0);
				double score = nlopt_vecRefine3(imgInfo, spaceRec, reviseID.at<float>(i, 0), vec2, param, "par", p1, p2);
				if (score > maxscore) 
				{
					maxscore = score;
					mp1 = p1;
					mp2 = p2;
				}
			}
		}

		if (maxscore > 0) {
			cv::Mat l = (cv::Mat_<float>(1, 7) << maxscore, mp1.x, mp1.y, mp1.z, mp2.x, mp2.y, mp2.z);
			l.row(0).copyTo(reviseRes.row(reviseID.at<float>(i, 0)));
		}

	}
	return reviseRes;
}

cv::Mat lineGrow2(IMG_INFO imgInfo, SPACE_REC spaceRec, cv::Mat revised3D, cv::Mat reviseID, PARAMS param) {
	cv::Mat reviseline;
	cv::Mat baseline = revised3D;
	for (int i = 0; i < reviseID.rows; i++) {
		reviseline.push_back(spaceRec.lines3D.row(reviseID.at<float>(i, 0)));
	}

	cv::Mat searchMat = (baseline.colRange(0, 3) + baseline.colRange(3, 6)) / 2;
	cv::Mat queriesMat = (reviseline.colRange(0, 3) + reviseline.colRange(3, 6)) / 2;

	cv::Mat vecDist, vecIndx;
	cv::flann::KDTreeIndexParams  indexParams(6);
	cv::flann::Index kdtree(searchMat, indexParams);

	cv::flann::SearchParams knn_params(64);
	kdtree.knnSearch(queriesMat, vecIndx, vecDist, 20, knn_params);
	int count = 0;
	double threshold = 3.141592653589793 / 2 - param.growAng;

	cv::Mat reviseRes = cv::Mat::zeros(spaceRec.lines3D.rows, 7, CV_32FC1);
	for (int i = 0; i < reviseID.rows; i++) {
		cv::Mat l3d1 = spaceRec.lines3D.row(reviseID.at<float>(i, 0));
		cv::Mat vec1 = l3d1.colRange(0, 3) - l3d1.colRange(3, 6);
		double maxscore = 0;
		cv::Point3d mp1, mp2;
		for (int j = 0; j < vecIndx.cols; j++) {
			int idbase = vecIndx.at<int>(i, j);
			cv::Mat planeVec = baseline.row(idbase).colRange(6, baseline.cols);
			if (std::acos(std::abs(vec1.dot(planeVec)) / cv::norm(vec1) / cv::norm(planeVec)) > threshold) {
				//double score = 0.0;
				cv::Point3d p1(0, 0, 0);
				cv::Point3d p2(0, 0, 0);
				double score = nlopt_vecRefine3(imgInfo, spaceRec, reviseID.at<float>(i, 0), planeVec, param, "var", p1, p2);
				if (score > maxscore) {
					maxscore = score;
					mp1 = p1;
					mp2 = p2;
				}
			}
		}

		if (maxscore == 0) continue;
		count++;
		cv::Mat l = (cv::Mat_<float>(1, 7) << maxscore, mp1.x, mp1.y, mp1.z, mp2.x, mp2.y, mp2.z);
		l.row(0).copyTo(reviseRes.row(reviseID.at<float>(i, 0)));

	}
	std::cout << "lineGrow2  " << count << std::endl;
	return reviseRes;
}

cv::Mat extractMaxLine(cv::Mat line1, cv::Mat line2) {
	cv::Mat lineRef;
	cv::Mat	oneLine = cv::Mat(1, line1.cols, CV_32FC1);
	int cc = 0;
	for (int i = 0; i < line1.rows; i++) {
		double vs = 0;
		int id = -1;
		if (line1.at<float>(i, 0) > line2.at<float>(i, 0)) {
			id = 0;
			vs = line1.at<float>(i, 0);
		}
		else {
			id = 1;
			vs = line2.at<float>(i, 0);
		}

		if (vs == 0) continue;

		if (id == 0) {
			line1.row(i).colRange(1, line1.cols).copyTo(oneLine.row(0).colRange(1, line1.cols));
			oneLine.at<float>(0, 0) = i;

			lineRef.push_back(oneLine.row(0));
		}
		if (id == 1) {
			line2.row(i).colRange(1, line2.cols).copyTo(oneLine.row(0).colRange(1, line2.cols));
			oneLine.at<float>(0, 0) = i;
			lineRef.push_back(oneLine.row(0));
		}
	}

	return lineRef;
}

void medianPt(std::vector<cv::Vec3d>O1, std::vector<cv::Vec3d> O2, cv::Mat& st, cv::Mat& ed) {
	double max_len = -999;
	for (int i = 0; i < O1.size(); i++) {
		for (int j = 0; j < O2.size(); j++) {
			if (cv::norm(O1[i] - O2[j]) > max_len) {
				max_len = cv::norm(O1[i] - O2[j]);
				st.at<float>(0, 0) = O1[i][0];
				st.at<float>(0, 1) = O1[i][1];
				st.at<float>(0, 2) = O1[i][2];

				ed.at<float>(0, 0) = O2[j][0];
				ed.at<float>(0, 1) = O2[j][1];
				ed.at<float>(0, 2) = O2[j][2];
			}
		}
	}
}

void reconstructLine(std::vector<double> x, cv::Mat cen, cv::Mat ray1, cv::Mat ray2, IMG_INFO imgInfo, cv::Mat camids, cv::Mat lineids, int counters, cv::Mat clusters,
	double& err, cv::Mat& st, cv::Mat& ed) {
	cv::Mat p1 = cen + ray1 * x[0];
	cv::Mat p2 = cen + ray2 * x[1];
	cv::Mat vecp1p2 = p1 - p2;
	int cc = 0;
	err = 0;
	std::vector<cv::Vec3d> O1, O2;
	double threshold = 5.0 * 3.141592653589793 / 180.0;

	for (int i = 0; i < counters; i++) {
		if (clusters.at<float>(0, i) == 0) continue;

		cv::Mat centers = imgInfo.centers[camids.at<float>(0, i)];
		cv::Mat P = imgInfo.cameras[camids.at<float>(0, i)];
		cv::Mat line = imgInfo.lines[camids.at<float>(0, i)].row(lineids.at<float>(0, i));

		cv::Mat ray1, ray2, l1, l2;
		l1 = (cv::Mat_<float>(1, 3) << line.at<float>(0, 0), line.at<float>(0, 1), 1);
		l2 = (cv::Mat_<float>(1, 3) << line.at<float>(0, 2), line.at<float>(0, 3), 1);
		cv::Mat A = P.rowRange(0, 3).colRange(0, 3);
		cv::solve(A, l1.t(), ray1, cv::DECOMP_SVD);
		cv::solve(A, l2.t(), ray2, cv::DECOMP_SVD);

		//if (std::acos(std::abs(vecp1p2.dot(ray1.t())) / cv::norm(vecp1p2) / cv::norm(ray1)) < threshold) continue;
		//if (std::acos(std::abs(vecp1p2.dot(ray2.t())) / cv::norm(vecp1p2) / cv::norm(ray2)) < threshold) continue;

		cv::Vec3d oo1, oo2;
		double dist = 0.0;
		cv::Mat ll;
		cv::hconcat(p1, p2, ll);
		line2lineDist(ll, centers, centers + ray1.t(), oo1, oo2, dist);
		O1.push_back(oo1);
		err += dist;

		line2lineDist(ll, centers, centers + ray2.t(), oo1, oo2, dist);
		O2.push_back(oo1);
		err += dist;
		cc++;

	}

	if (cc == 0) return;
	medianPt(O1, O2, st, ed);
	err = err / cc;
}

void reconstruct(cv::Mat line3d, IMG_INFO imageInfo, cv::Mat camid, cv::Mat lineid, int counters, cv::Mat clusters, double& err, cv::Mat& st, cv::Mat& ed) {
	int cam1 = camid.at<float>(0, 0);
	cv::Mat cen1 = imageInfo.centers[cam1];

	cv::Mat ray1 = line3d.colRange(0, 3) - cen1;
	double d1 = cv::norm(ray1);
	ray1 = ray1 / d1;

	cv::Mat ray2 = line3d.colRange(3, 6) - cen1;
	double d2 = cv::norm(ray2);
	ray2 = ray2 / d2;

	std::vector<double> x = { d1,d2 };
	reconstructLine(x, cen1, ray1, ray2, imageInfo, camid, lineid, counters, clusters, err, st, ed);
}

cv::Mat multiReconstruction(cv::Mat lines3D, IMG_INFO imgInfo, SPACE_REC spaceRec, PARAMS param) {
	cv::Mat midp = (lines3D.colRange(1, 4) + lines3D.colRange(4, 7)) / 2;
	cv::Mat vecs = lines3D.colRange(1, 4) - lines3D.colRange(4, 7);

	cv::Mat vecDist, vecIndx;
	cv::flann::KDTreeIndexParams indexParams(6);
	cv::flann::Index kdtree(midp, indexParams);

	cv::flann::SearchParams knn_params(64);
	kdtree.knnSearch(midp, vecIndx, vecDist, param.colinearNum + 1, knn_params);

	vecIndx = vecIndx.colRange(1, vecIndx.cols);
	cv::Mat gorwMark = cv::Mat::zeros(lines3D.rows, 1, CV_32FC1);

	for (int i = 0; i < lines3D.rows; i++) {
		cv::Mat vec1 = vecs.row(i);
		int id1 = lines3D.at<float>(i, 0);
		int cc = 0;
		for (int j = 0; j < vecIndx.cols; j++) {
			int id2 = vecIndx.at<int>(i, j);
			cv::Mat vec2 = vecs.row(id2);
			if (std::acos(vec1.dot(vec2) / cv::norm(vec1) / cv::norm(vec2)) > param.maxAng) continue;
			cc++;
		}
		if (cc < param.longLineNum) continue;
		gorwMark.at<float>(i, 0) = 1;

		double err;
		cv::Mat sted = cv::Mat::zeros(1, 6, CV_32FC1);
		reconstruct(lines3D.row(i).colRange(1, lines3D.cols), imgInfo, spaceRec.camid.row(id1), spaceRec.lineid.row(id1),
			spaceRec.counters.at<float>(id1, 0), spaceRec.clusters.row(id1), err, sted.colRange(0, 3), sted.colRange(3, 6));
		sted.row(0).copyTo(lines3D.row(i).colRange(1, lines3D.cols));
		//cv::Mat s = cv::hconcat(

	}
	return lines3D;
}


void lineCluster(SfMManager* sfm, MergeProcess* mergeProc) {
	IMG_INFO imgInfo;
	SFM_INFO sfmInfo;
	ARR_INFO arrInfo;
	SPACE_REC spaceRec;
	PARAMS param;

	imgInfo.cameras.resize(sfm->camsNumber());
	imgInfo.centers.resize(sfm->camsNumber());
	imgInfo.lines.resize(sfm->camsNumber());
	for (int i = 0; i < sfm->camsNumber(); i++) {
		imgInfo.cameras[i] = sfm->iCameraMat(i);
		imgInfo.centers[i] = sfm->iCameraCenter(i);
		imgInfo.lines[i] = sfm->getImageLines(i);
	}



	size_t maxCounters = 0;
	size_t zeroCount = 0;
	for (int i = 0; i < mergeProc->camsID.size(); i++) {
		if (mergeProc->camsID[i].size() == 0) {
			zeroCount++;
			continue;
		}
		for (int j = 0; j < mergeProc->camsID[i].size(); j++) {
			if (mergeProc->camsID[i][j].size() > maxCounters) {
				maxCounters = mergeProc->camsID[i][j].size();
			}
		}
	}

	sfmInfo.camID.resize(mergeProc->camsID.size() - zeroCount);
	sfmInfo.lineID.resize(mergeProc->camsID.size() - zeroCount);
	sfmInfo.counters.resize(mergeProc->camsID.size() - zeroCount);
	sfmInfo.pairID = cv::Mat::zeros(mergeProc->camsID.size() - zeroCount, 2, CV_32FC1);

	int indx = 0;
	for (int i = 0; i < mergeProc->camsID.size(); i++) {
		if (mergeProc->camsID[i].size() == 0) continue;
		sfmInfo.counters[indx] = cv::Mat::zeros(mergeProc->camsID[i].size(), 1, CV_32FC1);
		sfmInfo.camID[indx] = cv::Mat::zeros(mergeProc->camsID[i].size(), maxCounters, CV_32FC1);
		sfmInfo.lineID[indx] = cv::Mat::zeros(mergeProc->camsID[i].size(), maxCounters, CV_32FC1);
		for (int j = 0; j < mergeProc->camsID[i].size(); j++) {
			sfmInfo.counters[indx].at<float>(j, 0) = mergeProc->camsID[i][j].size();
			for (int k = 0; k < mergeProc->camsID[i][j].size(); k++) {
				sfmInfo.camID[indx].at<float>(j, k) = mergeProc->camsID[i][j][k];
				sfmInfo.lineID[indx].at<float>(j, k) = mergeProc->lineID[i][j][k];
			}
		}
		sfmInfo.pairID.at<float>(indx, 0) = sfmInfo.camID[indx].at<float>(0, 0) + 1;
		sfmInfo.pairID.at<float>(indx, 1) = sfmInfo.camID[indx].at<float>(0, 1) + 1;
		indx++;
	}

	// Satge 2: normBuild
	//normalBuild(sfm->multiPoints, pairID, imgInfo, arrInfo);
	//normalBuild(outputFolder, sfmInfo, imgInfo, arrInfo);
	normalBuild(sfm->multiPoints, sfmInfo, imgInfo, arrInfo);

	// Satge 3: cluster
	callAdaptiveLineCluster(sfmInfo, imgInfo, arrInfo, spaceRec);

	param.dist3D = arrInfo.distmean;

	// Satge 4: colinear refine
	cv::Mat colinearRef = colinearRefine(imgInfo, spaceRec, param);

	// Satge 5: coplanar refine
	cv::Mat planeRef = coplanarRef(imgInfo, spaceRec, param);

	cv::Mat control3D, reviseID;

	// Satge space line
	divideSpaceLine(control3D, reviseID, colinearRef, planeRef);

	// Satge line grow
	cv::Mat revised3D2 = lineGrow1(imgInfo, spaceRec, control3D, reviseID, param);

	cv::Mat planeids;
	for (int i = 0; i < planeRef.rows; i++) {
		if (planeRef.at<float>(i, 0) > 0) {
			planeids.push_back(planeRef.row(i).colRange(1, planeRef.cols));
		}
	}
	// Satge 8
	cv::Mat revised3D3 = lineGrow2(imgInfo, spaceRec, planeids, reviseID, param);

	// Satge 9
	cv::Mat lineref = extractMaxLine(revised3D2, revised3D3);

	cv::Mat in;
	cv::vconcat(control3D, lineref, in);

	// Satge 10
	cv::Mat lines3D = multiReconstruction(in, imgInfo, spaceRec, param);

	// Satge 11
	cv::Mat outMat = cv::Mat();
	for (int i = 0; i < lines3D.rows; i++) {
		if (lines3D.at<float>(i, 0) != 0) {
			outMat.push_back(lines3D.row(i).colRange(1, 7));
		}
	}

	outObj(sfm->outFolder + "\\elsrpp.obj", outMat);
	std::string linesFile, camsFile, nameFile;
	linesFile = sfm->outFolder + "\\spaceLines.txt";
	camsFile = sfm->outFolder + "\\cams.txt";
	nameFile = sfm->outFolder + "\\camName.txt";


	std::ofstream nf;
	nf.open(nameFile);
	for (int i = 0; i < sfm->allImageNames()->size(); i++) {
		nf << sfm->allImageNames()->at(i) << "\n";
	}
	nf.close();


	std::ofstream lf; lf.open(linesFile);
	std::ofstream cf; cf.open(camsFile);
	for (int i = 0; i < lines3D.rows; i++) {
		lf << lines3D.at<float>(i, 1) << " " << lines3D.at<float>(i, 2) << " " << lines3D.at<float>(i, 3)
			<< " " << lines3D.at<float>(i, 4) << " " << lines3D.at<float>(i, 5) << " " << lines3D.at<float>(i, 6) << "\n";

		int k = lines3D.at<float>(i, 0);
		for (int j = 0; j < (int)spaceRec.counters.at<float>(k, 0); j++) {
			cf << spaceRec.camid.at<float>(k, j) << " ";
		}
		cf << "\n";
	}
	lf.close();
	cf.close();

}