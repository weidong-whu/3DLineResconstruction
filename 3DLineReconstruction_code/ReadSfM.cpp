#include"read_VisualSfM.h"
#include"BasicMath.h"
#include <stdio.h>
//#include <jpeglib.h>

unsigned long byteswap_ulong(unsigned long i) {
	unsigned int j;
	j = (i << 24);
	j += (i << 8) & 0x00FF0000;
	j += (i >> 8) & 0x0000FF00;
	j += (i >> 24);
	return j;
}

inline int Abs(int x) {
	return  (x ^ (x >> 31)) - (x >> 31);
}

unsigned short byteswap_ushort(unsigned short i) {
	unsigned short j;
	j = (i << 8);
	j += (i >> 8);
	return j;
}

bool get_image_size_without_decode_image(const char* file_path, int* width, int* height, int* file_size) {
	bool has_image_size = false;
	*height = -1;
	*width = -1;
	*file_size = -1;
	FILE* fp = fopen(file_path, "rb");
	if (fp == NULL)
		return has_image_size;
	struct stat st;
	char sigBuf[26];
	if (fstat(fileno(fp), &st) < 0) {
		fclose(fp);
		return has_image_size;
	}
	else {
		*file_size = st.st_size;
	}
	if (fread(&sigBuf, 26, 1, fp) < 1) {
		fclose(fp);
		return has_image_size;
	}
	char* png_signature = "\211PNG\r\n\032\n";
	char* ihdr_signature = "IHDR";
	char* gif87_signature = "GIF87a";
	char* gif89_signature = "GIF89a";
	char* jpeg_signature = "\377\330";
	char* bmp_signature = "BM";
	if ((*file_size >= 10) && (memcmp(sigBuf, gif87_signature, strlen(gif87_signature)) == 0 || memcmp(sigBuf, gif89_signature, strlen(gif89_signature)) == 0)) {
		// image type: gif
		unsigned short* size_info = (unsigned short*)(sigBuf + 6);
		*width = size_info[0];
		*height = size_info[1];
		has_image_size = true;
	}
	else if ((*file_size >= 24) && (memcmp(sigBuf, png_signature, strlen(png_signature)) == 0 && memcmp(sigBuf + 12, ihdr_signature, strlen(ihdr_signature)) == 0)) {
		// image type:   png
		unsigned long* size_info = (unsigned long*)(sigBuf + 16);
		*width = byteswap_ulong(size_info[0]);
		*height = byteswap_ulong(size_info[1]);
		has_image_size = true;
	}
	else if ((*file_size >= 16) && (memcmp(sigBuf, png_signature, strlen(png_signature)) == 0)) {
		// image type: old png
		unsigned long* size_info = (unsigned long*)(sigBuf + 8);
		*width = byteswap_ulong(size_info[0]);
		*height = byteswap_ulong(size_info[1]);
		has_image_size = true;
	}
	else if ((*file_size >= 2) && (memcmp(sigBuf, jpeg_signature, strlen(jpeg_signature)) == 0)) {
		// image type: jpeg
		fseek(fp, 0, SEEK_SET);
		char b = 0;
		fread(&sigBuf, 2, 1, fp);
		fread(&b, 1, 1, fp);
		int w = -1;
		int h = -1;
		while (b && ((unsigned char)b & 0xff) != 0xDA) {
			while (((unsigned char)b & 0xff) != 0xFF) {
				fread(&b, 1, 1, fp);
			}
			while (((unsigned char)b & 0xff) == 0xFF) {
				fread(&b, 1, 1, fp);
			}
			if (((unsigned char)b & 0xff) >= 0xC0 && ((unsigned char)b & 0xff) <= 0xC3) {
				fread(&sigBuf, 3, 1, fp);
				fread(&sigBuf, 4, 1, fp);
				unsigned short* size_info = (unsigned short*)(sigBuf);
				h = byteswap_ushort(size_info[0]);
				w = byteswap_ushort(size_info[1]);
				if (h != -1 && w != -1)	break;
			}
			else {
				unsigned short chunk_size = 0;
				fread(&chunk_size, 2, 1, fp);
				if (fseek(fp, byteswap_ushort(chunk_size) - 2, SEEK_CUR) != 0)
					break;
			}
			fread(&b, 1, 1, fp);
		}
		if (w != -1 && h != -1) {
			*width = w;
			*height = h;
		}
		has_image_size = true;
	}
	else if ((*file_size >= 26) && (memcmp(sigBuf, bmp_signature, strlen(bmp_signature)) == 0)) {
		// image type: bmp
		unsigned int header_size = (*(sigBuf + 14));
		if (header_size == 12) {
			unsigned short* size_info = (unsigned short*)(sigBuf + 18);
			*width = size_info[0];
			*height = size_info[1];
		}
		else if (header_size >= 40) {
			unsigned int* size_info = (unsigned int*)(sigBuf + 18);
			*width = size_info[0];
			*height = Abs((size_info[1]));
		}
		has_image_size = true;
	}
	else if (*file_size >= 2) {
		// image type: ico
		fseek(fp, 0, SEEK_SET);
		unsigned short format = -1;
		unsigned short reserved = -1;
		fread(&reserved, 2, 1, fp);
		fread(&format, 2, 1, fp);
		if (reserved == 0 && format == 1) {
			unsigned short num = -1;
			fread(&num, 2, 1, fp);
			if (num > 1) {
				printf("this is a muti-ico file.");
			}
			else {
				char w = 0, h = 0;
				fread(&w, 1, 1, fp);
				fread(&h, 1, 1, fp);
				*width = (int)((unsigned char)w & 0xff);
				*height = (int)((unsigned char)h & 0xff);
			}
		}
		has_image_size = true;
	}
	if (fp != NULL)
		fclose(fp);
	return has_image_size;
}

void analysis_match(cv::Mat mscores, cv::Mat& imidx_Mf, int knn, int minconnect) {

	int ind;


	cv::Mat used = cv::Mat::zeros(mscores.rows, mscores.cols, CV_16SC1);

	cv::Mat per_pair(1, 2, CV_32FC1);

	for (int i = 0; i < mscores.cols; i++) {
		cv::Mat sortID;
		cv::sortIdx(mscores.row(i), sortID, cv::SORT_DESCENDING);
		int cur = 0;
		for (int j = 0; j < knn; j++) {
			ind = sortID.at<int>(0, j);

			if (used.at<ushort>(ind, i) == 1)
				continue;

			if (mscores.at<ushort>(i, ind) < minconnect)
				break;

			per_pair.at<float>(0, 0) = i;
			per_pair.at<float>(0, 1) = ind;
			imidx_Mf.push_back(per_pair);

			used.at<ushort>(ind, i) = 1;
			cur++;
		}
	}
}

void analysis_pair(cv::Mat imidx_Mf, cv::Mat mscores, cv::Mat& pairs_double_, int mincommon) {
	int ind1, ind2, ind3, ind4;

	for (int i = 0; i < mscores.cols; i++)
		mscores.at<float>(i, i) = 9999;

	cv::Mat pairs_double = cv::Mat::zeros(imidx_Mf.rows, imidx_Mf.rows, CV_16UC1);

	for (int i = 0; i < imidx_Mf.rows; i++) {
		ind1 = imidx_Mf.at<float>(i, 0);
		ind2 = imidx_Mf.at<float>(i, 1);

		for (int j = i + 1; j < imidx_Mf.rows; j++) {

			ind3 = imidx_Mf.at<float>(j, 0);
			ind4 = imidx_Mf.at<float>(j, 1);

			if (mscores.at<float>(ind1, ind3) < mincommon ||
				mscores.at<float>(ind1, ind4) < mincommon ||
				mscores.at<float>(ind2, ind3) < mincommon ||
				mscores.at<float>(ind2, ind4) < mincommon)
				continue;

			pairs_double.at<ushort>(i, j) = 1;
			pairs_double.at<ushort>(j, i) = 1;

		}

	}

	pairs_double.copyTo(pairs_double_);

}

struct point_info {
	float xx;
	float yy;
	uint camid;
};

double line_line_dis(float* p1, float* vec1, float* p2, float* vec2) {
	float vec_cross[3], vec_pt[3];

	cross_v3(vec1, vec2, vec_cross);
	double normv = norm_v3(vec_cross);

	vec_cross[0] = vec_cross[0] / normv;
	vec_cross[1] = vec_cross[1] / normv;
	vec_cross[2] = vec_cross[2] / normv;

	vec_pt[0] = p1[0] - p2[0];
	vec_pt[1] = p1[1] - p2[1];
	vec_pt[2] = p1[2] - p2[2];

	//printf("-----------------------------\n");
	//printf("%f %f %f\n",vec_pt[0],vec_pt[1],vec_pt[2]);
	//printf("%f %f %f\n",vec_cross[0],vec_cross[1],vec_cross[2]);


	float r = dot_v3(vec_cross, vec_pt);

	//if (r < 0)
		//r = -r;
	//printf("%f\n-----------------------------\n",r);
	return r;

}

void line2lineerr(std::vector<point_info>& points, std::vector<cv::Mat>& Ms, std::vector<cv::Mat>& Cs, std::vector<cv::Mat*>& errs) {
	int N = points.size();
	point_info pi1, pi2;
	float* cen1, * cen2, * M1, * M2;

	float p1[2], p2[2];

	float v1[3], v2[3];

	float minerr = 99999999;
	float trueminerr = 0;
	for (int i = 0; i < N; i++) {
		pi1 = points[i];

		cen1 = (float*)Cs[pi1.camid].data;
		M1 = (float*)Ms[pi1.camid].data;

		p1[0] = pi1.xx;
		p1[1] = pi1.yy;

		solverAxb(M1, p1, v1);

		for (int j = i + 1; j < N; j++) {
			pi2 = points[j];
			cen2 = (float*)Cs[pi2.camid].data;
			M2 = (float*)Ms[pi2.camid].data;

			p2[0] = pi2.xx;
			p2[1] = pi2.yy;

			solverAxb(M2, p2, v2);

			float err = line_line_dis(cen1, v1, cen2, v2);

			if (isnan(err))
				continue;
			errs[pi1.camid]->push_back(err);
		}
	}


}

void point2lineerr(float* pt3, std::vector<point_info>& points, std::vector<cv::Mat>& Ms, std::vector<cv::Mat>& Cs, std::vector<cv::Mat*>& errs) {
	int N = points.size();
	point_info pi1, pi2;
	float* cen1, * cen2, * M1, * M2;

	float p1[2], v1[3], pt[3];



	float err = 99999999;
	float trueminerr = 0;
	for (int i = 0; i < N; i++) {
		pi1 = points[i];

		cen1 = (float*)Cs[pi1.camid].data;
		M1 = (float*)Ms[pi1.camid].data;

		p1[0] = pi1.xx;
		p1[1] = pi1.yy;

		solverAxb(M1, p1, v1);

		pt[0] = cen1[0] + 5 * v1[0];
		pt[1] = cen1[1] + 5 * v1[1];
		pt[2] = cen1[2] + 5 * v1[2];
		err = point_2_line_dis_3D(pt3, cen1, pt);
		if (isnan(err))
			continue;
		errs[pi1.camid]->push_back(err);
	}


}




int read_VisualSfM(
	SfMManager* sfm,
	MatchManager* match,
	int knn_image,
	int connectionNum, bool fromcolmap) {

	std::string inputFolder = sfm->inputFolder();
	std::string nvmFile = inputFolder + sfm->nvmFile();
	// check if NVM file exists


	// read NVM file
	std::ifstream nvm_file;
	nvm_file.open(nvmFile.c_str());

	std::string nvm_line;
	std::getline(nvm_file, nvm_line); // ignore first line...
	std::getline(nvm_file, nvm_line); // ignore second line...

	// read number of images
	std::getline(nvm_file, nvm_line);
	std::stringstream nvm_stream(nvm_line);
	unsigned int num_cams;
	nvm_stream >> num_cams;

	std::vector<point_info> point_cluster;

	if (num_cams == 0) {
		std::cerr << "No aligned cameras in NVM file!" << std::endl;
		return -1;
	}


	// read camera data (sequentially)
	cv::Mat imidx_Mf;

	cv::Mat R(3, 3, CV_32FC1);
	cv::Mat C(3, 1, CV_32FC1);

	std::vector<cv::Mat> Ms;
	std::vector<cv::Mat> Cs;
	std::vector<cv::Mat> cams;

	std::vector<cv::Mat*>errs(num_cams);
	for (int i = 0; i < num_cams; i++)
		errs[i] = new cv::Mat;

	for (unsigned int i = 0; i < num_cams; ++i) {
		std::getline(nvm_file, nvm_line);

		// image filename
		std::string filename;

		// focal_length,quaternion,center,distortion
		double focal_length, qx, qy, qz, qw;
		double Cx, Cy, Cz, dist;

		nvm_stream.str("");
		nvm_stream.clear();
		nvm_stream.str(nvm_line);
		nvm_stream >> filename >> focal_length >> qw >> qx >> qy >> qz;
		nvm_stream >> Cx >> Cy >> Cz >> dist;

		sfm->addImageNames(filename);
		sfm->addCamsFocals(focal_length);

		// rotation amd translation
		R.at<float>(0, 0) = 1.0 - 2.0 * qy * qy - 2.0 * qz * qz;
		R.at<float>(0, 1) = 2.0 * qx * qy - 2.0 * qz * qw;
		R.at<float>(0, 2) = 2.0 * qx * qz + 2.0 * qy * qw;

		R.at<float>(1, 0) = 2.0 * qx * qy + 2.0 * qz * qw;
		R.at<float>(1, 1) = 1.0 - 2.0 * qx * qx - 2.0 * qz * qz;
		R.at<float>(1, 2) = 2.0 * qy * qz - 2.0 * qx * qw;

		R.at<float>(2, 0) = 2.0 * qx * qz - 2.0 * qy * qw;
		R.at<float>(2, 1) = 2.0 * qy * qz + 2.0 * qx * qw;
		R.at<float>(2, 2) = 1.0 - 2.0 * qx * qx - 2.0 * qy * qy;

		C.at<float>(0, 0) = Cx;
		C.at<float>(1, 0) = Cy;
		C.at<float>(2, 0) = Cz;

		cv::Mat t = -R * C;

		cv::Mat Rt;
		cv::hconcat(R, t, Rt);

		sfm->addCamsRT(Rt);

		sfm->addCamsCenter(C.t());


		// for error construction
		cv::Mat KK = cv::Mat::zeros(3, 3, CV_32FC1);

		KK.at<float>(0, 0) = focal_length;
		KK.at<float>(1, 1) = focal_length;
		KK.at<float>(2, 2) = 1;


		int aw, ah, as;
		get_image_size_without_decode_image((inputFolder + "/" + filename).c_str(), &aw, &ah, &as);
		
		
		KK.at<float>(0, 2) = aw / 2.0;
		KK.at<float>(1, 2) = ah / 2.0;

		Cs.push_back(C.t());
		cv::Mat cam = KK * Rt;
		Ms.push_back(cam.rowRange(0, 3).colRange(0, 3).t());
		cams.push_back(cam);

	}

	std::vector<float>xsize;
	std::vector<float>ysize;

	sfm->iniImageSize();
	sfm->iniCameraSize();
	// read image size
	for (int i = 0; i < num_cams; i++) {

		printf("read image %d\n", i);

		int aw, ah, as;
		get_image_size_without_decode_image((inputFolder + "/" + sfm->iImageNames(i)).c_str(), &aw, &ah, &as);

		xsize.push_back((float)aw / 2.0);
		ysize.push_back((float)ah / 2.0);

		sfm->addImSize(ah, aw, i);
		sfm->addCameraBySize(ah, aw, i);
	}

	//std::ofstream ofs;
	//ofs.open(inputFolder + "//output//multiPoints.txt", std::ios::out);

	// read number of images
	std::getline(nvm_file, nvm_line); // ignore line...
	std::getline(nvm_file, nvm_line);
	nvm_stream.str("");
	nvm_stream.clear();
	nvm_stream.str(nvm_line);
	unsigned int num_points;
	nvm_stream >> num_points;

	// read features (for image similarity calculation)
	cv::Mat pos3D(1, 3, CV_32FC1);

	std::vector<cv::Mat>points2D_N(num_cams);
	cv::Mat mscores = cv::Mat::zeros(num_cams, num_cams, CV_16UC1);

	std::vector<uint>cam_IDs;
	sfm->initialImagePoints();
	float depth;
	cv::Mat depthes;

	float pt3d[3];

	for (unsigned int i = 0; i < num_points; ++i) {
		// 3D position
		std::getline(nvm_file, nvm_line);
		std::istringstream iss_point3D(nvm_line);
		double px, py, pz, colR, colG, colB;
		iss_point3D >> px >> py >> pz;
		iss_point3D >> colR >> colG >> colB;

		pos3D.at<float>(0, 0) = px;
		pos3D.at<float>(0, 1) = py;
		pos3D.at<float>(0, 2) = pz;

		sfm->add_points_space3D(pos3D);
		//  num of views for each 3D points
		unsigned int num_views;
		iss_point3D >> num_views;

		unsigned int camID, siftID;
		float posX, posY;

		//std::string points;
		cv::Mat points;
		for (unsigned int j = 0; j < num_views; ++j) {
			iss_point3D >> camID >> siftID;
			iss_point3D >> posX >> posY;

			cam_IDs.push_back(camID);



			if (fromcolmap == 1) {
				posX = posX - xsize[camID];
				posY = posY - ysize[camID];
			}

		

			pos3D.at<float>(0, 0) = posX;
			pos3D.at<float>(0, 1) = posY;
			pos3D.at<float>(0, 2) = i + 1;


			//std::string subline = std::to_string(camID) + " " + std::to_string(posX + xsize[camID]) + " " + std::to_string(posY + ysize[camID]) + ",";
			//points = points + subline;
			cv::Mat subline = (cv::Mat_<float>(1, 3) << camID, posX + xsize[camID], posY + ysize[camID]);
			points.push_back(subline);

			sfm->addImagePoints(camID, pos3D);

			float* CM = sfm->iCameraMatPtr(camID);

			sfm->allPoints3D()->at<float>();

			point_cluster.push_back(point_info{ posX,posY,camID });
		}

		//ofs << points << std::endl;
		sfm->multiPoints.push_back(points);

		pt3d[0] = px; pt3d[1] = py; pt3d[2] = pz;

		if (1)
			point2lineerr(pt3d, point_cluster, Ms, Cs, errs);
		else
			line2lineerr(point_cluster, Ms, Cs, errs);


		point_cluster.clear();

		for (int ii = 0; ii < cam_IDs.size(); ii++)
			for (int jj = ii + 1; jj < cam_IDs.size(); jj++) {
				mscores.at<ushort>(cam_IDs[ii], cam_IDs[jj])++;
				mscores.at<ushort>(cam_IDs[jj], cam_IDs[ii])++;
			}

		cam_IDs.clear();
	}



	//ofs.close();
	int nbins = sfm->bins;

	cv::Mat generalized_err;
	cv::Mat mean_std(errs.size(), 2, CV_32FC1);
	cv::Mat max_error(errs.size(), 1, CV_32FC1);
	for (int mm = 0; mm < errs.size(); mm++) {
		cv::Mat err_sub = *errs[mm];

		if (err_sub.rows < 5)
			continue;

		cv::Mat std_, mean_;
		cv::meanStdDev(err_sub, mean_, std_);

		float mmax_thre = mean_.at<double>(0, 0) + 3 * std_.at<double>(0, 0);
		float mmin_thre = mean_.at<double>(0, 0) - 3 * std_.at<double>(0, 0);

		cv::Mat new_sub_err;
		float maxv = 0;
		for (int i = 0; i < err_sub.rows; i++) {
			if (err_sub.at<float>(i, 0) > mmax_thre || err_sub.at<float>(i, 0) < mmin_thre)
				continue;
			new_sub_err.push_back(err_sub.at<float>(i, 0));

			if (abs(err_sub.at<float>(i, 0)) > maxv)
				maxv = abs(err_sub.at<float>(i, 0));
		}

		cv::meanStdDev(new_sub_err, mean_, std_);
		float mean_new = mean_.at<double>(0, 0);
		float std_new = std_.at<double>(0, 0);

		mean_std.at<float>(mm, 0) = mean_new;
		mean_std.at<float>(mm, 1) = std_new;

		max_error.at<float>(mm, 0) = maxv;

		new_sub_err = (new_sub_err - mean_new) / std_new;

		generalized_err.push_back(new_sub_err);

	}

	std::vector<uint>counts(nbins, 0);
	float interval = 3.0 / nbins;
	for (int i = 0; i < generalized_err.rows; i++) {
		int ibin = std::floor(std::abs(generalized_err.at<float>(i, 0)) / interval);
		if (ibin >= nbins)ibin = nbins - 1;
		if (ibin < 0)continue;
		counts[ibin] = counts[ibin] + 1;
	}

	cv::Mat prior = cv::Mat::zeros(nbins, nbins, CV_32FC1);
	//write2txt((ushort*)mscores.data, mscores.rows, mscores.cols, sfm->inputFolder() + "\\scores.txt");

	for (int i = 0; i < nbins; i++) {
		float sum = 0;

		for (int j = 0; j < i; j++)
			sum = sum + counts[j];

		for (int j = 0; j < i; j++)
			prior.at<float>(i, j) = counts[j] / sum;


		float sub = nbins - i;
		for (int j = i; j < nbins; j++)
			prior.at<float>(i, j) = 1.0 / sub;
	}

	sfm->addTrainPrioi(prior, mean_std, max_error);
	sfm->interval = interval;


	//write2txt((float*)prior.data, prior.rows, prior.cols, "prior.txt");

	//printf("write");
	//getchar();
	//analysis match pair
	//sfm->write_points_space3D();
	match->addConnectScore(mscores);
	//write2txt((ushort*)mscores.data, mscores.rows, mscores.cols, inputFolder + "//output//mscores.txt");

	//analysis_match(mscores, imidx_Mf, knn_image, connectionNum);
	//match->initializeM(imidx_Mf);
	nvm_file.close();

	return 1;
}


int read_VisualSfM_PS(
	SfMManager* sfm,
	MatchManager* match,
	int knn_image,
	int connectionNum, bool fromcolmap) {
	return 1;
}