#include "PairMatch.h"
#include "BasicMath.h"
#include "IO.h"
#include "Parameters.h"

void skwe_mat3(cv::Mat M1f, cv::Mat &skew_Mf)
{
    cv::Mat skew_Mf_ = cv::Mat(3, 3, CV_32FC1);

    skew_Mf_.at<float>(0, 0) = 0;
    skew_Mf_.at<float>(0, 1) = -M1f.at<float>(2, 0);
    skew_Mf_.at<float>(0, 2) = M1f.at<float>(1, 0);

    skew_Mf_.at<float>(1, 0) = M1f.at<float>(2, 0);
    skew_Mf_.at<float>(1, 1) = 0;
    skew_Mf_.at<float>(1, 2) = -M1f.at<float>(0, 0);

    skew_Mf_.at<float>(2, 0) = -M1f.at<float>(1, 0);
    skew_Mf_.at<float>(2, 1) = M1f.at<float>(0, 0);
    skew_Mf_.at<float>(2, 2) = 0;

    skew_Mf_.copyTo(skew_Mf);
}

void computeFAE(cv::Mat P1, cv::Mat P2, cv::Mat &F_Mf, cv::Mat &AE_Mf)
{

    cv::Mat P1_inv;
    cv::invert(P1, P1_inv, cv::DECOMP_SVD);

    cv::Mat w, u, vt, e;
    // svd  Decomp for null space
    cv::SVD::compute(P1, w, u, vt, cv::SVD::FULL_UV);
    // cv::SVDecomp(P1, w, u, vt);

    cv::Mat C1 = vt.rowRange(3, 4).clone().t();

    cv::Mat x = P2 * C1;

    skwe_mat3(x, e);

    cv::Mat F = e * (P2 * P1_inv);
    F.copyTo(F_Mf);

    // AE
    cv::SVD::compute(F, w, u, vt, cv::SVD::FULL_UV);
    cv::Mat eprime = u.colRange(2, 3).clone();
    cv::Mat e_prime_cross;
    skwe_mat3(eprime, e_prime_cross);

    cv::Mat A = e_prime_cross * F;
    cv::Mat AE = cv::Mat::zeros(1, 12, CV_32FC1);

    float *a = (float *)A.data;
    float *eprime_ = (float *)eprime.data;

    for (int mm = 0; mm < 9; mm++)
        AE.at<float>(0, mm) = a[mm];
    for (int mm = 0; mm < 3; mm++)
        AE.at<float>(0, mm + 9) = eprime_[mm];

    AE.copyTo(AE_Mf);
}

void get3DLine(cv::Mat CM, cv::Mat CN, cv::Mat F, float *l1, float *l2, float *p_3d_1, float *p_3d_2)
{
    cv::Mat pt1 = cv::Mat(3, 1, CV_32FC1);
    cv::Mat pt2 = cv::Mat(3, 1, CV_32FC1);
    cv::Mat pt3 = cv::Mat(3, 1, CV_32FC1);
    cv::Mat pt4 = cv::Mat(3, 1, CV_32FC1);

    pt1.at<float>(0, 0) = l1[0];
    pt1.at<float>(1, 0) = l1[1];
    pt1.at<float>(2, 0) = 1;

    cv::Mat ep1 = F * pt1;

    pt2.at<float>(0, 0) = l1[2];
    pt2.at<float>(1, 0) = l1[3];
    pt2.at<float>(2, 0) = 1;

    cv::Mat ep2 = F * pt2;

    pt3.at<float>(0, 0) = l2[0];
    pt3.at<float>(1, 0) = l2[1];
    pt3.at<float>(2, 0) = 1;

    pt4.at<float>(0, 0) = l2[2];
    pt4.at<float>(1, 0) = l2[3];
    pt4.at<float>(2, 0) = 1;

    cv::Mat l2f = pt3.cross(pt4);

    cv::Mat pt5 = l2f.cross(ep1);
    cv::Mat pt6 = l2f.cross(ep2);

    pt5 = pt5 / pt5.at<float>(2, 0);
    pt6 = pt6 / pt6.at<float>(2, 0);

    std::vector<cv::Point2f> pt1Vec;
    std::vector<cv::Point2f> pt2Vec;

    pt1Vec.push_back(cv::Point2f(pt1.at<float>(0, 0), pt1.at<float>(1, 0)));
    pt1Vec.push_back(cv::Point2f(pt2.at<float>(0, 0), pt2.at<float>(1, 0)));

    pt2Vec.push_back(cv::Point2f(pt5.at<float>(0, 0), pt5.at<float>(1, 0)));
    pt2Vec.push_back(cv::Point2f(pt6.at<float>(0, 0), pt6.at<float>(1, 0)));

    cv::Mat pnt3D;

    cv::triangulatePoints(CM, CN, pt1Vec, pt2Vec, pnt3D);

    p_3d_1[0] = pnt3D.at<float>(0, 0);
    p_3d_1[1] = pnt3D.at<float>(1, 0);
    p_3d_1[2] = pnt3D.at<float>(2, 0);
    p_3d_1[3] = pnt3D.at<float>(3, 0);

    p_3d_2[0] = pnt3D.at<float>(0, 1);
    p_3d_2[1] = pnt3D.at<float>(1, 1);
    p_3d_2[2] = pnt3D.at<float>(2, 1);
    p_3d_2[3] = pnt3D.at<float>(3, 1);
}

// revise0324 the line shold not be induced by its self junction
// record the plane and eliminate those line that induced by weak plane
void guidedMatching(MatchManager *match, int mid1, int mid2, int task_id, float *lines1, float *lines_range,
                    int *lines1_knn, float *lines2, cv::Mat line2_map, int lsize1, int lsize2, float *homos,
                    float *planes, ushort *planes_line_id, int hom_size, cv::Mat CM, cv::Mat CN, cv::Mat F, float *M1,
                    float *C1, float *M2, float *C2, int knn_num, float dist, int imr, int imc)
{

    float *l1, *l2, *H, *P, *F_ptr, *CM_ptr, *CN_ptr;
    float p1[3], p2[3], p1_[3], p2_[3], vec1[2], vec2[2], p3[3], p4[3];
    float epl1[3], epl2[3], l_f1[3], l_f2[3];
    float epp1[3], epp2[3];
    int addvec[2];
    float p_3d_1[4], p_3d_2[4];
    float pt3d1[3], pt3d2[3];

    cv::Mat match_cell(1, 4, CV_16UC1);
    cv::Mat pt3Ds = cv::Mat::zeros(1, 6, CV_32FC1);
    cv::Mat pt3Ds_tempall;
    cv::Mat matchs_tempall;
    std::vector<int> xx, yy;

    F_ptr = (float *)F.data;
    CM_ptr = (float *)CM.data;
    CN_ptr = (float *)CN.data;

    int *id_candidate = new int[knn_num];
    float *score_candidate = new float[knn_num];

    float *score_plane = new float[hom_size];
    for (int i = 0; i < hom_size; i++)
        score_plane[i] = 0;

    float *id_plane = new float[knn_num];
    std::vector<int> match_plane_id;

    std::vector<float> score_match;

    int allocate_num = 500;
    int *searched_ID = new int[allocate_num];
    int searched_size;
    bool buffer_control = 0;

    int sx, sy, lid, buffer, l1_id, homos_id, lastID, bestID, candidate_size = 0;
    float bestDis, maxdis, dis1, dis2, max_depth1, min_depth1, max_depth2, min_depth2, pjl_length, vec_cos;

    buffer = round(dist + 0.5);

    for (int i = 0; i < lsize1; i++)
    {
        l1_id = i;
        l1 = lines1 + l1_id * 7;

        for (int mm = 0; mm < knn_num; mm++)
        {
            id_candidate[mm] = 0;
            score_candidate[mm] = 0;
        }

        candidate_size = 0;

        min_depth1 = lines_range[i * 4];
        max_depth1 = lines_range[i * 4 + 1];

        min_depth2 = lines_range[i * 4 + 2];
        max_depth2 = lines_range[i * 4 + 3];

        for (int j = 0; j < knn_num; j++)
        {
            lastID = 0;
            bestID = 0;
            bestDis = 0;

            homos_id = lines1_knn[i * knn_num + j];
            H = homos + homos_id * 11 + 2;
            P = planes + homos_id * 4;

            p1[0] = l1[0];
            p1[1] = l1[1];
            p1[2] = 1;

            p2[0] = l1[2];
            p2[1] = l1[3];
            p2[2] = 1;

            // mapping with homography
            mult_3_3_3(H, p1, p1_);
            mult_3_3_3(H, p2, p2_);
            norm_by_v3(p1_);
            norm_by_v3(p2_);

            cross_v3(p1_, p2_, l_f1);
            //
            vec1[0] = p2_[0] - p1_[0];
            vec1[1] = p2_[1] - p1_[1];
            pjl_length = norm_v2(vec1);

            if (pjl_length > 3 * l1[6] || pjl_length < l1[6] / 3)
                continue;

            if (vec1[0] > vec1[1])
            {
                addvec[0] = 0;
                addvec[1] = 1;
            }
            else
            {
                addvec[0] = 1;
                addvec[1] = 0;
            }

            Bresenham(round(p1_[0]), round(p1_[1]), round(p2_[0]), round(p2_[1]), xx, yy);

            searched_size = 0;
            for (int mm = 0; mm < xx.size(); mm += 6)
                for (int k = -buffer; k <= buffer; k++)
                {
                    buffer_control = !buffer_control;
                    if (buffer_control)
                        continue;

                    sx = xx[mm] + k * addvec[0];
                    sy = yy[mm] + k * addvec[1];

                    if (sx <= 0 || sx >= imc || sy <= 0 || sy >= imr)
                        continue;

                    lid = line2_map.at<int>(sy, sx) - 1;
                    if (lid < 0 || lid >= lsize2 || lastID == lid || lid == bestID)
                        continue;
                    lastID = lid;

                    if (ID_in_array(searched_ID, searched_size, lid, allocate_num))
                        continue;

                    if (searched_size < allocate_num - 1)
                    {
                        searched_ID[searched_size] = lid;
                        searched_size++;
                    }

                    l2 = lines2 + lid * 7;

                    p3[0] = l2[0];
                    p3[1] = l2[1];
                    p3[2] = 1;

                    p4[0] = l2[2];
                    p4[1] = l2[3];
                    p4[2] = 1;

                    vec2[0] = p4[0] - p3[0];
                    vec2[1] = p4[1] - p3[1];

                    cross_v3(p3, p4, l_f2);
                    // check direction
                    // make sure they are in the same directiom
                    if (vec1[0] * vec2[0] + vec1[1] * vec2[1] < 0)
                        continue;

                    // check epipolar line
                    mult_3_3_3(F_ptr, p1, epl1);
                    mult_3_3_3(F_ptr, p2, epl2);

                    cross_v3(l_f2, epl1, epp1);
                    cross_v3(l_f2, epl2, epp2);
                    norm_by_v3(epp1);
                    norm_by_v3(epp2);

                    // two lines intersection;
                    if (!twoLines_intersec(epp1, epp2, p3, p4, LINE_OVERLAP))
                        continue;

                    // distance check
                    dis1 = point_2_line_dis(p3, l_f1);
                    if (dis1 > dist)
                        continue;

                    dis2 = point_2_line_dis(p4, l_f1);
                    if (dis2 > dist)
                        continue;

                    // intersection check
                    if (!twoLines_intersec(p3, p4, p1_, p2_, LINE_OVERLAP))
                        continue;

                    // get the 3D line and depth check
                    /*
                    if (!tringulate3Dline(
                        CN_ptr, l2,
                        M1, C1, l1,
                        p_3d_1, p_3d_2))
                        continue;
                    */
                    if (!tringulate3Dline(CN_ptr, l2, M1, C1, l1, p_3d_1, p_3d_2))
                        continue;
                    ;

                    // line to plane distance check

                    // if (point_plane_dis3d(p_3d_1, P) > space_threshold)
                    // continue;

                    // if (point_plane_dis3d(p_3d_2, P) > space_threshold)
                    // continue;

                    // revise0322 plane angle check
                    // revise0425
                    // float best_line_plane_sin = linePlaneSin(p_3d_1, p_3d_2, P);
                    // if (best_line_plane_sin > 0.1736)//10 deg
                    // continue;

                    // revise0322 eliminate norm_by_v4(p_3d_1);

                    if (1)
                    {
                        float w1 = CM_ptr[11] + CM_ptr[8] * p_3d_1[0] + CM_ptr[9] * p_3d_1[1] + CM_ptr[10] * p_3d_1[2];

                        if (w1 < min_depth1 || w1 > max_depth1)
                            continue;

                        // norm_by_v4(p_3d_2);
                        float w2 = CM_ptr[11] + CM_ptr[8] * p_3d_2[0] + CM_ptr[9] * p_3d_2[1] + CM_ptr[10] * p_3d_2[2];

                        if (w2 < min_depth2 || w2 > max_depth2)
                            continue;
                    }

                    maxdis = exp(-max_2(dis2, dis1) / (2 * dist));

                    if (bestDis >= maxdis)
                        continue;

                    bestID = lid;
                    bestDis = maxdis;
                }

            if (bestID == 0)
                continue;

            bool founded = 0;
            for (int mm = 0; mm < candidate_size; mm++)
                if (id_candidate[mm] == bestID)
                {
                    score_candidate[mm] += bestDis;
                    id_plane[mm] = homos_id;
                    founded = 1;

                    break;
                }

            if (!founded)
            {
                score_candidate[candidate_size] = bestDis;
                id_candidate[candidate_size] = bestID;
                id_plane[candidate_size] = homos_id;
                candidate_size++;
            }
        }

        if (candidate_size == 0)
            continue;

        // find max score
        maxdis = 0;
        int min_id = -1;
        int min_homid = -1;

        for (int mm = 0; mm < candidate_size; mm++)
        {
            if (score_candidate[mm] > maxdis)
            {
                maxdis = score_candidate[mm];
                min_id = id_candidate[mm];
                min_homid = id_plane[mm];
            }
        }

        l2 = lines2 + min_id * 7;

        if (!tringulate3Dline(CN_ptr, l2, M1, C1, l1, (float *)pt3Ds.data, (float *)pt3Ds.data + 3))
            continue;

        match_cell.at<ushort>(0, 0) = i;
        match_cell.at<ushort>(0, 1) = mid1;
        match_cell.at<ushort>(0, 2) = min_id;
        match_cell.at<ushort>(0, 3) = mid2;

        matchs_tempall.push_back(match_cell);
        match_plane_id.push_back(min_homid);
        pt3Ds_tempall.push_back(pt3Ds);

        score_match.push_back(maxdis);

        score_plane[min_homid]++;
    }

    // eliminate contractory junc
    ushort *planes_line_i = planes_line_id;
    ushort *planes_line_j;
    bool is_best = true;

    for (int i = 0; i < hom_size; i++)
    {
        planes_line_i = planes_line_i + 4;

        if (score_plane[i] <= 0)
            continue;

        is_best = true;
        for (int j = i + 1; j < hom_size; j++)
        {
            planes_line_j = planes_line_id + j * 4;

            if (!((planes_line_i[0] == planes_line_j[0] && planes_line_i[1] != planes_line_j[1]) ||
                  (planes_line_i[0] == planes_line_j[2] && planes_line_i[1] != planes_line_j[3]) ||
                  (planes_line_i[2] == planes_line_j[0] && planes_line_i[3] != planes_line_j[1]) ||
                  (planes_line_i[2] == planes_line_j[2] && planes_line_i[3] != planes_line_j[3])))
                continue;
            /*
            printf("%d %d %d %d\n %d %d %d %d\n......\n",
                planes_line_i[0], planes_line_i[1], planes_line_i[2], planes_line_i[3],
                planes_line_j[0], planes_line_j[1], planes_line_j[2], planes_line_j[3]);
            getchar();
            */

            if (score_plane[i] >= score_plane[j])
            {
                score_plane[j] = 0;
            }
            else
            {
                score_plane[i] = 0;
                break;
            }
        }
    }

    for (int i = 0; i < match_plane_id.size(); i++)
    {
        if (score_plane[match_plane_id[i]] == 0)
            continue;

        if (score_match[i] == 0)
            continue;

        is_best = true;

        for (int j = i + 1; j < match_plane_id.size(); j++)
        {
            if (matchs_tempall.at<ushort>(i, 2) != matchs_tempall.at<ushort>(j, 2))
                continue;

            if (score_match[i] >= score_match[j])
                score_match[j] = 0;
            else
            {
                is_best = false;
                break;
            }
        }

        if (is_best == false)
            continue;

        match->addLine3D(pt3Ds_tempall.row(i).clone(), task_id);
        match->pushMatch(matchs_tempall.row(i).clone(), task_id);
    }

    delete[] id_candidate;
    delete[] score_candidate;
    delete[] searched_ID;
    delete[] score_plane;
    delete[] id_plane;
}

void line2KDtree(cv::Mat lines_Mf, cv::Mat homo_Mf, cv::Mat *inter_knn_Mi, int support_H_num)
{
    // construct kdtree
    cv::flann::Index flannIndex(homo_Mf.colRange(0, 2), cv::flann::KDTreeIndexParams());

    // store amd query
    cv::Mat inter_2_pt3index_dist;

    flannIndex.knnSearch(lines_Mf.colRange(4, 6).clone(), *inter_knn_Mi, inter_2_pt3index_dist, support_H_num,
                         cv::flann::SearchParams());
}

void matSkew(cv::Mat M1f, cv::Mat &skew_Mf)
{
    cv::Mat skew_Mf_ = cv::Mat(3, 3, CV_32FC1);

    skew_Mf_.at<float>(0, 0) = 0;
    skew_Mf_.at<float>(0, 1) = -M1f.at<float>(2, 0);
    skew_Mf_.at<float>(0, 2) = M1f.at<float>(1, 0);

    skew_Mf_.at<float>(1, 0) = M1f.at<float>(2, 0);
    skew_Mf_.at<float>(1, 1) = 0;
    skew_Mf_.at<float>(1, 2) = -M1f.at<float>(0, 0);

    skew_Mf_.at<float>(2, 0) = -M1f.at<float>(1, 0);
    skew_Mf_.at<float>(2, 1) = M1f.at<float>(0, 0);
    skew_Mf_.at<float>(2, 2) = 0;

    skew_Mf_.copyTo(skew_Mf);
}

void createMap(int imr, int imc, cv::Mat inter_lines_Mf, cv::Mat lines_Mf, cv::Mat &inter_map_)
{
    cv::Mat inter_Mat_ushort = cv::Mat::zeros(imr, imc, CV_32SC1);
    std::vector<int> xx, yy;
    int x1, y1, x2, y2;

    float *lines_p = (float *)lines_Mf.data;
    int *inter_Mat_p = (int *)inter_Mat_ushort.data;
    float *inter_lines_p = (float *)inter_lines_Mf.data;

    int ind = 0;

    for (int i = 0; i < lines_Mf.rows; i++)
    {
        ind = i * lines_Mf.cols;
        x1 = round(lines_p[ind]);
        y1 = round(lines_p[ind + 1]);
        x2 = round(lines_p[ind + 2]);
        y2 = round(lines_p[ind + 3]);

        Bresenham(x1, y1, x2, y2, xx, yy);

        for (int j = 0; j < xx.size(); j++)
        {
            if (xx.at(j) < 0 || yy.at(j) < 0 || xx.at(j) >= imc || yy.at(j) >= imr)
                continue;

            inter_Mat_p[yy.at(j) * inter_Mat_ushort.cols + xx.at(j)] = i + 1;

            // std::cout << sparse_inter_map.ref<short>(yy.at(j), xx.at(j)) << " " << inter_Mat_p[yy.at(j) *
            // inter_Mat_ushort.cols + xx.at(j)] << std::endl; std::getchar();
        }
    }

    int px, py;
    for (int i = 0; i < inter_lines_Mf.rows; i++)
    {
        ind = i * inter_lines_Mf.cols;

        px = inter_lines_p[ind + 2];
        py = inter_lines_p[ind + 3];

        if (px < 0 || py < 0 || px >= imc || py >= imr)
            continue;

        inter_Mat_p[py * inter_Mat_ushort.cols + px] = -i - 1;
    }

    inter_Mat_ushort.copyTo(inter_map_);
}

void createMap(int imr, int imc, cv::Mat lines_Mf, cv::Mat &inter_map_)
{
    cv::Mat inter_Mat_ushort = cv::Mat::zeros(imr, imc, CV_32SC1);
    std::vector<int> xx, yy;
    int x1, y1, x2, y2;

    float *lines_p = (float *)lines_Mf.data;
    int *inter_Mat_p = (int *)inter_Mat_ushort.data;

    int ind = 0;

    for (int i = 0; i < lines_Mf.rows; i++)
    {
        ind = i * lines_Mf.cols;

        x1 = round(lines_p[ind]);
        y1 = round(lines_p[ind + 1]);
        x2 = round(lines_p[ind + 2]);
        y2 = round(lines_p[ind + 3]);

        Bresenham(x1, y1, x2, y2, xx, yy);

        for (int j = 0; j < xx.size(); j++)
        {

            if (xx.at(j) < 0 || yy.at(j) < 0 || xx.at(j) >= imc || yy.at(j) >= imr)
                continue;

            inter_Mat_p[yy.at(j) * inter_Mat_ushort.cols + xx.at(j)] = i + 1;
        }
    }

    inter_Mat_ushort.copyTo(inter_map_);
}

void matchPair(SfMManager *sfm, MatchManager *match, int match_ind, float error_max, float ang_max)
{

    int mid1, mid2;
    match->iPairIndex(match_ind, mid1, mid2);

    if (sfm->iImageLineSize(mid1) == 0 || sfm->iImageLineSize(mid2) == 0)
    {
        return;
    }

    cv::Mat CM = sfm->iCameraMat(mid1);
    cv::Mat CN = sfm->iCameraMat(mid2);

    cv::Mat F, Ae;
    computeFAE(CM, CN, F, Ae);

    Ae = Ae.mul(-1);

    int rm2, cm2;
    sfm->iImSize(mid2, rm2, cm2);

    // load lines and intersections
    cv::Mat lines_range1, lines_range2, l2l_range1, l2l_range2;
    cv::Mat lines1_Mf, lines2_Mf, l2l_1_Mf, l2l_2_Mf;
    cv::Mat inter_range_i, line_range_i;

    lines_range1 = *(sfm->iImageLines(mid1));
    lines_range2 = *(sfm->iImageLines(mid2));

    lines1_Mf = lines_range1.colRange(0, 7).clone();
    lines2_Mf = lines_range2.colRange(0, 7).clone();
    line_range_i = lines_range1.colRange(7, 11).clone();

    l2l_range1 = *(sfm->iJunctionLines(mid1));
    l2l_range2 = *(sfm->iJunctionLines(mid2));

    l2l_1_Mf = l2l_range1.colRange(0, 8).clone();
    l2l_2_Mf = l2l_range2.colRange(0, 8).clone();
    inter_range_i = l2l_range1.colRange(8, 10).clone();

    cv::Mat l2l_Mf;

    createMap(rm2, cm2, l2l_2_Mf, lines2_Mf, l2l_Mf);

    float *M1 = sfm->iCamera33TransPtr(mid1);
    float *C1 = sfm->iCameraCenterPtr(mid1);
    cv::Mat homos_junc;
    cv::Mat planes_junc;
    cv::Mat planes_lineid_junc;

    if (sfm->iJunctionLines(mid1)->rows >= 1 && sfm->iJunctionLines(mid2)->rows >= 1)
        findHomography((float *)inter_range_i.data, (float *)l2l_1_Mf.data, l2l_1_Mf.rows, (float *)l2l_2_Mf.data,
                       l2l_2_Mf.rows, (int *)l2l_Mf.data, (float *)lines1_Mf.data, (float *)lines2_Mf.data, CM, CN, Ae,
                       F, M1, C1, rm2, cm2, homos_junc, planes_junc, planes_lineid_junc, error_max, ang_max, match_ind);

    cv::Mat parral_map;
    cv::Mat homos_parra;
    cv::Mat planes_parra;
    cv::Mat planes_lineid_parra;

    cv::Mat parra_Lines1 = *sfm->iParraLines(mid1);
    cv::Mat parra_Lines2 = *sfm->iParraLines(mid2);

    if (parra_Lines1.rows >= 1 && parra_Lines2.rows >= 1)
    {
        createMap(rm2, cm2, parra_Lines2.colRange(2, parra_Lines2.cols).clone(), parral_map);
        findPLHomography(parra_Lines1, parra_Lines2, (int *)parral_map.data, (float *)lines1_Mf.data,
                         (float *)lines2_Mf.data, line_range_i, CM, CN, Ae, F, M1, C1, rm2, cm2, homos_parra,
                         planes_parra, planes_lineid_parra, error_max, ang_max, match_ind);
    }

    cv::Mat homos;
    cv::Mat planes;
    cv::Mat planes_lineid;

    int supportnum = SUPPORT_HOMO_NUM;

    if (homos_parra.rows + homos_junc.rows <= 0)
    {
        return;
    }
    else if (homos_parra.rows == 0)
    {
        homos = homos_junc;
        planes = planes_junc;
        planes_lineid = planes_lineid_junc;
    }
    else if (homos_junc.rows == 0)
    {
        homos = homos_parra;
        planes = planes_parra;
        planes_lineid = planes_lineid_parra;
    }
    else
    {

        cv::vconcat(homos_junc, homos_parra, homos);
        cv::vconcat(planes_junc, planes_parra, planes);
        cv::vconcat(planes_lineid_junc, planes_lineid_parra, planes_lineid);
    }

    cv::Mat line_knn_Mi;

    if (homos.rows <= supportnum + 10)
    {
        // create adjacent matrix
        line_knn_Mi = cv::Mat::zeros(lines1_Mf.rows, supportnum, CV_32SC1);

        for (int i = 0; i < lines1_Mf.rows; i++)
            for (int j = 0; j < supportnum; j++)
                line_knn_Mi.at<int>(i, j) = j;
    }
    else
        line2KDtree(lines1_Mf, homos, &line_knn_Mi, supportnum);

    cv::Mat pt3Ds, matches_r2l;
    // printf("match in %d %d %d in\n", homos.rows, lines1_Mf.rows, match_ind);

    // match for single lines
    float dist = error_max;

    float *M2 = sfm->iCamera33TransPtr(mid2);
    float *C2 = sfm->iCameraCenterPtr(mid2);

    guidedMatching(match, mid1, mid2, match_ind, (float *)lines1_Mf.data, (float *)line_range_i.data,
                   (int *)line_knn_Mi.data, (float *)lines2_Mf.data, l2l_Mf, lines1_Mf.rows, lines2_Mf.rows,
                   (float *)homos.data, (float *)planes.data, (ushort *)planes_lineid.data, homos.rows, CM, CN, F, M1,
                   C1, M2, C2, line_knn_Mi.cols, dist, rm2, cm2);

    match->matched_mark(match_ind);

    std::cout << "matched out " << match_ind << std::endl;
}
