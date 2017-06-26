#include "MultiCamera.h"

bool sort_by_first(std::pair<int,std::pair<int,int> > a, std::pair<int,std::pair<int,int> > b)
{
	return a.first < b.first; 
}

//导入相机内参
void MultiCamera::read_intrisic()
{
	//load calibration matrix
	cv::FileStorage fs;
	if(fs.open("out_camera_data.yml",cv::FileStorage::READ)) {
		fs["camera_matrix"]>>cam_matrix;
		fs["distortion_coefficients"]>>distortion_coeff;
	} else {
		//no calibration matrix file - mockup calibration
		double max_w_h = MAX(MyMatch.imgs_size.height,MyMatch.imgs_size.width);
		cam_matrix = (cv::Mat_<double>(3,3) <<	max_w_h ,0,MyMatch.imgs_size.width/2.0,
							0,max_w_h,MyMatch.imgs_size.height/2.0,
							0,0,1);
		distortion_coeff = cv::Mat_<double>::zeros(1,4);
	}
	
	K = cam_matrix;
	invert(K, Kinv); //get inverse of camera matrix
}


//主入口
void MultiCamera::LetsGo(const std::string dir_path)
{
	MyMatch.ReadFromDirectory(dir_path);
	std::sort(MyMatch.imglist.begin(), MyMatch.imglist.end());
	
	MyMatch.MatchImageListsFeatures(MyMatch.imglist);//寻找图像列表两两匹配点

	read_intrisic();
	
	GetBaseLineTriangulation();//寻找第一组匹配的图像

	
	cv::Matx34d P1 = Pmats[m_second_view];
	cv::Mat_<double> t = (cv::Mat_<double>(1,3) << P1(0,3), P1(1,3), P1(2,3));
	cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0), P1(0,1), P1(0,2), 
												   P1(1,0), P1(1,1), P1(1,2), 
												   P1(2,0), P1(2,1), P1(2,2));
	cv::Mat_<double> rvec(1,3); 
	cv::Rodrigues(R, rvec);

	done_views.insert(m_first_view);
	done_views.insert(m_second_view);
	good_views.insert(m_first_view);
	good_views.insert(m_second_view);

	//loop images to incrementally recover more cameras 
	while (done_views.size() != MyMatch.imglist.size())
	{
		//find image with highest 2d-3d correspondance [Snavely07 4.2]
		unsigned int max_2d3d_view = -1, max_2d3d_count = 0;
		std::vector<cv::Point3f> max_3d; 
		std::vector<cv::Point2f> max_2d;
		for (unsigned int _i=0; _i < MyMatch.imglist.size(); _i++) 
		{	
			if(done_views.find(_i) != done_views.end()) 
				continue; //already done with this view

			std::vector<cv::Point3f> tmp3d; 
			std::vector<cv::Point2f> tmp2d;
			std::cout << MyMatch.imglist[_i] << ": ";
			Find2D3DCorrespondences(_i,tmp3d,tmp2d);
			if(tmp3d.size() > max_2d3d_count) {
				max_2d3d_count = tmp3d.size();
				max_2d3d_view = _i;
				max_3d = tmp3d; 
				max_2d = tmp2d;
			}
		}
		int i = max_2d3d_view; //highest 2d3d matching view

		std::cout << "-------------------------- " << MyMatch.imglist[i] << " --------------------------\n";
		done_views.insert(i); // don't repeat it for now

		bool pose_estimated = FindPoseEstimation(i,rvec,t,R,max_3d,max_2d);
		std::cout<<"pose_estimated: "<<pose_estimated<<std::endl;
		if(!pose_estimated)
			continue;

		//store estimated pose	
		Pmats[i] = cv::Matx34d(R(0,0),R(0,1),R(0,2),t(0),
								 R(1,0),R(1,1),R(1,2),t(1),
								 R(2,0),R(2,1),R(2,2),t(2));
		
		// start triangulating with previous GOOD views
		for (std::set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view) 
		{
			int view = *done_view;
			if( view == i ) 
				continue; //skip current...

			std::cout << " -> " << MyMatch.imglist[view] << std::endl;
			std::vector<CloudPoint> new_triangulated;
			std::vector<int> add_to_cloud;

			bool good_triangulation = TriangulatePointsBetweenViews(i,view,new_triangulated,add_to_cloud);
			if(!good_triangulation) 
				continue;

			std::cout << "before triangulation: " << all_pcloud.size();
			for (int j=0; j<add_to_cloud.size(); j++) 
			{
				if(add_to_cloud[j] == 1)
					all_pcloud.push_back(new_triangulated[j]);
			}
			std::cout << " after " << all_pcloud.size() << std::endl;
		}
		good_views.insert(i);
	}
	

	/*for(int i=0;i<MyMatch.imgs_orig.size();i++)
	{
		 char dst[30];
		 sprintf(dst, "/root/result/%03d.jpg", i);
		 cv::imwrite(dst, MyMatch.imgs_orig[i]);
	}*/

	
	//获取3维点对应颜色
	std::vector<cv::Vec3b> RGBforCloud;
	GetRGBForPointCloud(RGBforCloud);

	//生成PLY文件
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	cloud.width = all_pcloud.size();
	cloud.height = 1;
	cloud.is_dense = true;
	cloud.points.resize(cloud.width * cloud.height);

	for(unsigned int k=0; k < all_pcloud.size();k++)
	{
		pcl::PointXYZRGB pclp;
		
		pclp.x = all_pcloud[k].pt.x;
		pclp.y = all_pcloud[k].pt.y;
		pclp.z = all_pcloud[k].pt.z;
		pclp.b = RGBforCloud[k][0];
		pclp.g = RGBforCloud[k][1];
		pclp.r = RGBforCloud[k][2];
		//printf("%d,%d,%d\n",pclp.b,pclp.g,pclp.r);
		cloud.push_back(pclp);
	}
			
	
	pcl::io::savePLYFileBinary("color.ply", cloud);

	
	std::cout << "======================================================================\n";
	std::cout << "========================= Depth Recovery DONE ========================\n";
	std::cout << "======================================================================\n";
}


void MultiCamera::GetBaseLineTriangulation()
{
	std::cout << "=========================== Baseline triangulation ===========================\n";

	cv::Matx34d P(1,0,0,0,
				  0,1,0,0,
				  0,0,1,0),
				P1(1,0,0,0,
				   0,1,0,0,
				   0,0,1,0);
	
	
	//sort pairwise matches to find the lowest Homography inliers [Snavely07 4.2]
	std::cout << "Find highest match...";
	std::list<std::pair<int,std::pair<int,int> > > matches_sizes;
	for(std::map<std::pair<int,int> ,std::vector<cv::DMatch> >::iterator i = MyMatch.matches_matrix.begin(); 
		i != MyMatch.matches_matrix.end(); ++i) 
	{
		if((*i).second.size() < 100)
			matches_sizes.push_back(std::make_pair(100,(*i).first));
		else 
		{
			std::vector<cv::Point2f> ipts,jpts;
			std::vector<cv::DMatch> matches = MyMatch.matches_matrix[std::make_pair((*i).first.first,(*i).first.second)];
		    for (unsigned int j=0; j< matches.size(); j++) 
			{
				ipts.push_back(MyMatch.imgpts[(*i).first.first][matches[j].queryIdx].pt);
				jpts.push_back(MyMatch.imgpts[(*i).first.second][matches[j].trainIdx].pt);
			}
			std::vector<uchar> status;
			double minVal,maxVal; 
			cv::minMaxIdx(ipts,&minVal,&maxVal); 
			cv::findHomography(ipts,jpts,status,CV_RANSAC, 0.004 * maxVal); //threshold from Snavely07
			int Hinliers = cv::countNonZero(status);
			int percent = (int)(((double)Hinliers) / ((double)(*i).second.size()) * 100.0);
			std::cout << "[" << (*i).first.first << "," << (*i).first.second << " = "<<percent<<"] ";
			matches_sizes.push_back(std::make_pair((int)percent,(*i).first));
		}
	}
	std::cout << std::endl;
	matches_sizes.sort(sort_by_first);//asc sort by the ratio of Hinliers and the number of F inliers


	//Reconstruct from two views
	bool goodF = false;
	int highest_pair = 0;
	m_first_view = 0;
	m_second_view = 0;
	
	//reverse iterate by number of matches
	for(std::list<std::pair<int,std::pair<int,int> > >::iterator highest_pair = matches_sizes.begin(); 
		highest_pair != matches_sizes.end() && !goodF;  //goodF, once find , then quit
		++highest_pair) 
	{
		m_second_view = (*highest_pair).second.second;
		m_first_view  = (*highest_pair).second.first;

		std::cout << " -------- " << MyMatch.imglist[m_first_view] << " and " << MyMatch.imglist[m_second_view] << " -------- " <<std::endl;
	
		goodF = FindCameraMatrices(m_first_view, m_second_view, 
						P, P1, MyMatch.matches_matrix[std::make_pair(m_first_view,m_second_view)]);

		if (goodF) 
		{
			std::vector<CloudPoint> new_triangulated;
			std::vector<int> add_to_cloud;

			Pmats[m_first_view] = P;
			Pmats[m_second_view] = P1;
			bool good_triangulation = TriangulatePointsBetweenViews(m_second_view,m_first_view,new_triangulated,add_to_cloud);
			if(!good_triangulation || cv::countNonZero(add_to_cloud) < 10) 
			{
				std::cout << "triangulation failed" << std::endl;
				goodF = false;
				Pmats[m_first_view] = 0;
				Pmats[m_second_view] = 0;
				m_second_view++;
			} else 
			{
				std::cout << "before triangulation: " << all_pcloud.size();
				for (unsigned int j=0; j<add_to_cloud.size(); j++) 
				{
					if(add_to_cloud[j] == 1)
						all_pcloud.push_back(new_triangulated[j]);
				}
				std::cout << " after " << all_pcloud.size() << std::endl;
			}				
		}
	}
	if (!goodF)
	{
		std::cout<<"Cannot find a good pair of images to obtain a baseline triangulation"<<std::endl;
	}
		
}

bool MultiCamera::FindCameraMatrices( const int first_view, const int second_view,
						cv::Matx34d& P, cv::Matx34d& P1, std::vector<cv::DMatch>& matches)
{
	std::cout<<"Find camera matrices..."<<std::endl;

	double minVal,maxVal;
	std::vector<unsigned char> inliersMask(matches.size());
	std::vector<cv::Point2f> imgpts1;
	std::vector<cv::Point2f> imgpts2;

	for (unsigned int i=0; i< matches.size(); i++) 
	{
		imgpts1.push_back(MyMatch.imgpts[first_view][matches[i].queryIdx].pt);
		imgpts2.push_back(MyMatch.imgpts[second_view][matches[i].trainIdx].pt);
	}
	
	cv::Mat F = cv::findFundamentalMat(imgpts1, imgpts2, inliersMask, cv::FM_LMEDS);
	cv::Mat_<double> E = K.t()*F*K;
	//std::cout<<"E:\n"<<E<<std::endl;
	
	cv::Mat_<double> R1(3,3);
	cv::Mat_<double> R2(3,3);
	cv::Mat_<double> t1(1,3);
	cv::Mat_<double> t2(1,3);
	P = cv::Matx34d(1,0,0,0,
				  0,1,0,0,
				  0,0,1,0);
	P1 = cv::Matx34d(1,0,0,0,
				   0,1,0,0,
				   0,0,1,0);
	
	if(fabsf(determinant(E)) > 1e-06) {
		std::cout << "det(E) != 0 : " << fabsf(determinant(E)) << "\n";
		P1 = 0;
		return false;
	}
	
	//decompse E to P'
	if (!ComputeTriangle.DecomposeEtoRandT(E,R1,R2,t1,t2))
		return false;

	if(determinant(R1)+1.0 < 1e-09) 
	{
		//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
		std::cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << std::endl;
		E = -E;
		ComputeTriangle.DecomposeEtoRandT(E,R1,R2,t1,t2);
	}
	if (!ComputeTriangle.CheckCoherentRotation(R1))
	{
		std::cout << "resulting rotation is not coherent\n";
		P1 = 0;
		return false;
	}
	P1 = cv::Matx34d(R1(0,0), R1(0,1), R1(0,2), t1(0),
				 R1(1,0), R1(1,1), R1(1,2), t1(1),
				 R1(2,0), R1(2,1), R1(2,2), t1(2));
	//std::cout << "Testing P1 " << std::endl << cv::Mat(P1) << std::endl;
	
	//triangulation
	std::vector<CloudPoint> pcloud, pcloud1;
	double reproj_error1 = TriangulatePoints(imgpts1, imgpts2, P, P1, pcloud);
	double reproj_error2 = TriangulatePoints(imgpts2, imgpts1, P1, P, pcloud1);
	std::vector<uchar> tmp_status;
	if (!ComputeTriangle.TestTriangulation(pcloud,P1,tmp_status) || !ComputeTriangle.TestTriangulation(pcloud1,P,tmp_status) 
		|| reproj_error1 > 100.0 || reproj_error2 > 100.0) 
	{
		if (!ComputeTriangle.CheckCoherentRotation(R2)) {
			std::cout << "resulting rotation is not coherent\n";
			P1 = 0;
			return false;
		}
					
		P1 = cv::Matx34d(R2(0,0), R2(0,1), R2(0,2), t1(0),
					 R2(1,0), R2(1,1), R2(1,2), t1(1),
					 R2(2,0), R2(2,1), R2(2,2), t1(2));
		//std::cout << "Testing P1 "<< std::endl << cv::Mat(P1) << std::endl;

		pcloud.clear(); pcloud1.clear();
		reproj_error1 = TriangulatePoints(imgpts1, imgpts2, P, P1, pcloud);
		reproj_error2 = TriangulatePoints(imgpts2, imgpts1, P1, P, pcloud1);
					
		if (!ComputeTriangle.TestTriangulation(pcloud,P1,tmp_status) || !ComputeTriangle.TestTriangulation(pcloud1,P,tmp_status) 
			|| reproj_error1 > 100.0 || reproj_error2 > 100.0) 
		{
			P1 = cv::Matx34d(R2(0,0), R2(0,1), R2(0,2), t1(0),
					 R2(1,0), R2(1,1), R2(1,2), t1(1),
					 R2(2,0), R2(2,1), R2(2,2), t1(2));
			//std::cout << "Testing P1 "<< std::endl << cv::Mat(P1) << std::endl;

			pcloud.clear(); pcloud1.clear(); 
			reproj_error1 = TriangulatePoints(imgpts1, imgpts2, P, P1, pcloud);
			reproj_error2 = TriangulatePoints(imgpts2, imgpts1, P1, P, pcloud1);
			
			if (!ComputeTriangle.TestTriangulation(pcloud,P1,tmp_status) || !ComputeTriangle.TestTriangulation(pcloud1,P,tmp_status) 
				|| reproj_error1 > 100.0 || reproj_error2 > 100.0) {
				std::cout << "Shit." << std::endl; 
				return false;
			}
		}				
	}
	std::cout<<"oh yeah!"<<std::endl;
	std::cout<<"the size of pcloud" << pcloud.size()<<std::endl;
	/*for (unsigned int i=0; i<pcloud.size(); i++) 
	{
		all_pcloud.push_back(pcloud[i]);
	}*/
	return true;
}



double MultiCamera::TriangulatePoints(const std::vector<cv::Point2f>& _pt_set1_pt, 
						const std::vector<cv::Point2f>& _pt_set2_pt, 
						const cv::Matx34d& P,
						const cv::Matx34d& P1,
						std::vector<CloudPoint>& pointcloud)
{
	cv::Matx44d P1_(P1(0,0),P1(0,1),P1(0,2),P1(0,3),
				P1(1,0),P1(1,1),P1(1,2),P1(1,3),
				P1(2,0),P1(2,1),P1(2,2),P1(2,3),
				0,		0,		0,		1);
	cv::Matx44d P1inv(P1_.inv());
	
	std::cout << "Triangulating...";
	std::vector<double> reproj_error;
	unsigned int pts_size = _pt_set1_pt.size();
	
	cv::Mat_<double> KP1 = K * cv::Mat(P1);
	for (int i=0; i<pts_size; i++) 
	{
		cv::Point2f kp = _pt_set1_pt[i]; 
		cv::Point3d u(kp.x,kp.y,1.0);
		cv::Mat_<double> um = Kinv * cv::Mat_<double>(u); 
		u.x = um(0); u.y = um(1); u.z = um(2);

		cv::Point2f kp1 = _pt_set2_pt[i]; 
		cv::Point3d u1(kp1.x,kp1.y,1.0);
		cv::Mat_<double> um1 = Kinv * cv::Mat_<double>(u1); 
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);
		
		cv::Mat_<double> X = ComputeTriangle.IterativeLinearLSTriangulation(u,P,u1,P1);
		
		cv::Mat_<double> xPt_img = KP1 * X;				//reproject
		cv::Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
		double reprj_err = cv::norm(xPt_img_-kp1);
		reproj_error.push_back(reprj_err);

		CloudPoint cp; 
		cp.pt = cv::Point3d(X(0),X(1),X(2));
		cp.reprojection_error = reprj_err;
		
		pointcloud.push_back(cp);
	}
	cv::Scalar mse = cv::mean(reproj_error);
	std::cout << "Done. ("<<pointcloud.size()<<"points,"  <<", mean reproj err = " << mse[0] << ")"<< std::endl;
	return mse[0];
}



bool MultiCamera::TriangulatePointsBetweenViews(const int working_view, const int older_view,
														   std::vector<CloudPoint>& new_triangulated, std::vector<int>& add_to_cloud)
{
	std::cout << " Triangulate " << MyMatch.imglist[working_view] << " and " << MyMatch.imglist[older_view] << std::endl;
	cv::Matx34d P = Pmats[older_view];
	cv::Matx34d P1 = Pmats[working_view];
	std::vector<cv::Point2f> pt_set1,pt_set2;
	std::vector<cv::DMatch> matches = MyMatch.matches_matrix[std::make_pair(older_view,working_view)];

	for (unsigned int i=0; i< matches.size(); i++) 
	{
		pt_set1.push_back(MyMatch.imgpts[older_view][matches[i].queryIdx].pt);
		pt_set2.push_back(MyMatch.imgpts[working_view][matches[i].trainIdx].pt);
	}
	double reproj_error = TriangulatePoints(pt_set1, pt_set2, P, P1, new_triangulated);
	std::cout << "triangulation reproj error " << reproj_error << std::endl;

	std::vector<uchar> trig_status;
	if(!ComputeTriangle.TestTriangulation(new_triangulated, P, trig_status) || !ComputeTriangle.TestTriangulation(new_triangulated, P1, trig_status))
	{
		std::cout<<"Triangulation did not succeed"<<std::endl;
		return false;
	}

	//filter out outlier points with high reprojection
	std::vector<double> reprj_errors;
	for(int i=0;i<new_triangulated.size();i++)
	{
		reprj_errors.push_back(new_triangulated[i].reprojection_error);
	}
	std::sort(reprj_errors.begin(),reprj_errors.end());

	//get the 80% precentile
	double reprj_err_cutoff = reprj_errors[4 * reprj_errors.size() / 5] * 2.4; //threshold from Snavely07 4.2
	std::vector<CloudPoint> new_triangulated_filtered;
	std::vector<cv::DMatch> new_matches;
	for(int i=0;i<new_triangulated.size();i++) 
	{
		if(trig_status[i] == 0)
			continue; //point was not in front of camera
		if(new_triangulated[i].reprojection_error > 16.0) {
			continue; //reject point
		} 
		if(new_triangulated[i].reprojection_error < 4.0 ||
			new_triangulated[i].reprojection_error < reprj_err_cutoff) 
		{
			new_triangulated_filtered.push_back(new_triangulated[i]);
			new_matches.push_back(matches[i]);
		} 
		else 
		{
			continue;
		}
	}
	std::cout << "filtered out " << (new_triangulated.size() - new_triangulated_filtered.size()) << " high-error points" << std::endl;
	//all points filtered?
	if(new_triangulated_filtered.size() <= 0) 
		return false;

	//scan new triangulated points, if they were already triangulated before - strengthen cloud
	new_triangulated = new_triangulated_filtered;
	matches = new_matches;
	MyMatch.matches_matrix[std::make_pair(older_view,working_view)] = new_matches; //just to make sure, remove if unneccesary
	MyMatch.matches_matrix[std::make_pair(working_view,older_view)] = MyMatch.FlipMatches(new_matches);
	add_to_cloud.clear();
	add_to_cloud.resize(new_triangulated.size(),1);
	int found_other_views_count = 0;
	
	for (int j = 0; j<new_triangulated.size(); j++) 
	{
		//每个特征点有N个索引，每个索引代表当前帧的初始特征点序列标记
		new_triangulated[j].imgpt_for_img = std::vector<int>(MyMatch.imglist.size(),-1); //initial the index of the image who contains speicified feature points  j with -1
		new_triangulated[j].imgpt_for_img[older_view] = matches[j].queryIdx;	//2D reference to <older_view>
		new_triangulated[j].imgpt_for_img[working_view] = matches[j].trainIdx;		//2D reference to <working_view>

		bool found_in_other_view = false;
		for (unsigned int view_ = 0; view_ < MyMatch.imglist.size(); view_++) 
		{
			if(view_ != older_view) 
			{
				//Look for points in <view_> that match to points in <working_view>
				std::vector<cv::DMatch> submatches = MyMatch.matches_matrix[std::make_pair(view_,working_view)];
				for (unsigned int ii = 0; ii < submatches.size(); ii++) 
				{
					if (submatches[ii].trainIdx == matches[j].trainIdx && !found_in_other_view) 
					{
						//Point was already found in <view_> - strengthen it in the known cloud, if it exists there
						for (unsigned int pt3d=0; pt3d < all_pcloud.size(); pt3d++) 
						{
							//all_pcloud[pt3d].imgpt_for_img[working_view] = matches[j].trainIdx;
							//all_pcloud[pt3d].imgpt_for_img[older_view] = matches[j].queryIdx;
							found_in_other_view = true;
							add_to_cloud[j] = 0;
							//std::cout<<"the "<<pt3d<<"th cloud point of older_view:"<<older_view<<" has Idx:"<<matches[j].queryIdx<<std::endl;
							//std::cout<<"the "<<pt3d<<"th cloud point of working_view:"<<working_view<<" has Idx:"<<matches[j].trainIdx<<std::endl;
						}
					}
				}
			}
		}
		if (found_in_other_view) 
		{
			found_other_views_count++;
		} else {
			add_to_cloud[j] = 1;
		}
	}
	std::cout << found_other_views_count << "/" << new_triangulated.size() << " points were found in other views, adding " << cv::countNonZero(add_to_cloud) << " new\n";
	return true;
}



bool MultiCamera::FindPoseEstimation(int working_view, cv::Mat_<double>& rvec, cv::Mat_<double>& t, cv::Mat_<double>& R,
							std::vector<cv::Point3f> ppcloud,	std::vector<cv::Point2f> imgPoints)
{
	if(ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) 
	{ 
		//something went wrong aligning 3D to 2D points..
		std::cout << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" << std::endl;
		return false;
	}
	
	std::vector<int> inliers;
	//use CPU, 根据2d和3d点的对应关系，求camera pose (r,t)
	double minVal,maxVal; cv::minMaxIdx(imgPoints,&minVal,&maxVal);
	cv::solvePnPRansac(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true,
		1000, 0.006 * maxVal, 0.9, inliers);//confidence is 0.9

	std::vector<cv::Point2f> projected3D;
	cv::projectPoints(ppcloud, rvec, t, K, distortion_coeff, projected3D);
	//瞎搞了!当solvePnPRansac没有返回合格点的索引inliers
	if(inliers.size()==0)
	{
		//get inliers
		for(int i=0;i<projected3D.size();i++) 
		{
			if(cv::norm(projected3D[i]-imgPoints[i]) < 5.0)
				inliers.push_back(i);
		}
	}
	if(inliers.size() < (double)(imgPoints.size())/5.0) 
	{
		std::cout << "not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")"<< std::endl;
		return false;
	}

	if(cv::norm(t) > 200.0) {
		// this is bad...
		std::cout << "estimated camera movement is too big, skip this camera\r\n";
		return false;
	}
	cv::Rodrigues(rvec, R);
	if(!ComputeTriangle.CheckCoherentRotation(R)) {
		std::cout << "rotation is incoherent. we should try a different base view..." << std::endl;
		return false;
	}

	std::cout << "found t = " << t << "\nR = \n"<<R<<std::endl;
	return true;
}


void MultiCamera::Find2D3DCorrespondences(int working_view, 
	std::vector<cv::Point3f>& ppcloud, 
	std::vector<cv::Point2f>& imgPoints) 
{
	
	ppcloud.clear(); 
	imgPoints.clear();

	std::vector<int> pcloud_status(all_pcloud.size(),0);
	for (std::set<int>::iterator done_view = good_views.begin(); done_view != good_views.end(); ++done_view) 
	{
		int old_view = *done_view;
		
		//check for matches_from_old_to_working between i'th frame and <old_view>'th frame (and thus the current cloud)
		std::vector<cv::DMatch> matches_from_old_to_working = MyMatch.matches_matrix[std::make_pair(old_view,working_view)];
		for (unsigned int match_from_old_view=0; match_from_old_view < matches_from_old_to_working.size(); match_from_old_view++) 
		{
			// the index of the matching point in <old_view>
			int idx_in_old_view = matches_from_old_to_working[match_from_old_view].queryIdx;
			//scan the existing cloud (pcloud) to see if this point from <old_view> exists
			for (unsigned int pcldp=0; pcldp<all_pcloud.size(); pcldp++) 
			{
				// see if corresponding point was found in this point
				if (idx_in_old_view == all_pcloud[pcldp].imgpt_for_img[old_view] && pcloud_status[pcldp] == 0) //prevent duplicates
				{
					//3d point in cloud
					ppcloud.push_back(all_pcloud[pcldp].pt);
					//2d point in image i
					imgPoints.push_back(MyMatch.imgpts[working_view][matches_from_old_to_working[match_from_old_view].trainIdx].pt);

					pcloud_status[pcldp] = 1;
					break;
				}
			}
		}
	}
	std::cout << "found " << ppcloud.size() << " 3d-2d point correspondences"<<std::endl;
}



void MultiCamera::GetRGBForPointCloud(std::vector<cv::Vec3b>& RGBforCloud)
{
	RGBforCloud.resize(all_pcloud.size());
	for (unsigned long i=0; i< all_pcloud.size(); i++)
	{
		unsigned int good_view = 0;
		std::vector<cv::Vec3b> point_colors;
		for(; good_view < MyMatch.imgs_orig.size(); good_view++) 
		{
			if(all_pcloud[i].imgpt_for_img[good_view] != -1) 
			{
				int pt_idx = all_pcloud[i].imgpt_for_img[good_view];
				//std::cout<<"the "<<i<<"th cloud point of the "<<good_view<<"th image has pt_idx:"<<pt_idx<<std::endl;
				if(pt_idx >= MyMatch.imgpts[good_view].size()) {
					std::cerr << "BUG: point id:" << pt_idx << " should not exist for img #" << good_view << " which has only " << MyMatch.imgpts[good_view].size() << std::endl;
					continue;
				}
				cv::Point _pt = MyMatch.imgpts[good_view][pt_idx].pt;
				assert(good_view < MyMatch.imgs_orig.size() && _pt.x < MyMatch.imgs_orig[good_view].cols && _pt.y < MyMatch.imgs_orig[good_view].rows);
				point_colors.push_back(MyMatch.imgs_orig[good_view].at<cv::Vec3b>(_pt));
				//RGBforCloud[i] = MyMatch.imgs_orig[good_view].at<cv::Vec3b>(_pt);
				//break;
			}
		}
		cv::Scalar res_color = cv::mean(point_colors);
		RGBforCloud[i] = (cv::Vec3b(res_color[0],res_color[1],res_color[2])); //bgr2rgb
		//RGBforCloud[i] = cv::mean(point_colors);
		/*if(good_view == MyMatch.imglist.size()) //nothing found.. put red dot
			RGBforCloud.push_back(cv::Vec3b(255,0,0));*/
	}
}

