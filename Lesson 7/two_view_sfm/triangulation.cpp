#include "triangulation.h"

void read_intrisic(cv::Mat& K, cv::Mat& Kinv, cv::Mat& distortion_coeff, cv::Mat& cam_matrix, cv::Mat src)
{
	//load calibration matrix
	cv::FileStorage fs;
	if(fs.open("out_camera_data.yml",cv::FileStorage::READ)) {
		fs["camera_matrix"]>>cam_matrix;
		fs["distortion_coefficients"]>>distortion_coeff;
	} else {
		//no calibration matrix file - mockup calibration
		cv::Size imgs_size = src.size();
		double max_w_h = MAX(imgs_size.height,imgs_size.width);
		cam_matrix = (cv::Mat_<double>(3,3) <<	max_w_h ,0,imgs_size.width/2.0,
							0,max_w_h,imgs_size.height/2.0,
							0,0,1);
		distortion_coeff = cv::Mat_<double>::zeros(1,4);
	}
	
	K = cam_matrix;
	invert(K, Kinv); //get inverse of camera matrix
}

void TakeSVDOfE(cv::Mat_<double>& E, cv::Mat& svd_u, cv::Mat& svd_vt, cv::Mat& svd_w) 
{
	/*
	//Using OpenCV's SVD
	cv::SVD svd(E,cv::SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;*/
	//Using Eigen's SVD
	std::cout << "Eigen3 SVD..\n";
	Eigen::Matrix3f  e = Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> >((double*)E.data).cast<float>();
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXf Esvd_u = svd.matrixU();
	Eigen::MatrixXf Esvd_v = svd.matrixV();
	svd_u = (cv::Mat_<double>(3,3) << Esvd_u(0,0), Esvd_u(0,1), Esvd_u(0,2),
						  Esvd_u(1,0), Esvd_u(1,1), Esvd_u(1,2), 
						  Esvd_u(2,0), Esvd_u(2,1), Esvd_u(2,2)); 
	cv::Mat_<double> svd_v = (cv::Mat_<double>(3,3) << Esvd_v(0,0), Esvd_v(0,1), Esvd_v(0,2),
						  Esvd_v(1,0), Esvd_v(1,1), Esvd_v(1,2), 
						  Esvd_v(2,0), Esvd_v(2,1), Esvd_v(2,2));
	svd_vt = svd_v.t();
	svd_w = (cv::Mat_<double>(1,3) << svd.singularValues()[0] , svd.singularValues()[1] , svd.singularValues()[2]);
	std::cout << "----------------------- SVD ------------------------\n";
	std::cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<std::endl;
	std::cout << "----------------------------------------------------\n";
}

bool DecomposeEtoRandT(
	cv::Mat_<double>& E,
	cv::Mat_<double>& R1,
	cv::Mat_<double>& R2,
	cv::Mat_<double>& t1,
	cv::Mat_<double>& t2) 
{
	//Using HZ E decomposition
	cv::Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		std::cout << "singular values are too far apart\n";
		//return false;
	}

	cv::Matx33d W(0,-1,0,	//HZ 9.13
		1,0,0,
		0,0,1);
	cv::Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd_u * cv::Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * cv::Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
	return true;
}

bool CheckCoherentRotation(cv::Mat_<double>& R) {

	
	if(fabsf(determinant(R))-1.0 > 1e-06) {
		std::cerr << "det(R) != +-1.0, this is not a rotation matrix："<<fabsf(determinant(R))-1.0 << std::endl;
		return false;
	}

	return true;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
									cv::Matx34d P,		//camera 1 matrix
									cv::Point3d u1,		//homogenous image point in 2nd camera
									cv::Matx34d P1		//camera 2 matrix
								) 
{
	
	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	//	cout << "u " << u <<", u1 " << u1 << endl;
	//	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u.x*P(1)-u.y*P(0);
	//	A(3) = u1.x*P1(2)-P1(0);
	//	A(4) = u1.y*P1(2)-P1(1);
	//	A(5) = u1.x*P(1)-u1.y*P1(0);
	//	Matx43d A; //not working for some reason...
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u1.x*P1(2)-P1(0);
	//	A(3) = u1.y*P1(2)-P1(1);
	cv::Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),		
			  u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),		
			  u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),	
			  u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
			  );
	//assume the X=(x,y,z,1), the forth value is known, move the forth column of P to the right side of equation Ax=0
	cv::Matx41d B(-(u.x*P(2,3)	-P(0,3)),
			  -(u.y*P(2,3)	-P(1,3)),
			  -(u1.x*P1(2,3)	-P1(0,3)),
			  -(u1.y*P1(2,3)	-P1(1,3)));
	
	cv::Mat_<double> X;
	cv::solve(A,B,X,cv::DECOMP_SVD);
	
	return X;
}




/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
											cv::Matx34d P,			//camera 1 matrix
											cv::Point3d u1,			//homogenous image point in 2nd camera
											cv::Matx34d P1			//camera 2 matrix
											) {
	double wi = 1, wi1 = 1;
	cv::Mat_<double> X(4,1); 
	for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
		cv::Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
		
		//recalculate weights
		double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2)*X)(0);
		double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);
		
		//breaking point
		if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;
		
		wi = p2x;
		wi1 = p2x1;
		
		//reweight equations and solve
		cv::Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,		
				  (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,		
				  (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,	
				  (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
				  );
		cv::Mat_<double> B = (cv::Mat_<double>(4,1) <<	  -(u.x*P(2,3)	-P(0,3))/wi,
												  -(u.y*P(2,3)	-P(1,3))/wi,
												  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
												  -(u1.y*P1(2,3)	-P1(1,3))/wi1
						  );
		
		cv::solve(A,B,X_, cv::DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}


double TriangulatePoints(const std::vector<cv::Point2f>& _pt_set1_pt, 
						const std::vector<cv::Point2f>& _pt_set2_pt, 
						const cv::Mat& K,
						const cv::Mat& Kinv,
						const cv::Mat& distcoeff,
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
		
		cv::Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);
		
//		cout << "3D Point: " << X << endl;
//		Mat_<double> x = Mat(P1) * X;
//		cout <<	"P1 * Point: " << x << endl;
//		Mat_<double> xPt = (Mat_<double>(3,1) << x(0),x(1),x(2));
//		cout <<	"Point: " << xPt << endl;
		cv::Mat_<double> xPt_img = KP1 * X;				//reproject
//		cout <<	"Point * K: " << xPt_img << endl;
		cv::Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
		double reprj_err = norm(xPt_img_-kp1);
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

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts) {
	std::vector<cv::Point3d> out;
	for (unsigned int i=0; i<cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}

bool TestTriangulation(const std::vector<CloudPoint>& pcloud, const cv::Matx34d& P, std::vector<uchar>& status) 
{
	std::vector<cv::Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	std::vector<cv::Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());
	
	cv::Matx44d P4x4 = cv::Matx44d::eye(); 
	for(int i=0;i<12;i++) 
		P4x4.val[i] = P.val[i];
	
	cv::perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	
	status.resize(pcloud.size(),0);
	for (int i=0; i<pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = cv::countNonZero(status);

	double percentage = ((double)count / (double)pcloud.size());
	std::cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << std::endl;
	if(percentage < 0.75)
		return false; //less than 75% of the points are in front of the camera

	//check for coplanarity of points
	if(false) //not
	{
		cv::Mat_<double> cldm(pcloud.size(),3);
		for(unsigned int i=0;i<pcloud.size();i++) {
			cldm.row(i)(0) = pcloud[i].pt.x;
			cldm.row(i)(1) = pcloud[i].pt.y;
			cldm.row(i)(2) = pcloud[i].pt.z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
		cv::Vec3d x0 = pca.mean;
		double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

		for (int i=0; i<pcloud.size(); i++) {
			cv::Vec3d w = cv::Vec3d(pcloud[i].pt) - x0;
			double D = fabs(nrm.dot(w));
			if(D < p_to_plane_thresh) num_inliers++;
		}

		std::cout << num_inliers << "/" << pcloud.size() << " are coplanar" << std::endl;
		if((double)num_inliers / (double)(pcloud.size()) > 0.85)
			return false;
	}
	return true;
}
