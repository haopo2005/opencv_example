#include "Triangle.h"

//对基础矩阵E进行奇异值分解
void Triangle::TakeSVDOfE(cv::Mat_<double>& E, cv::Mat& svd_u, cv::Mat& svd_vt, cv::Mat& svd_w) 
{
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
	/*std::cout << "----------------------- SVD ------------------------\n";
	std::cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<std::endl;
	std::cout << "----------------------------------------------------\n";*/
}

//从基础矩阵中E提取R和T，注意解不唯一
bool Triangle::DecomposeEtoRandT(cv::Mat_<double>& E,cv::Mat_<double>& R1,cv::Mat_<double>& R2,
								cv::Mat_<double>& t1,cv::Mat_<double>& t2) 
{
	//Using HZ E decomposition
	cv::Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) 
		singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		std::cout << "singular values are too far apart: "<<singular_values_ratio<<"\n";
		return false;
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

//检测旋转矩阵R的合法性，行列式值为1
bool Triangle::CheckCoherentRotation(cv::Mat_<double>& R) {
	
	if(fabsf(determinant(R))-1.0 > 1e-06) {
		std::cerr << "det(R) != +-1.0, this is not a rotation matrix："<<fabsf(determinant(R))-1.0 << std::endl;
		return false;
	}
	return true;
}


/**
 线性最小二乘法求解三角化问题，文章出处，"Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> Triangle::LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
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
 *迭代10次，提高三角化精度
 */
cv::Mat_<double> Triangle::IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
											cv::Matx34d P,			//camera 1 matrix
											cv::Point3d u1,			//homogenous image point in 2nd camera
											cv::Matx34d P1			//camera 2 matrix
											) {
	double wi = 1, wi1 = 1;
	cv::Mat_<double> X(4,1); 
	for (int i=0; i<10; i++) //Hartley suggests 10 iterations at most
	{
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


//检验三角化合法性
bool Triangle::TestTriangulation(const std::vector<CloudPoint>& pcloud, const cv::Matx34d& P, std::vector<uchar>& status) 
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
	{	
		//std::cout<<"less than 75% of the points are in front of the camera:" <<percentage<<std::endl;
		return false; //less than 75% of the points are in front of the camera
	}
	std::cout << "the points are in front of the camera:" << percentage << std::endl;
	return true;
}

std::vector<cv::Point3d> Triangle::CloudPointsToPoints(const std::vector<CloudPoint> cpts) {
	std::vector<cv::Point3d> out;
	for (unsigned int i=0; i<cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}


