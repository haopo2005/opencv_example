------------------
Build instructions
------------------

Required tools:
* Cmake 
* c/c++ compiler (gcc)
* opencv > 3.1.0
* point cloud library >1.8
* SSBA =3.0,https://github.com/eokeeffe/SSBA

after SSBA installtion, need small modification:
diff --git a/Math/v3d_mathutilities.h b/Math/v3d_mathutilities.h
index 0ba3b40..6bf453d 100644
--- a/Math/v3d_mathutilities.h
+++ b/Math/v3d_mathutilities.h
@@ -8,6 +8,7 @@
 
 #include <vector>
 
+
 #ifdef WIN32
 #include "win32config.h"
 #endif
@@ -420,7 +421,7 @@ namespace V3D
          scale += d2/d1;
       }
       scale /= n;
-      if(isinf(scale)||isnan(scale)||scale==0.0)scale=1.0;
+      if(std::isinf(scale)||std::isnan(scale)||scale==0.0)scale=1.0;
       for (size_t i = 0; i < n; ++i) scaleVectorIP(scale, left[i]);
 
       getEuclideanTransformation(left, right, R, T);


-----------------
Linux compilation
-----------------
[make sure all the packages needed are under the correct path in CMakeLists.txt]
cmake .
make

two_view_sfm:
./cloud_viewer picname1 picname2

multi_view_sfm and multi_view_sfm_ba:
./cloud_viewer path_image_directory

The programe will generate ply files. You may use meshlab to watch the rgbd result. 
You may watch my video for better understanding.[only chinese]
http://edu.csdn.net/course/detail/4350