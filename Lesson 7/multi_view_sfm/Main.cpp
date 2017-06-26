#include <iostream>
#include "MultiCamera.h"

int main(int argc, char **argv)
{
	MultiCamera sfm;
	std::stringstream buf;
	std::string dir_path;
	
	buf << argv[1];
	dir_path = buf.str();
	

	sfm.LetsGo(dir_path);
	
	return 0;
}
