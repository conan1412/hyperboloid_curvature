#pragma once
#include <string>
#include <vector>


struct fPointXYZ
{
	float x;
	float y;
	float z;
};

void save_pointcloud_aspcd(std::vector<fPointXYZ> vPts, std::string filename);