#include "reconstmanager.h"
#include "CloudMaker.h"

ReconstManager::ReconstManager()
    :viewer("Cloud")
{}

ReconstManager::~ReconstManager()
{
    delete cloudMaker;
}

void ReconstManager::run()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = cloudMaker->apply();

    viewer.showCloud(cloud);
}
