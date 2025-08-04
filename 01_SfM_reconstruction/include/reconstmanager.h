#ifndef RECONSTMANAGER_H
#define RECONSTMANAGER_H

#include <QThread>
#include "CloudMaker.h"

class ReconstManager : public QThread
{
    Q_OBJECT
public:
    ReconstManager();
    ~ReconstManager();

    CloudMaker *cloudMaker = nullptr;
    pcl::visualization::CloudViewer viewer;

private:
    //allows to run the program on the saparate thread
    void run() override;
};

#endif // RECONSTMANAGER_H
