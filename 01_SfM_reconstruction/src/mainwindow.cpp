#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include "PointFinderr.h"
#include "PointReconstructor.h"
#include <QMessageBox>

QString MainWindow::dir;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    reconstManager = new ReconstManager;

    connect(reconstManager, &ReconstManager::finished, this, &MainWindow::threadFinished);

    connect(ui->ChooseDirPushButton, &QPushButton::released, this, &MainWindow::chooseDir);
    connect(ui->runPushButton, &QPushButton::released, this, &MainWindow::runButtonClicked);
    connect(ui->orbScaleFacLineEdit, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orbNLevelLineEdit, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orbEdgeThLineEdit, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orbFirstLevLineEdit, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orbPatchSizLineEdit, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orbFastThLineEdit, &QLineEdit::textChanged, this, &MainWindow::checkInput);

    connect(ui->ROIZMinLineEdit, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->ROIZMaxLineEdit, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->ROIdT, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->ROIdEuler, &QLineEdit::textChanged, this, &MainWindow::checkInput);

    connect(ui->pointMaxDistance, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->pointMinDistRatio, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->pointRANSAC_LMedS, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->RANSACthreshold, &QLineEdit::textChanged, this, &MainWindow::checkInput);


    connect(ui->orientationTx, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orientationTy, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orientationTz, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orientationEx, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orientationEy, &QLineEdit::textChanged, this, &MainWindow::checkInput);
    connect(ui->orientationEz, &QLineEdit::textChanged, this, &MainWindow::checkInput);

    }

MainWindow::~MainWindow()
{
    delete reconstManager;
    delete ui;
}

void MainWindow::chooseDir()
{
    dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"));
    ui->ORBGroupBox->setEnabled(1);
    ui->pointRecGroupBox->setEnabled(1);
    ui->ROIGroupBox->setEnabled(1);
    ui->distortionGroupBox->setEnabled(1);

    ui->runPushButton->setEnabled(1);
}

void MainWindow::checkInput()
{
    bool flag = true;
    flag &= !ui->orbScaleFacLineEdit->text().isEmpty();
    flag &= !ui->orbNLevelLineEdit->text().isEmpty();
    flag &= !ui->orbEdgeThLineEdit->text().isEmpty();
    flag &= !ui->orbFirstLevLineEdit->text().isEmpty();
    flag &= !ui->orbPatchSizLineEdit->text().isEmpty();
    flag &= !ui->orbPatchSizLineEdit->text().isEmpty();
    flag &= !ui->orbFastThLineEdit->text().isEmpty();

    flag &= !ui->ROIZMinLineEdit->text().isEmpty();
    flag &= !ui->ROIZMaxLineEdit->text().isEmpty();
    flag &= !ui->ROIdT->text().isEmpty();
    flag &= !ui->ROIdEuler->text().isEmpty();

    flag &= !ui->pointMaxDistance->text().isEmpty();
    flag &= !ui->pointMinDistRatio->text().isEmpty();
    flag &= !ui->pointRANSAC_LMedS->text().isEmpty();
    flag &= !ui->RANSACthreshold->text().isEmpty();

    flag &= !ui->orientationTx->text().isEmpty();
    flag &= !ui->orientationTy->text().isEmpty();
    flag &= !ui->orientationTz->text().isEmpty();
    flag &= !ui->orientationEx->text().isEmpty();
    flag &= !ui->orientationEy->text().isEmpty();
    flag &= !ui->orientationEz->text().isEmpty();

    ui->runPushButton->setEnabled(flag);

}

void MainWindow::runButtonClicked()
{
    if (!dir.isEmpty())
    {
        if (reconstManager->cloudMaker == nullptr)
        {
            reconstManager->cloudMaker = new CloudMaker(dir.toStdString());
        }

        ui->runPushButton->setEnabled(0);
        ui->runPushButton->setText("Running...");

        PointFinder::setORBscaleFactor(ui->orbScaleFacLineEdit->text().toFloat());
        PointFinder::setORBnLevels(ui->orbNLevelLineEdit->text().toInt());
        PointFinder::setORBedgeThreshold(ui->orbEdgeThLineEdit->text().toInt());
        PointFinder::setORBpatchSize(ui->orbPatchSizLineEdit->text().toInt());
        PointFinder::setORBfastThreshold(ui->orbFastThLineEdit->text().toInt());
        PointFinder::setORBfirstLevel(ui->orbFirstLevLineEdit->text().toInt());


        PointReconstructor::setZmin(ui->ROIZMinLineEdit->text().toFloat());
        PointReconstructor::setZmax(ui->ROIZMaxLineEdit->text().toFloat());
        PointReconstructor::setdEuler(ui->ROIdEuler->text().toFloat() / 180.0 * 3.14159);
        PointReconstructor::setdT(ui->ROIdT->text().toFloat());
        PointReconstructor::setRansacThreshold(ui->RANSACthreshold->text().toFloat());
        PointReconstructor::setMaxDistance(ui->pointMaxDistance->text().toFloat());
        PointReconstructor::setMinDistRatio(ui->pointMinDistRatio->text().toFloat());
        PointReconstructor::setUseRansac(ui->pointRANSAC_LMedS->text().toInt());      

        reconstManager->start();

    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("Something went wrong.");
        msgBox.exec();
    }
}

void MainWindow::threadFinished()
{
    ui->runPushButton->setText("Run");
    ui->runPushButton->setEnabled(1);
}
