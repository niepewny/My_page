#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "reconstmanager.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void chooseDir();
    void checkInput();
    void runButtonClicked();
    void threadFinished();

private:
    Ui::MainWindow *ui;
    ReconstManager *reconstManager = nullptr;

    static QString dir;
};
#endif // MAINWINDOW_H
