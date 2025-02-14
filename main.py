import sys
import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import Qt, QThreadPool, QRunnable, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from ui_code.ui import *
from transfer import transfer_Process, scaleSize
from pre_process import image_to_tensor
import cv2
import numpy as np

class RunningThread(QRunnable):
    def __init__(self, process):
        super(RunningThread, self).__init__()
        self.generator = None
        self.process = process
        self.generator = transfer_Process(self.process.contentImg, self.process.styleImg)

    def run(self):
        self.generator.run(self.process)


class MainUI(QWidget):
    received = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.app = QtWidgets.QApplication(sys.argv)
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.MainWindow.show()
        self.ui.style1.clicked.connect(self.update) 
        self.ui.style2.clicked.connect(self.update) 
        self.ui.style3.clicked.connect(self.update) 
        self.ui.style4.clicked.connect(self.update) 
        self.ui.pushButton_2.clicked.connect(self.stop) 
        self.ui.pushButton.clicked.connect(self.generate) 
        self.ui.otherstyle.clicked.connect(self.chooseStyleImg) 
        self.ui.actionOpen.triggered.connect(self.chooseOriginalImg)
        self.ui.actionSaveAs.triggered.connect(self.saveAs)
        self.received.connect(self.displayProgess)
        self.ui.horizontalLayoutWidget.hide()

        self.contentImg = None
        self.styleImg = None
        self.result = None
        self.generating = False
        self.threadpool = QThreadPool()
    
    def update(self):
        if not self.generating:
            sender = self.sender()
            self.ui.styleImg.setStyleSheet("")
            self.styleImg = cv2.imread("ui_code/" + sender.path)
            self.ui.styleImg.setPixmap(self.genPixmap(self.ui.styleImg.width(), 
                                                            self.ui.styleImg.height(), self.styleImg))
            self.styleImg = image_to_tensor(self.styleImg)
        else:
            self.ui.label.setText("An image is generating. Please press 'stop' button")

    def chooseStyleImg(self):
        if not self.generating:
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
            dialog.setNameFilter("Images (*.png *.jpg)")
            dialog.setViewMode(QFileDialog.ViewMode.List)
            if dialog.exec():
                filename = dialog.selectedFiles()
                self.styleImg = cv2.imread(filename[0])
                self.ui.styleImg.setPixmap(self.genPixmap(self.ui.styleImg.width(), 
                                                            self.ui.styleImg.height(), self.styleImg))
                self.ui.styleImg.setStyleSheet("")
                self.styleImg = image_to_tensor(self.styleImg)
        else:
            self.ui.label.setText("An image is generating. Please press 'stop' button")

    def chooseOriginalImg(self):
        if not self.generating:
            self.ui.label.setText("")
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
            dialog.setNameFilter("Images (*.png *.jpg)")
            dialog.setViewMode(QFileDialog.ViewMode.List)
            if dialog.exec():
                filename = dialog.selectedFiles()
                self.contentImg = cv2.imread(filename[0])
                self.ui.originalImg.setPixmap(self.genPixmap(self.ui.originalImg.width(), 
                                                            self.ui.originalImg.height(), self.contentImg))
                self.ui.originalImg.setStyleSheet("")
                self.contentImg = image_to_tensor(self.contentImg)
        else:
            self.ui.label.setText("An image is generating. Please press 'stop' button")
        
    def initUI(self):
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('PyQt events with keyboard')
        self.show()
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.threadpool.clear()
            self.close()
    
    def generate(self):
        if self.contentImg is not None and self.styleImg is not None and not self.generating:
            self.ui.horizontalLayoutWidget.show()
            self.ui.newImg.clear()
            self.ui.newImg.setText("New image here")
            self.ui.newImg.setStyleSheet("background-color: rgb(191, 191, 191);")
            self.ui.label.setText("")
            self.generating = True
            runner = RunningThread(self)
            self.threadpool.start(runner)
            
            
        elif not self.generating:
            self.ui.label.setText("Cannot find content or style")
        else:
            self.ui.label.setText("An image is generating. Please press 'stop' button")

    def genPixmap(self, max_w, max_h, img):
        nw, nh = scaleSize(img.shape[1], img.shape[0], max_w, max_h)
        rep = np.copy(img)
        rep = cv2.resize(rep, (nw, nh))
        h, w, c = rep.shape
        if rep.ndim == 1:
            rep =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        qimg = QImage(rep.data, w, h, 3*w, QImage.Format_BGR888) 
        pixmap = QPixmap(qimg)
        return pixmap
    
    def stop(self):
        self.generating = False
        self.ui.horizontalLayoutWidget.hide()
        self.ui.label.setText("")

    def saveAs(self):
        if self.result is not None:
            options = QtWidgets.QFileDialog.Options()
            fileName = QtWidgets.QFileDialog.getSaveFileName(self, 
                "Save File", "", "Images (*.png *.jpg)", options = options)
            if fileName[0]:
                cv2.imwrite(fileName[0], self.result)
        else:
            self.ui.label.setText("No result to save")

    @pyqtSlot(int)
    def displayProgess(self, num):
        self.ui.progressBar.setValue(num)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainUI()
    sys.exit(app.exec_())