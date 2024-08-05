import sys

from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import copy


class Tagger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "POS_Tagger"
        self.initUI()
        self.data_dict = dict()
        self.sentence = ""
        self.err_list = []

    def initUI(self):
        self.resize(782, 390)
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.line = QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 70, 771, 20))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line.setObjectName("line")

        self.gridLayoutWidget = QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(270, 100, 241, 241))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        # tag 버튼 시작-------------------------------------------------
        self.pushButton_bper = QPushButton(self.gridLayoutWidget)
        self.pushButton_bper.setObjectName("pushButton_bper")
        self.pushButton_bper.clicked.connect(self.b_per_tag_handler)
        self.gridLayout.addWidget(self.pushButton_bper, 0, 0, 1, 1)

        self.pushButton_iper = QPushButton(self.gridLayoutWidget)
        self.pushButton_iper.setObjectName("pushButton_iper")
        self.pushButton_iper.clicked.connect(self.i_per_tag_handler)
        self.gridLayout.addWidget(self.pushButton_iper, 0, 1, 1, 1)

        self.pushButton_bcom = QPushButton(self.gridLayoutWidget)
        self.pushButton_bcom.setObjectName("pushButton_bcom")
        self.pushButton_bcom.clicked.connect(self.b_com_tag_handler)
        self.gridLayout.addWidget(self.pushButton_bcom, 2, 0, 1, 1)

        self.pushButton_o = QPushButton(self.gridLayoutWidget)
        self.pushButton_o.setObjectName("pushButton_o")
        self.pushButton_o.clicked.connect(self.o_tag_handler)
        self.gridLayout.addWidget(self.pushButton_o, 6, 1, 1, 1)

        self.pushButton_bloc = QPushButton(self.gridLayoutWidget)
        self.pushButton_bloc.setObjectName("pushButton_bloc")
        self.pushButton_bloc.clicked.connect(self.b_loc_tag_handler)
        self.gridLayout.addWidget(self.pushButton_bloc, 1, 0, 1, 1)

        self.pushButton_iloc = QPushButton(self.gridLayoutWidget)
        self.pushButton_iloc.setObjectName("pushButton_iloc")
        self.pushButton_iloc.clicked.connect(self.i_loc_tag_handler)
        self.gridLayout.addWidget(self.pushButton_iloc, 1, 1, 1, 1)

        self.pushButton_icom = QPushButton(self.gridLayoutWidget)
        self.pushButton_icom.setObjectName("pushButton_icom")
        self.pushButton_icom.clicked.connect(self.i_com_tag_handler)
        self.gridLayout.addWidget(self.pushButton_icom, 2, 1, 1, 1)

        self.pushButton_baff = QPushButton(self.gridLayoutWidget)
        self.pushButton_baff.setObjectName("pushButton_baff")
        self.pushButton_baff.clicked.connect(self.b_aff_tag_handler)
        self.gridLayout.addWidget(self.pushButton_baff, 3, 0, 1, 1)

        self.pushButton_iaff = QPushButton(self.gridLayoutWidget)
        self.pushButton_iaff.setObjectName("pushButton_iaff")
        self.pushButton_iaff.clicked.connect(self.i_aff_tag_handler)
        self.gridLayout.addWidget(self.pushButton_iaff, 3, 1, 1, 1)

        self.pushButton = QPushButton(self.gridLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.blank_tag_handler)
        self.gridLayout.addWidget(self.pushButton, 6, 0, 1, 1)

        self.pushButton_bpos = QPushButton(self.gridLayoutWidget)
        self.pushButton_bpos.setObjectName("pushButton_bpos")
        self.pushButton_bpos.clicked.connect(self.b_pos_tag_handler)
        self.gridLayout.addWidget(self.pushButton_bpos, 4, 0, 1, 1)

        self.pushButton_ipos = QPushButton(self.gridLayoutWidget)
        self.pushButton_ipos.setObjectName("pushButton_ipos")
        self.pushButton_ipos.clicked.connect(self.i_pos_tag_handler)
        self.gridLayout.addWidget(self.pushButton_ipos, 4, 1, 1, 1)

        self.pushButton_bedu = QPushButton(self.gridLayoutWidget)
        self.pushButton_bedu.setObjectName("pushButton_bedu")
        self.pushButton_bedu.clicked.connect(self.b_edu_tag_handler)
        self.gridLayout.addWidget(self.pushButton_bedu, 5, 0, 1, 1)

        self.pushButton_iedu = QPushButton(self.gridLayoutWidget)
        self.pushButton_iedu.setObjectName("pushButton_iedu")
        self.pushButton_iedu.clicked.connect(self.i_edu_tag_handler)
        self.gridLayout.addWidget(self.pushButton_iedu, 5, 1, 1, 1)

        self.pushButton_previous = QPushButton(self.centralwidget)
        self.pushButton_previous.setGeometry(QtCore.QRect(120, 310, 61, 28))
        self.pushButton_previous.setObjectName("pushButton_previous")
        self.pushButton_previous.clicked.connect(self.get_previous)
        self.pushButton_next = QPushButton(self.centralwidget)
        self.pushButton_next.setGeometry(QtCore.QRect(190, 310, 61, 28))
        self.pushButton_next.setObjectName("pushButton_next")
        self.pushButton_next.clicked.connect(self.get_next)

        self.pushButton_checker = QPushButton(self.centralwidget)
        self.pushButton_checker.setGeometry(QtCore.QRect(60, 230, 93, 31))
        self.pushButton_checker.setObjectName("pushButton_checker")
        self.pushButton_checker.clicked.connect(self.check_error)

        self.plainTextEdit_idx = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_idx.setGeometry(QtCore.QRect(10, 110, 31, 31))
        self.plainTextEdit_idx.setObjectName("plainTextEdit_idx")
        # tag 버튼 끝--------------------------------------------------

        self.label_present = QLabel(self.centralwidget)
        self.label_present.setGeometry(QtCore.QRect(270, 10, 241, 29))
        self.label_present.setObjectName("label_present")
        self.horizontalLayoutWidget_3 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(780, 308, 301, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.horizontalLayoutWidget_2 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 38, 761, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        # 수정 파트
        self.plainTextEdit_checkprv = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_checkprv.setGeometry(QtCore.QRect(10, 110, 241, 31))
        self.plainTextEdit_checkprv.setObjectName("plainTextEdit_checkprv")
        self.plainTextEdit_checknext = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_checknext.setGeometry(QtCore.QRect(10, 190, 241, 31))
        self.plainTextEdit_checknext.setObjectName("plainTextEdit_checknext")

        self.plainTextEdit_idx = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_idx.setGeometry(QtCore.QRect(10, 150, 31, 31))
        self.plainTextEdit_idx.setObjectName("plainTextEdit_idx")
        self.plainTextEdit_word = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_word.setGeometry(QtCore.QRect(50, 150, 101, 31))
        self.plainTextEdit_word.setObjectName("plainTextEdit_word")
        self.plainTextEdit_tag = QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_tag.setGeometry(QtCore.QRect(160, 150, 91, 31))
        self.plainTextEdit_tag.setObjectName("plainTextEdit_tag")

        self.pushButton_edit = QPushButton(self.centralwidget)
        self.pushButton_edit.setGeometry(QtCore.QRect(160, 230, 93, 31))
        self.pushButton_edit.setObjectName("pushButton_edit")
        self.pushButton_edit.clicked.connect(self.edit_tag)

        self.pushButton_pr = QPushButton(self.centralwidget)
        self.pushButton_pr.setGeometry(QtCore.QRect(10, 309, 93, 29))
        self.pushButton_pr.setObjectName("pushButton_pr")
        self.pushButton_pr.clicked.connect(self.get_present)

        self.plainTextEdit_previous = QPlainTextEdit(self.horizontalLayoutWidget_2)
        self.plainTextEdit_previous.setObjectName("plainTextEdit_previous")
        self.horizontalLayout_2.addWidget(self.plainTextEdit_previous)

        self.plainTextEdit_present = QPlainTextEdit(self.horizontalLayoutWidget_2)
        self.plainTextEdit_present.setObjectName("plainTextEdit_present")
        self.horizontalLayout_2.addWidget(self.plainTextEdit_present)

        self.plainTextEdit_next = QPlainTextEdit(self.horizontalLayoutWidget_2)
        self.plainTextEdit_next.setObjectName("plainTextEdit_next")
        self.horizontalLayout_2.addWidget(self.plainTextEdit_next)

        self.label_all = QLabel(self.centralwidget)
        self.label_all.setGeometry(QtCore.QRect(530, 80, 301, 31))
        self.label_all.setObjectName("label_all")
        self.label_previous = QLabel(self.centralwidget)
        self.label_previous.setGeometry(QtCore.QRect(10, 10, 241, 29))
        self.label_previous.setObjectName("label_previous")

        self.listWidget = QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(530, 110, 241, 219))
        self.listWidget.setObjectName("listWidget")

        self.label_next = QLabel(self.centralwidget)
        self.label_next.setGeometry(QtCore.QRect(520, 10, 251, 29))
        self.label_next.setObjectName("label_next")
        # 데이터 저장 버튼
        self.pushButton_save = QPushButton(self.centralwidget)
        self.pushButton_save.setGeometry(QtCore.QRect(160, 270, 93, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.pushButton_save.clicked.connect(self.save_data)

        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1089, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # 파일 열기
        self.actionopen = QAction(self)
        self.actionopen.setObjectName("actionopen")
        self.actionopen.setShortcut("Ctrl+O")
        self.actionopen.triggered.connect(self.openFile)
        # 프로그램 종료
        self.actionExit = QAction(self)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.close)

        # 프로그램 열기 닫기 창 추가
        self.menu.addAction(self.actionopen)
        self.menu.addAction(self.actionExit)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        self.show()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "LDCC Tagger"))
        self.pushButton_iper.setText(_translate("MainWindow", "PER_I"))
        self.pushButton_bcom.setText(_translate("MainWindow", "COM_B"))
        self.pushButton_o.setText(_translate("MainWindow", "O"))
        self.pushButton_bloc.setText(_translate("MainWindow", "LOC_B"))
        self.pushButton_iloc.setText(_translate("MainWindow", "LOC_I"))
        self.pushButton_icom.setText(_translate("MainWindow", "COM_I"))
        self.pushButton_bper.setText(_translate("MainWindow", "PER_B"))
        self.pushButton_baff.setText(_translate("MainWindow", "AFF_B"))
        self.pushButton_iaff.setText(_translate("MainWindow", "AFF_I"))
        self.pushButton_bpos.setText(_translate("MainWindow", "POS_B"))
        self.pushButton_ipos.setText(_translate("MainWindow", "POS_I"))
        self.pushButton_bedu.setText(_translate("MainWindow", "EDU_B"))
        self.pushButton_iedu.setText(_translate("MainWindow", "EDU_I"))
        self.pushButton.setText(_translate("MainWindow", "Blank"))
        self.label_present.setText(_translate("MainWindow", "현재 문장:"))
        self.label_all.setText(_translate("MainWindow", "모든 문장:"))
        self.label_previous.setText(_translate("MainWindow", "이전 문장:"))
        self.label_next.setText(_translate("MainWindow", "다음 문장:"))
        self.pushButton_save.setText(_translate("MainWindow", "Save"))
        self.pushButton_checker.setText(_translate("MainWindow", "Check"))
        self.pushButton_edit.setText(_translate("MainWindow", "Edit"))
        self.pushButton_previous.setText(_translate("MainWindow", "Prev"))
        self.pushButton_pr.setText(_translate("MainWindow", "Present"))
        self.pushButton_next.setText(_translate("MainWindow", "Next"))
        self.menu.setTitle(_translate("MainWindow", "파일"))
        self.actionopen.setText(_translate("MainWindow", "Open"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.filename = fname[0]
        if fname[0]:
            with open(fname[0], "r", encoding="utf-8") as f:
                rdf = f.readlines()
                for i, data in enumerate(rdf):
                    data = data.split()
                    self.data_dict[i] = data
        if len(self.data_dict) < 1:
            buttonReply = QMessageBox.information(
                self, "Tagger", "Data가 존재하지 않습니다.",
                QMessageBox.Ok
            )
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{}|{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

        self.t = self.get_idx()
        self.idx = copy.copy(self.t)
        # self.idx = self.get_idx()
        self.show_previous(self.t)
        self.show_present(self.t)
        self.show_next(self.t)

    def get_idx(self):
        idx = 0
        for i in self.data_dict:
            if len(self.data_dict[i]) == 2:
                if i == 0:
                    return idx
                else:
                    idx = i
                    return idx
        return idx

    def show_present(self, idx):

        if len(self.data_dict[idx]) == 2:
            self.plainTextEdit_present.setPlainText("{}\t{}\t".format(self.data_dict[idx][0], self.data_dict[idx][1]))
        elif len(self.data_dict[idx]) == 3:
            self.plainTextEdit_present.setPlainText(
                "{}\t{}\t{}".format(self.data_dict[idx][0], self.data_dict[idx][1], self.data_dict[idx][2]))
        else:
            self.plainTextEdit_present.setPlainText("")

    def show_next(self, idx):
        if idx < len(self.data_dict):
            if len(self.data_dict[idx + 1]) == 2:
                self.plainTextEdit_next.setPlainText(
                    "{}\t{}\t".format(self.data_dict[idx + 1][0], self.data_dict[idx + 1][1]))
            elif len(self.data_dict[idx + 1]) == 3:
                self.plainTextEdit_next.setPlainText(
                    "{}\t{}\t{}".format(self.data_dict[idx + 1][0], self.data_dict[idx + 1][1],
                                        self.data_dict[idx + 1][2]))
            else:
                self.plainTextEdit_next.setPlainText("")
        else:
            self.plainTextEdit_next.clear()

    def show_previous(self, idx):
        if idx - 1 >= 0:
            self.plainTextEdit_previous.setPlainText(self.show_data(idx - 1, self.data_dict))

        else:
            self.plainTextEdit_previous.clear()

    def b_per_tag_handler(self):
        self.get_tag("PER_B")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)

        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def i_per_tag_handler(self):
        self.get_tag("PER_I")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)

        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def b_loc_tag_handler(self):
        self.get_tag("LOC_B")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)

        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def i_loc_tag_handler(self):
        self.get_tag("LOC_I")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)

        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def b_com_tag_handler(self):
        self.get_tag("COM_B")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def i_com_tag_handler(self):
        self.get_tag("COM_I")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def b_aff_tag_handler(self):
        self.get_tag("AFF_B")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def i_aff_tag_handler(self):
        self.get_tag("AFF_I")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def b_pos_tag_handler(self):
        self.get_tag("POS_B")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")
    def i_pos_tag_handler(self):
        self.get_tag("POS_I")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")
    def b_edu_tag_handler(self):
        self.get_tag("EDU_B")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")
    def i_edu_tag_handler(self):
        self.get_tag("EDU_I")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def o_tag_handler(self):
        self.get_tag("O")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def blank_tag_handler(self):
        self.get_tag("")
        if self.t + 2 < len(self.data_dict):
            self.t += 1
            self.show_previous(self.t)
            self.show_present(self.t)
            self.show_next(self.t)
        else:
            if self.t + 2 == len(self.data_dict):
                self.t += 1
                self.show_previous(self.t)
                self.show_present(self.t)
                self.plainTextEdit_next.setPlainText("*********")
            else:
                self.t += 1
                self.show_previous(self.t)
                self.plainTextEdit_present.setPlainText("*********")
                self.plainTextEdit_next.setPlainText("*********")
                QMessageBox.information(self, "Over Message", "끝났다. 저장합시다.", QMessageBox.Ok)
        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")

    def show_data(self, idx, data):
        if len(data[idx]) == 2:
            return "{}\t{}".format(data[idx][0], data[idx][1])
        elif len(data[idx]) == 3:
            return "{}\t{}\t{}".format(data[idx][0], data[idx][1], data[idx][2])
        else:
            return ""

    def get_present(self):
        self.plainTextEdit_present.setPlainText(self.show_data(self.t, self.data_dict))
        if self.t == 0:
            if self.t < len(self.data_dict):
                self.plainTextEdit_next.setPlainText(self.show_data(self.t + 1, self.data_dict))
        else:
            self.plainTextEdit_previous.setPlainText(self.show_data(self.t - 1, self.data_dict))
            self.plainTextEdit_present.setPlainText(self.show_data(self.t, self.data_dict))
            if self.t < len(self.data_dict):
                self.plainTextEdit_next.setPlainText(self.show_data(self.t + 1, self.data_dict))

    def check_error(self):
        tag_list = ["B_PER", "I_PER", "B_LOC", "I_LOC", "B_EDU", "I_EDU", "B_AFF", "I_AFF", "B_ORG", "I_ORG", "O"]
        self.plainTextEdit_idx.clear()
        self.plainTextEdit_word.clear()
        self.plainTextEdit_checknext.clear()
        self.plainTextEdit_checkprv.clear()
        for i in range(self.t):
            if len(self.data_dict[i]) == 2:
                if i not in self.err_list:
                    self.err_list.append(i)
            elif len(self.data_dict[i]) == 3:
                if self.data_dict[i][2] not in tag_list:
                    if i not in self.err_list:
                        self.err_list.append(i)
            elif len(self.data_dict[i]) == 1:
                if i not in self.err_list:
                    self.err_list.append(i)

        if len(self.err_list) == 0:
            QMessageBox.information(self, "Check Message", "에러가 없어보임", QMessageBox.Ok)
        else:
            err_flag = self.err_list[0]
            self.plainTextEdit_idx.setPlainText("{}".format(err_flag))
            self.plainTextEdit_word.setPlainText("{}".format(self.data_dict[err_flag][0]))
            self.plainTextEdit_present.setPlainText(self.show_data(err_flag, self.data_dict))
            if err_flag == 0:
                self.plainTextEdit_checknext.setPlainText("{}".format(self.data_dict[err_flag + 1][0]))
                self.plainTextEdit_next.setPlainText("{}".format(self.data_dict[err_flag + 1][0]))
            else:
                self.plainTextEdit_checkprv.setPlainText(self.show_data(err_flag - 1, self.data_dict))
                if err_flag < len(self.data_dict):
                    self.plainTextEdit_checknext.setPlainText(self.show_data(err_flag + 1, self.data_dict))
                else:
                    self.plainTextEdit_checknext.setPlainText("")
            self.show_previous(err_flag)
            self.show_present(err_flag)
            self.show_next(err_flag)

    def get_tag(self, tag):
        if len(self.data_dict[self.t]) == 2:
            self.data_dict[self.t].append(tag)
        elif len(self.data_dict[self.t]) == 3:
            self.data_dict[self.t][2] = tag

    def edit_tag(self):
        if self.plainTextEdit_idx.toPlainText() == "":
            QMessageBox.information(self, "Edit Error", "수정해야할 데이터가 없습니다.", QMessageBox.Ok)
        else:
            idx = self.plainTextEdit_idx.toPlainText()
            if self.plainTextEdit_tag.toPlainText() == "":
                QMessageBox.information(self, "Edit Error", "수정해야할 데이터가 없습니다.", QMessageBox.Ok)
            else:
                idx = int(idx)
                self.data_dict[idx][2] = self.plainTextEdit_tag.toPlainText()
                QMessageBox.information(self, "Edit completed",
                                        "{}번째 : {}\t{}\t{} 로 변경되었습니다.".format(self.plainTextEdit_idx.toPlainText(),
                                                                              self.data_dict[idx][0],
                                                                              self.data_dict[idx][1],
                                                                              self.data_dict[idx][2], QMessageBox.Ok))

        self.listWidget.clear()
        for i in self.data_dict:
            if len(self.data_dict[i]) == 0:
                self.listWidget.addItem("")
            elif len(self.data_dict[i]) == 2:
                self.listWidget.addItem("{} {}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1]))
            elif len(self.data_dict[i]) == 3:
                self.listWidget.addItem(
                    "{} {}\t{}\t{}".format(i, self.data_dict[i][0], self.data_dict[i][1], self.data_dict[i][2]))
            else:
                self.listWidget.addItem("ERROR")
        del self.err_list[0]
        self.show_previous(self.t)
        self.show_present(self.t)
        self.show_next(self.t)
        self.plainTextEdit_checkprv.clear()
        self.plainTextEdit_checknext.clear()
        self.plainTextEdit_idx.clear()
        self.plainTextEdit_word.clear()
        self.plainTextEdit_tag.clear()

    def save_data(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            for data in self.data_dict.items():
                if len(data[1]) == 0:
                    f.write("\n")
                elif len(data[1]) == 2:
                    f.write("{}\t{}\n".format(data[1][0], data[1][1]))
                elif len(data[1]) == 3:
                    f.write("{}\t{}\t{}\n".format(data[1][0], data[1][1], data[1][2]))
            QMessageBox.information(self, "Save Message", "Save complete", QMessageBox.Ok)

    def get_previous(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_previous(self.idx)
            self.show_present(self.idx)
            self.show_next(self.idx)
        else:
            QMessageBox.information(self, "Message", "더 이상 뒤로 갈 수 없습니다.", QMessageBox.Ok)

    def get_next(self):
        if self.idx < len(self.data_dict):
            self.idx += 1
            self.show_previous(self.idx)
            self.show_present(self.idx)
            self.show_next(self.idx)
        else:
            QMessageBox.information(self, "Message", "더 이상 앞으로 갈 수 없습니다.", QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = Tagger()
    sys.exit(app.exec_())