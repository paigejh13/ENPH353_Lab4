#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

## Documentation for My_App.
# This is the class definition for My_App which utilizes SIFT and Flann algorithms 
# to perform image recognition. The app will have user select image to match and 
# try to match it to some input from the webcam. If it finds the input image, it 
# will draw a blue box around the image in the webcam stream


class My_App(QtWidgets.QMainWindow):
	
	##minimum number of matching key points that must be found to identify a match
	#must be > 4
	MIN_MATCH_NUMBER = 10

	## The constructor
	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self._cam_id = 0
		self._cam_fps = 10
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)
	
	##Gives functionality to browse button and allows user to choose image
	#@param self: the object pointer	
	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)
		print("Loaded template image file: " + self.template_path)

	##Converts an image from cv format to pixmap format
	#@param self: the object pointer
	#@param cv_img: the image to convert
	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	##Matches chosen image with camera input (if the chosen image is also present in the webcam frame)
	#@param self: the object pointer
	#output: blue box around matched image in webcam frame if match found, otherwise, nothing	
	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()


		img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
	
		sift = cv2.xfeatures2d.SIFT_create()
		kp_image, desc_image = sift.detectAndCompute(img, None)

		# Feature matching
		index_params = dict(algorithm=0, trees=5)
		search_params = dict()
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
		kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
		matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
		good_points = []
		for m, n in matches:
			if m.distance < 0.6 * n.distance:
				good_points.append(m)

		query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
		train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
		if len(good_points) > MIN_MATCH_NUMBER:
			matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
			matches_mask = mask.ravel().tolist()

			h, w = img.shape
			pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
			dst = cv2.perspectiveTransform(pts, matrix)

			homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
			
			pixmap = self.convert_cv_to_pixmap(homography)
			self.live_image_label.setPixmap(pixmap)
		else:
			pixmap = self.convert_cv_to_pixmap(frame)
			self.live_image_label.setPixmap(pixmap)

	##Gives functionality to enable camera button and allows user to toggle carmera on and off
	#@param self: the object pointer
	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())




