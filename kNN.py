# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


def image_to_feature_vector(image, size=(50, 50)):

	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	else:
		cv2.normalize(hist, hist)

	return hist.flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default= 'archive/images/Images',
 	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=3,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


#args['dataset'] = 'C:\\Users\\barto\\Downloads\\dogs-vs-cats\\train\\train'




print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
print(imagePaths)

rawImages = []
features = []
labels = []


for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
	# image = cv2.imread(imagePath)
	# cv2.imshow('image', image)
	# cv2.waitKey(0)
	try:
		#label = imagePath.split('.')[0]
		label = imagePath.rsplit('_', 1)[0]

	except Exception as e:
		continue

	pixels = image_to_feature_vector(image)
	# hist = extract_color_histogram(image)

	rawImages.append(pixels)
	# features.append(hist)
	labels.append(label)

	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))


rawImages = np.array(rawImages)
print(labels)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
# (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
# 	features, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])


model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])

#
# model.fit(trainFeat, trainLabels)
# acc = model.score(testFeat, testLabels)
# print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

