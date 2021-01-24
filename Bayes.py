import cv2
from imutils import paths
import imutils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

DIR = 'archive/images/Images/'
images = []
imagePaths = list(paths.list_images(DIR))
def image_to_feature_vector(image, size=(50, 50)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()

print("[INFO] describing images...")

rawImages = []
features = []
labels = []

for (i, imagePath) in enumerate(imagePaths):

	#image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
	image = cv2.imread(imagePath)
	# cv2.imshow('image', image)
	# cv2.waitKey(0)
	try:
		#label = imagePath.split('.')[0]
		label = imagePath.split('_')[0]
		# label = label.split('\\')[0]
	except Exception as e:
		continue


	pixels = image_to_feature_vector(image)

	# cv2.imshow('image', pixels)
	# cv2.waitKey(0)
	rawImages.append(pixels)
	#hist = extract_color_histogram(image)
	labels.append(label)
	#features.append(hist)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
labels = np.array(labels)
#features = np.array(features)

(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
#(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
#	features, labels, test_size=0.25, random_state=42)
#
# gnb = GaussianNB()
# gnb.fit(trainRI, trainRL)
# y_pred = gnb.predict(testRI)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (testRI.shape[0], (testRL != y_pred).sum()))
#
# from sklearn import metrics
# print("Bad answers:")
# print(f'{(((testRL != y_pred).sum())/testRI.shape[0])*100} %')
#
# print("Accuracy:",metrics.accuracy_score(trainRL, y_pred.sum()))
# (trainRI, testRI, trainRL, testRL) = train_test_split(
# 	rawImages, labels, test_size=0.33, random_state=42)
# (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
# 	features, labels, test_size=0.33, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = GaussianNB()
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a k-NN classifer on the histogram
# representations
# print("[INFO] evaluating histogram accuracy...")
# model = GaussianNB()
# model.fit(trainFeat, trainLabels)
# acc = model.score(testFeat, testLabels)
# print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
