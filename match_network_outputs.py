import argparse, os
import cv2
import h5py
import numpy as np
from common import Timer
from find_obj import init_feature, filter_matches, explore_match

def main(args):
	with h5py.File(args['ah'], "r") as a:
		kps = a['keypoints'][()]
		kp1 = [cv2.KeyPoint(kp[0],kp[1],kp[2],kp[3]) for kp in kps]
		desc1 = a['descriptors'][()]

	with h5py.File(args['bh'], "r") as b:
		kps = b['keypoints'][()]
		kp2 = [cv2.KeyPoint(kp[0],kp[1],kp[2],kp[3]) for kp in kps]
		desc2 = b['descriptors'][()]

	detector, matcher=init_feature("sift")

	with Timer('matching'):
		raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
	p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

	if len(p1) >= 4:
		H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
		print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
		# do not draw outliers (there will be a lot of them)
		kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
	else:
		H, status = None, None
		print('%d matches found, not enough for homography estimation' % len(p1))

	img1 = cv2.imread(args['a'], 0)
	img2 = cv2.imread(args['b'], 0)

	explore_match("LIFT Match", img1, img2, kp_pairs, None, H)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	pass

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("a", help="path of first image file", default="P:/neuralnetworkdata/ECCV/piccadilly/2382421_04936eef53_o.jpg")
	ap.add_argument("ah", help="path of first image keypoints file", default="P:/neuralnetworkdata/ECCV/piccadilly/2382421_04936eef53_o.jpg")
	ap.add_argument("b", help="path of second image file", default="P:/neuralnetworkdata/ECCV/piccadilly/2382421_04936eef53_o.jpg")
	ap.add_argument("bh", help="path of second image keypoints file", default="P:/neuralnetworkdata/ECCV/piccadilly/2382421_04936eef53_o.jpg")
	# ap.add_argument('-m',"--modeldir", help="directory containing models to use in testing", default="release-aug")

	# ap.add_argument('-d',"--dir", help="name of directory which holds files to train on")
	# ap.add_argument('-o',"--outnameprefix", help="prefix of the output models, used in creating the log directories")
	# ap.add_argument('-c',"--compute", default=True, help="only compute the keypoints")
	# ap.add_argument('-m',"--match", default=False, help="only match previously computed keypoints")
	# ap.add_argument('-f',"--feature", default='sift-flann', help="detector&matcher type to use")
	# ap.add_argument('-l',"--loglevel", default='WARNING', help="logging level for debug purposes. Default:WARNING")
	args= vars(ap.parse_args())
	main(args)