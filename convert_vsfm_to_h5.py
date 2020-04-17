import numpy as np
import h5py
import cv2
import copyreg
import argparse
import os, math
from imutils import paths
from itertools import zip_longest
import h5py

model = {}

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def readImageASiftFile(filepath):
	# print(filepath)
	with open(filepath, "r") as imgsiftfile:
		lines = imgsiftfile.readlines()
		# drop unneccessary header line
		lines = lines[1:]
		# the second line and every eighth line after are feature details
		kps = []
		# TODO: should desc and temp be nparrs?
		desc = []
		mark = 0
		temp = []
		for line in lines:
			vals = line.lstrip().split(" ")
			if mark is 0:#if mark is for a header line for a keypoint
				kp = cv2.KeyPoint(float(vals[1]), float(vals[0]), float(vals[2]), math.degrees(float(vals[3])))
				kps.append(kp)
			elif mark > 0:
				# each line inbetween is a set of the vector values for the descriptor
				# concatenate the lines of vector values into a temporary array
				temp += list(map(int, vals))

			if mark is 7:
				# TODO: convert array to nparr?
				desc.append(np.asarray(temp))
				temp = []
				mark = 0
			else:
				mark +=1

		# print(kps)

	return np.asarray(kps), np.asarray(desc)


def getNvmModel(lines,startLine = 0, mark = 0, outliers = False):
	global model
	# print(lines[startLine])
	numberOfImages = int(lines[startLine])
	model[mark] = {}
	model[mark]['imgs'] = {}
	model[mark]['pts'] = {}

	if not outliers:
		for idx,x in enumerate(range(startLine,startLine+numberOfImages)):
			model[mark]['imgs'][idx] = lines[x+1].replace("\t"," ").split(" ")[0][:-4]
	else:
		for idx,x in enumerate(range(startLine,startLine+numberOfImages)):
			model[mark]['imgs'][idx] = lines[x+1].split(" ")[0][:-4]


	if not outliers:
		lineWith3DKpValue = startLine+numberOfImages+2
		numberOf3DKeypoints = int(lines[lineWith3DKpValue])
		for idx,line in enumerate(lines[lineWith3DKpValue+1:lineWith3DKpValue+numberOf3DKeypoints+1]):
			model[mark]['pts'][idx] = {}
			vals = line.split(" ")
			measurements = grouper(vals[7:-1],4)
			for jdx,m in enumerate(measurements):
				# print(jdx, m)
				# idx is index of measurement (setid?)
				# jdx is index of point used in measurement
				# img_idx is the index of the image
				# ftr_img is the index of the feature in the image
				model[mark]['pts'][idx][jdx] = {"img_idx":int(m[0]), "ftr_idx":int(m[1])}

		nextModelStartLine = lineWith3DKpValue+numberOf3DKeypoints+4
		# print("model {1} start line:{0}".format(nextModelStartLine, mark))
		try:
			nextVal = int(lines[nextModelStartLine])
		except ValueError as err:
			print("Getting outlier image list...")
			# numberOfImages = int(lines[nextModelStartLine-1])
			mark = -1
			getNvmModel(lines,nextModelStartLine-1, mark, True)
		else:
			# print(nextVal)
			if nextVal != 0:
				mark+=1
				getNvmModel(lines,nextModelStartLine, mark)
			else:
				print("No outliers in model!")
				return model
	else:
		# print(model[50])
		# print(model[-1])
		print("Completed nvm reading, returning model(s)...")
		return model
		

def readnvmFile(filepath):
	modelmark = 0
	with open(filepath, "r") as nvmfile:
		lines = nvmfile.readlines()
		# strip useless header data
		lines = lines[2:]
		return getNvmModel(lines, 0,  modelmark)
		# print(temp)

	# return modelDict

			
def main(args):
	# print(args)
	# get a dictionary of indexes to file names in pipeline
	# and dictionary of features which survived SfM reconstruction
	print("Indexing sfm-nvm results...")
	readnvmFile(args['nvmfile'])
	modelDict = model
	# and a dictionary of filenames to indexes
	# imgDict = {v:k for k,v in idxDict.items()}

	# print(idxDict)
	# print(imgDict)
	# print(vsfmkpDict)

	# load up all the keypoint arrays as "other_keypoints"
	print("Loading Image kp arrays...")
	kp_arrs = {}
	# for img in paths.list_files(args['dir'], ".sift_bak"):
	kpfilename = os.path.join(args["dir"], os.path.basename(os.path.splitext(args['dir'])[0]) + "-kps.h5")
	with h5py.File(kpfilename, "r") as kpfile:
		# print(list(kpfile.keys()))
		for img in paths.list_images(args['dir'], ".jpg"):
			filename = os.path.basename(img).split(".")[0]
			# print(filename)
			kp_arrs[filename] = {"valid_keypoints":[], 
			"valid_ftr_idxs":[],
			"other_keypoints":[], 
			"all_keypoints":[], 
			"path":img}
			# kp_arrs[filename]["kps"] = readImageASiftFile(img)[0]
			# print(filename)
			# print(kpfile[img]['kps'][()])
			kps = kpfile[img]['kps'][()]
			kp_arrs[filename]["kps"] = [cv2.KeyPoint(kp[0], kp[1], kp[2], kp[3], kp[4], int(kp[5]), int(kp[6])) for kp in kps]
			kp_arrs[filename]["valid_mdl_sfmpt_imgftr_idxs"] = []
			if(len(kps) == 0):
				print("{0} has {1} keypoints...".format(img, len(kps)))

	print("Meshing nvm file with image features to create .h5 files for LIFT network...")
	# for each 3d kp which survived the SfM Pipeline:
	for mdl in modelDict:
		vsfmkpDict = modelDict[mdl]['pts']
		idxDict = modelDict[mdl]['imgs']
		# print(idxDict)
		for sfmkpidx in vsfmkpDict:
			# get the list of points which comprise the SfM keypoint
			pts = vsfmkpDict[sfmkpidx]
			# for each point in the list
			for ptidx in pts:
				# get the point index for the image
				pt = pts[ptidx]
				# print(pt)
				# get the image
				img = kp_arrs[idxDict[pt['img_idx']]]
				# get the image keypoint
				imgkp = img["kps"][pt["ftr_idx"]]
				# print(imgkp)
				# format it to match the LIFT network
				kp = [imgkp.pt[0],imgkp.pt[1],imgkp.size, imgkp.angle]
				# print(kp)
				# print(type(kp[2]))
				if(kp[2] >= float(2.0)):
					# print("Keypoint greater than max scale, saving keypoint...")
					kp += [sfmkpidx,mdl]
					# add it to the list of valid keypoints for the image (valid being that it survived SfM)
					img['valid_keypoints'] += [kp]
					# keep track of the feature indexes which are valid to filter out other_keypoints later
					img["valid_ftr_idxs"] += [pt["ftr_idx"]]
					# img["valid_mdl_sfmpt_imgftr_idxs"] += [(mdl,sfmkpidx, pt['ftr_idx'])]
				else:
					kp += [-1,-1]
					img['other_keypoints'] += [kp]
				# img["all_keypoints"] += [kp]
					
				#else skip it

	print("Writing .h5 files...")
	#write the kp-minsc files:
	for img in kp_arrs:
		# print(img)
		# print(kp_arrs[img])
		# kp_arrs[img]["other_keypoints"] = [x for idx,x in enumerate(kp_arrs[img]['kps']) if idx not in kp_arrs[img]['valid_ftr_idxs']]
		# print(kp_arrs[img]['other_keypoints'])
		# fill pointid and setid in with -1
		# for idx,kp in enumerate(kp_arrs[img]['kps']):
		# 	if idx in kp_arrs[img]['valid_ftr_idxs']:
		# 		continue
		# 	else:
		# 		# print(idx, kp.pt[0],kp.pt[1],kp.size,kp.angle)
		# 		kp_arrs[img]['other_keypoints'] += [[kp.pt[0],kp.pt[1],kp.size,kp.angle,-1,-1]]

		# kp_arrs[img]["other_keypoints"] = [list((x.pt[0],x.pt[1],x.size,x.angle,-1,-1)) for x in kp_arrs[img]['other_keypoints']]
		outname = os.path.normpath(os.path.splitext(kp_arrs[img]['path'])[0] + "-kp-minsc-2.0.h5")
		print("Writing {0}".format(outname))
		with h5py.File(outname, "w") as outfile:
			outfile['valid_keypoints'] = kp_arrs[img]['valid_keypoints']
			outfile['other_keypoints'] = kp_arrs[img]['other_keypoints']
			# outfile["valid_mdl_sfmpt_imgftr_idxs"] = kp_arrs[img]['valid_mdl_sfmpt_imgftr_idxs']
			# outfile["all_keypoints"] = kp_arrs[img]['all_keypoints']

	print("Done?")


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-d',"--dir", required=True, help="path to input folder of images with keypoint files")
	ap.add_argument('-n', "--nvmfile", required=True, help="path to nvmfile exported from vsfm")
	# ap.add_argument('-c',"--compute", default=True, help="only compute the keypoints")
	# ap.add_argument('-m',"--match", default=True, help="only match previously computed keypoints")
	# ap.add_argument('-f',"--feature", default='sift-flann', help="detector&matcher type to use")
	# ap.add_argument('-l',"--loglevel", default='WARNING', help="logging level for debug purposes. Default:WARNING")
	args= vars(ap.parse_args())

	main(args)
