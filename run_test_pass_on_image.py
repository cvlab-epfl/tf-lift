from subprocess import run
import argparse, os
import itertools as it
from shutil import copytree

def main(args):
	# iterationList = [0,5,10,25,50]
	# hundreds = [1,2,3,4,5,6,7,8,9]
	# iterationList += [h*100 for h in hundreds]
	# thousands = [1,2,3,4,5]
	# iterationList += [t*1000 for t in thousands]
	# theRemainder = list(range(10000, 100000000, 5000))
	# iterationList += theRemainder + [100000000]

	modes = ["kp", "ori", "desc"]
	outTypes = ["txt","txt","h5"]
	params = zip(modes, outTypes)
	fileheader = os.path.basename(args['img'])[:-4]
	outdir = "tests"

	outfiles = []
	for m,o in params:
		outfilename = "{3}-{0}-{1}.{2}".format(fileheader, m, o, os.path.basename(args['modeldir']))
		outfiles.append(outfilename)



	# keypoint
	run("python main.py --task=test --subtask={0} --logdir={1} --test_img_file={2} --test_out_file={3} ".format(modes[0], args['modeldir'], args['img'], outfiles[0]), shell=True)
	# orientation
	run("python main.py --task=test --subtask={0} --logdir={1} --test_img_file={2} --test_out_file={3} --test_kp_file={4} --use_batch_norm=False".format(modes[1], args['modeldir'], args['img'], outfiles[1], outfiles[0]), shell=True)
	# description
	run("python main.py --task=test --subtask={0} --logdir={1} --test_img_file={2} --test_out_file={3} --test_kp_file={4} --use_batch_norm=False".format(modes[2], args['modeldir'], args['img'], outfiles[2],outfiles[1]), shell=True)


	# jointSteps = [10,100,1000]

	# params = list(it.product(iterationList, jointSteps))

	# for idx, i in enumerate(iterationList):
	# 	if i != 0:
	# 		logdirname = "logs\\{0}-{1}".format(args['outnameprefix'], i)
	# 		nextlogdirname = "logs\\{0}-{1}".format(args['outnameprefix'], iterationList[idx+1])
	# 		# print(logdirname)
	# 		iterstep = i - iterationList[idx-1]
	# 		# print(iterstep)
	# 		print("Computing runs from {1}->{0}".format(i, iterationList[idx-1]))
	# 		run("python main.py --task=train --subtask=desc --logdir={0} --max_step={1}".format(logdirname, iterstep), shell=True)
	# 		run("python main.py --task=train --subtask=ori --logdir={0} --max_step={1}".format(logdirname, iterstep), shell=True)
	# 		run("python main.py --task=train --subtask=kp --logdir={0} --max_step={1}".format(logdirname, iterstep), shell=True)
	# 		print("Copying runs as seed for next segment...")
	# 		copytree(logdirname, nextlogdirname)
	# 		# run("python main.py --task==train --subtask==kp --logdir={0} --max_step={2} --data_name={3}".format(logdirname, iterstep, args['dir']), shell=True)
	# 	else:
	# 		continue

	pass

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument('-i',"--img", help="path of image to test", default="P:/neuralnetworkdata/ECCV/piccadilly/2382421_04936eef53_o.jpg")
	ap.add_argument('-m',"--modeldir", help="directory containing models to use in testing", default="release-aug")

	# ap.add_argument('-d',"--dir", help="name of directory which holds files to train on")
	# ap.add_argument('-o',"--outnameprefix", help="prefix of the output models, used in creating the log directories")
	# ap.add_argument('-c',"--compute", default=True, help="only compute the keypoints")
	# ap.add_argument('-m',"--match", default=False, help="only match previously computed keypoints")
	# ap.add_argument('-f',"--feature", default='sift-flann', help="detector&matcher type to use")
	# ap.add_argument('-l',"--loglevel", default='WARNING', help="logging level for debug purposes. Default:WARNING")
	args= vars(ap.parse_args())
	main(args)
