from trainingHelper import *

keys = [i for j in (range(33,64),range(96,126)) for i in j] # allowed ascii keys

#Input:
#	Image to run training on
# 	Rectangles: List of rectangles contours
# 	Threshold: threshold transformed area
#	(Optional parameters)
# 		Keys: list of acceptable ascii keys 
#Output:
def manualTrainRects(image,rectangles,thresh,keys=keys):
	samples =  np.empty((0,100))
	responses = []

	for (x,y,w,h) in rectangles:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
		# print (x,y,w,h)

		roi = thresh[y:y+h,x:x+w]
		roismall = cv2.resize(roi,(10,10))

		cv2.imshow('training',image)
		key = cv2.waitKey(0)

		if key == 27:  # (escape to quit)
			sys.exit()
		elif key == 13: # return key
			break
		elif key == 127: # delete key 
			break # functionality not implemeted yet but I want it to remove previous and retrain from previous
		elif key in keys:
			responses.append(key)
			sample = roismall.reshape((1,100))
			samples = np.append(samples,sample,0)
		cv2.rectangle(image,(x,y),(x+w,y+h),(4,115,15),2)
		cv2.imshow('training',image)

	responses = np.array(responses,np.float32)
	responses = responses.reshape((responses.size,1))
	print "Training complete"

	np.savetxt('testsamples.data',samples)
	np.savetxt('testresponses.data',responses)
	print "Saved samples & responses"

#Input:
#	Image: to run training on
# 	Rectangles: List of rectangles contours
# 	Threshold: threshold transformed area
# 	keyName
#	(Optional parameters)
#		keyName: Name of ascii key
#		ketInt: ascii key integer
# 		Keys: list of acceptable ascii keys 
#Output:
def autoTrainRects(image,rectangles,thresh,keyName=chr(127),keyInt=127,keys=keys):
	samples =  np.empty((0,100))
	responses = []
	character = chr(keyInt)

	if keyName != character:
		print "Error character and name does not match!"
	else:
		for (x,y,w,h) in rectangles:
			#cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
			# print (x,y,w,h)

			roi = thresh[y:y+h,x:x+w]
			roismall = cv2.resize(roi,(10,10))

			#cv2.imshow('training',image)
			key = keyInt

			if key == 27:  # (escape to quit)
				sys.exit()
			elif key == 13: # return key
				break
			elif key == 127: # delete key 
				break # functionality not implemeted yet but I want it to remove previous and retrain from previous
				print "Error!!!"
			elif key in keys:
				responses.append(key)
				sample = roismall.reshape((1,100))
				samples = np.append(samples,sample,0)

		responses = np.array(responses,np.float32)
		responses = responses.reshape((responses.size,1))
		print "Training complete for character", character

		np.savetxt('xintest/testAlpha3/'+character+'Samples.data',samples)
		np.savetxt('xintest/testAlpha3/'+character+'Responses.data',responses)
		print "Saved samples & responses for character", character

#Input: 
#	path to image file
#Output:
#	2 data files containing a mapping of responses to samples
def manualRun(fileName):
	im,im3 = readImage(fileName)

	gray = color2gray(im)
	gblur = gaussianBlur(gray)
	thresh = adapThreshold(gblur)
	contours, hierarchy = findCountours(thresh)

	rectangles = findCountourAreas(contours,1) # leave out 1 if not combining tibble
	rectangles = removeOverlaps(rectangles)
	rectangles = sortListedRect(rectangles)
	rectangles = mergeT(rectangles) # combine tibble 
	
	manualTrainRects(im, rectangles, thresh)

#Input: 
#	path to directory
#Output:
#	2 data files containing a mapping of responses to samples
def autoRun(directory):
	allFiles = getAllFiles(directory)
	for files in allFiles:
		print "Running training on", files
		fullpath = directory+'/'+files
		im,im3 = readImage(fullpath)
		justName, asciiKey = readFilename(files)

		gray = color2gray(im)
		gblur = gaussianBlur(gray)
		thresh = adapThreshold(gblur)
		contours, hierarchy = findCountours(thresh)

		rectangles = findCountourAreas(contours,1) # leave out 1 if not combining tibble
		rectangles = removeOverlaps(rectangles)
		rectangles = sortListedRect(rectangles)
		rectangles = mergeT(rectangles) # combine tibble
	
		autoTrainRects(im, rectangles, thresh, justName, asciiKey)

autoRun('MachineLearning/Alphabet')

print "Training completed!"