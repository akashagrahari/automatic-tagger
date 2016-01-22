#!/usr/bin/python
#-*- coding: utf-8 -*- 
from nltk.corpus import wordnet as wn
import re
import nltk
import copy
import numpy as np
import math
# crap

# normalizedText = re.sub('[^a-zA-Z0-9]', ' ', text)
# words = normalizedText.split()

# listOfPos = []
# for element in posResult:
# 	listOfPos.append(element[1])
# listOfPos = list(set(listOfPos))

# extract list of words for a list of synsets
# def extractWordFromSynset(list):
# 	newList=[]
# 	for item in list:
# 		newList.append(item.name().split('.')[0])
# 	return newList


# allNouns = extractWordFromSynset(list(wn.all_synsets('n')))
# allVerbs = extractWordFromSynset(list(wn.all_synsets('v')))
# allAdjectives = extractWordFromSynset(list(wn.all_synsets('s')))

# crap

# Read article and make a list of words
impPos = ['JJ','JJR','JJS','NN','NNP','NNPS','NNS','VB','VBD','VBG','VBN','VBP','VBZ']
adjectivePos = ['JJ','JJR','JJS']
nounPos = ['NN','NNP','NNPS','NNS']
verbPos = ['VB','VBD','VBG','VBN','VBP','VBZ']
tags = ['Business', 'Movies','Sports', 'Shopping', 'Health','Government']
tagSynsets = []

for tag in tags:
	tagSynsets.append(wn.synsets(tag)[0])

def getAggregatedDict(inputLayer,inputWeights):
	dict = {}
	for i in range(0,len(inputLayer)):
		indices = [j for j, x in enumerate(inputLayer) if x == inputLayer[i]]
		n = 0
		for index in indices:
			n+=inputWeights[index]
		dict[inputLayer[i]] = (n/len(indices))*2/(1+math.exp(-1*(n-1)))
	return dict

def isSimilarPos(pythonPos, wordnetPos):
	if pythonPos in adjectivePos and wordnetPos == 's':
		return True
	elif pythonPos in nounPos and wordnetPos == 'n':
		return True
	elif pythonPos in verbPos and wordnetPos == 'v':
		return True
	return False

def getInitialInputLayer(sievedWords,sievedPos):
	initialLayer = []
	for i in range(0,len(sievedWords)):
		word = sievedWords[i]
		pos = sievedPos[i]
		try:
			synsets = wn.synsets(word)
		except UnicodeDecodeError:
			pass
		for synset in synsets:
			if isSimilarPos(pos,synset.name().split('.')[1]):
				initialLayer.append(synset)
				break
	return initialLayer

def getOutputLayerDict(inputLayerDict):
	inputLayerKeys = inputLayerDict.keys()
	inputLayerValues = inputLayerDict.values()
	outputLayerKeys = []
	for word in inputLayerKeys:
		try:
			synsets = wn.synsets(word.name().split('.')[0])
		except UnicodeDecodeError:
			pass
		for synset in synsets:
			outputLayerKeys.append(synset)
	# outputLayerKeys = list(set(outputLayerKeys)) # remove duplicates
	outputLayerValues = getOutputWeights(inputLayerKeys,inputLayerValues,outputLayerKeys)
	return dict(zip(outputLayerKeys, outputLayerValues))
	# return getAggregatedDict(outputLayerKeys,outputLayerValues)

def getOutputWeights(inputLayer,inputWeights,outputLayer):
	outputWeights = []
	path=0
	lch=0
	wup=0
	for i in range(0,len(outputLayer)):
		outputWord = outputLayer[i]
		outputWeight = 0
		for j in range(0,len(inputLayer)):
			inputWord = inputLayer[j]
			inputWeight = inputWeights[j]
			try:
				try:
					path = float(inputWord.path_similarity(outputWord))
				except Exception:
					pass
				try:
					lch = float(inputWord.lch_similarity(outputWord)/Math.exp(1))
				except Exception:
					pass
				# try:
				# 	wup = float(inputWord.wup_similarity(outputWord))
				# except Exception:
				# 	pass	
				outputWeight+=(inputWeight*float(path+lch+wup))
			except TypeError:
				outputWeight+=(inputWeight*0.0)
		outputWeights.append(outputWeight)
	return outputWeights

def getNewLayer(outputLayerDict,cutoffThreshold):
	tempDict = {}
	tempDict = copy.deepcopy(outputLayerDict)
	for key,value in tempDict.iteritems():
		if value <= cutoffThreshold:
			del outputLayerDict[key]
	return outputLayerDict

def getCutoffThreshold(outputLayerDict):
	weights = outputLayerDict.values()
	return np.percentile(weights,55)

print "enter file"
file = raw_input()
headerFile = file+"/header.txt"
bodyFile = file+"/body.txt"
with open (headerFile, "r") as myfile:
    headerData=unicode(myfile.readlines())
with open (bodyFile, "r") as myfile:
    bodyData=unicode(myfile.readlines())
headerText=""
bodyText=""

for para in headerData:
	headerText+=para+' ' 
 
for para in bodyData:
	bodyText+=para+' ' 

headerWords = nltk.word_tokenize(headerText)
bodyWords = nltk.word_tokenize(bodyText)
posResultHeader = nltk.pos_tag(headerWords)
posResultBody = nltk.pos_tag(bodyWords)

sievedWordsHeader = []
sievedPosHeader = []
sievedWordsBody = []
sievedPosBody = []

for element in posResultHeader:
	if element[1] in impPos:
		sievedWordsHeader.append(element[0])
		sievedPosHeader.append(element[1])

for element in posResultBody:
	if element[1] in impPos:
		sievedWordsBody.append(element[0])
		sievedPosBody.append(element[1])
inputLayerHeader = getInitialInputLayer(sievedWordsHeader,sievedPosHeader)
inputWeightsHeader = [1.5] * len(inputLayerHeader)
inputLayerBody = getInitialInputLayer(sievedWordsBody,sievedPosBody)
# inputLayer = list(set(inputLayer))
inputWeightsBody = [1] * len(inputLayerBody)
inputLayer = inputLayerHeader+inputLayerBody
inputWeights = inputWeightsHeader+inputWeightsBody
inputLayerDict = dict(zip(inputLayer, inputWeights))
# inputLayerDict = getAggregatedDict(inputLayer,inputWeights)
i=0
# outputLayer = []
# outputWeights = []
while True:
	i=i+1
	try :
		outputLayerDict = getOutputLayerDict(inputLayerDict)
	except UnicodeDecodeError:
		pass
	# print outputLayerDict
	cutoffThreshold = getCutoffThreshold(outputLayerDict)
	# print cutoffThreshold
	inputLayerDict = getNewLayer(outputLayerDict,cutoffThreshold)
	# print inputLayerDict
	if i == 4:
		break

for tag in tagSynsets:
	relevance = 0
	path=0
	lch=0
	wup=0
	for word in inputLayerDict:
		try:
			try:
				path = float(word.path_similarity(tag))
			except Exception:
				pass
			try:
				lch = float(word.lch_similarity(tag)/Math.exp(1))
			except Exception:
				pass
			# try:
			# 	wup = word.wup_similarity(tag)
			# except Exception:
			# 	pass	
			relevance+= float(path+lch+wup)
		except TypeError:
			pass
	print tag
	print relevance
	# outputLayer = list(set(outputLayer))
	# outputWeights = getOutputWeights(inputLayer,inputWeights,outputLayer)
	# inputLayer = chuckOutLowValues(outputLayer)


# for word in sievedWords:
# 	print word
# 	print wn.synsets(word)