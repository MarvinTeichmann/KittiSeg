#!/usr/bin/env python
#
#  THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
#
#  Copyright (C) 2013
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#  Authors: Tobias Kuehnl <tkuehnl@cor-lab.uni-bielefeld.de>
#           Jannik Fritsch <jannik.fritsch@honda-ri.de>
#

import numpy as np
# import pylab
import matplotlib.cm as cm
import os
# import cv2

def make_overlay(image, gt_prob):

    mycm = cm.get_cmap('bwr')

    overimage = mycm(gt_prob, bytes=True)
    output = 0.4*overimage[:,:,0:3] + 0.6*image

    return output


def overlayImageWithConfidence(in_image, conf, vis_channel = 1, threshold = 0.5):
    '''
    
    :param in_image:
    :param conf:
    :param vis_channel:
    :param threshold:
    '''
    if in_image.dtype == 'uint8':
        visImage = in_image.copy().astype('f4')/255
    else:
        visImage = in_image.copy()
    
    channelPart = visImage[:, :, vis_channel] * (conf > threshold) - conf
    channelPart[channelPart < 0] = 0
    visImage[:, :, vis_channel] = 0.5*visImage[:, :, vis_channel] + 255*conf
    return visImage

def evalExp(gtBin, cur_prob, thres, validMap = None, validArea=None):
    '''
    Does the basic pixel based evaluation!
    :param gtBin:
    :param cur_prob:
    :param thres:
    :param validMap:
    '''

    assert len(cur_prob.shape) == 2, 'Wrong size of input prob map'
    assert len(gtBin.shape) == 2, 'Wrong size of input prob map'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
    
    #Merge validMap with validArea
    if validMap is not None:
        if validArea is not None:
            validMap = (validMap == True) & (validArea == True)
    elif validArea is not None:
        validMap=validArea

    # histogram of false negatives
    if validMap is not None:
        fnArray = cur_prob[(gtBin == True) & (validMap == True)]
    else:
        fnArray = cur_prob[(gtBin == True)]
    fnHist = np.histogram(fnArray,bins=thresInf)[0]
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0+len(thres)];
    
    if validMap is not None:
        fpArray = cur_prob[(gtBin == False) & (validMap == True)]
    else:
        fpArray = cur_prob[(gtBin == False)]
    
    fpHist  = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thres)]

    # count labels and protos
    #posNum = fnArray.shape[0]
    #negNum = fpArray.shape[0]
    if validMap is not None:
        posNum = np.sum((gtBin == True) & (validMap == True))
        negNum = np.sum((gtBin == False) & (validMap == True))
    else:
        posNum = np.sum(gtBin == True)
        negNum = np.sum(gtBin == False)
    return FN, FP, posNum, negNum

def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = None):
    '''

    @param totalPosNum: scalar
    @param totalNegNum: scalar
    @param totalFN: vector
    @param totalFP: vector
    @param thresh: vector
    '''

    #Calc missing stuff
    totalTP = totalPosNum - totalFN
    totalTN = totalNegNum - totalFP


    valid = (totalTP>=0) & (totalTN>=0)
    assert valid.all(), 'Detected invalid elements in eval'

    recall = totalTP / float( totalPosNum )
    TNR    = totalTN / float( totalNegNum )
    precision =  totalTP / (totalTP + totalFP + 1e-10)

    accuracy = (totalTP + totalTN) / (float( totalPosNum ) + float( totalNegNum ))
    
    selector_invalid = (recall==0) & (precision==0)
    recall = recall[~selector_invalid]
    precision = precision[~selector_invalid]
        
    maxValidIndex = len(precision)
    
    #Pascal VOC average precision
    AvgPrec = 0
    counter = 0
    for i in np.arange(0,1.1,0.1):
        ind = np.where(recall>=i)
        if ind == None:
            continue
        pmax = max(precision[ind])
        AvgPrec += pmax
        counter += 1
    AvgPrec = AvgPrec/counter
    
    
    # F-measure operation point
    beta = 1.0
    betasq = beta**2
    F = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    index = F.argmax()
    MaxF= F[index]
    
    recall_bst = recall[index]
    precision_bst =  precision[index]

    TP = totalTP[index]
    TN = totalTN[index]
    FP = totalFP[index]
    FN = totalFN[index]
    valuesMaxF = np.zeros((1,4),'u4')
    valuesMaxF[0,0] = TP
    valuesMaxF[0,1] = TN
    valuesMaxF[0,2] = FP
    valuesMaxF[0,3] = FN

    #ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)
    prob_eval_scores  = calcEvalMeasures(valuesMaxF)
    prob_eval_scores['AvgPrec'] = AvgPrec
    prob_eval_scores['MaxF'] = MaxF
    prob_eval_scores['accuracy'] = accuracy

    #prob_eval_scores['totalFN'] = totalFN
    #prob_eval_scores['totalFP'] = totalFP
    prob_eval_scores['totalPosNum'] = totalPosNum
    prob_eval_scores['totalNegNum'] = totalNegNum

    prob_eval_scores['precision'] = precision
    prob_eval_scores['recall'] = recall
    prob_eval_scores['TNR'] = TNR
    #prob_eval_scores['precision_bst'] = precision_bst
    #prob_eval_scores['recall_bst'] = recall_bst
    prob_eval_scores['thresh'] = thresh
    if thresh is not None:
        BestThresh= thresh[index]
        prob_eval_scores['BestThresh'] = BestThresh

    #return a dict
    return prob_eval_scores



def calcEvalMeasures(evalDict, tag  = '_wp'):
    '''
    
    :param evalDict:
    :param tag:
    '''
    # array mode!
    TP = evalDict[:,0].astype('f4')
    TN = evalDict[:,1].astype('f4')
    FP = evalDict[:,2].astype('f4')
    FN = evalDict[:,3].astype('f4')
    Q = TP / (TP + FP + FN)
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N
    A = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    #numSamples = TP + TN + FP + FN
    correct_rate = A

    # F-measure
    #beta = 1.0
    #betasq = beta**2
    #F_max = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    
    
    outDict =dict()

    outDict['TP'+ tag] = TP
    outDict['FP'+ tag] = FP
    outDict['FN'+ tag] = FN
    outDict['TN'+ tag] = TN
    outDict['Q'+ tag] = Q
    outDict['A'+ tag] = A
    outDict['TPR'+ tag] = TPR
    outDict['FPR'+ tag] = FPR
    outDict['FNR'+ tag] = FNR
    outDict['PRE'+ tag] = precision
    outDict['REC'+ tag] = recall
    outDict['correct_rate'+ tag] = correct_rate
    return outDict

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)
        
def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

#     COLORMAP = {
#         'r': {'marker': None, 'dash': (None,None)},
#         'g': {'marker': None, 'dash': [5,2]},
#         'm': {'marker': None, 'dash': [11,3]},
#         'b': {'marker': None, 'dash': [6,3,2,3]},
#         'c': {'marker': None, 'dash': [1,3]},
#         'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
#         'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
#         }
    COLORMAP = {
        'r': {'marker': "None", 'dash': ("None","None")},
        'g': {'marker': "None", 'dash': [5,2]},
        'm': {'marker': "None", 'dash': [11,3]},
        'b': {'marker': "None", 'dash': [6,3,2,3]},
        'c': {'marker': "None", 'dash': [1,3]},
        'y': {'marker': "None", 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': ("None","None")} #[1,2,1,10]}
        }

    for line in ax.get_lines():
        origColor = line.get_color()
        #line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)
        
def plotPrecisionRecall(precision, recall, outFileName, Fig=None, drawCol=1, textLabel = None, title = None, fontsize1 = 24, fontsize2 = 20, linewidth = 3):
    '''
    
    :param precision:
    :param recall:
    :param outFileName:
    :param Fig:
    :param drawCol:
    :param textLabel:
    :param fontsize1:
    :param fontsize2:
    :param linewidth:
    '''
                      
    clearFig = False  
           
    if Fig == None:
        Fig = pylab.figure()
        clearFig = True
        
    #tableString = 'Algo avgprec Fmax prec recall accuracy fpr Q(TonITS)\n'
    linecol = ['g','m','b','c']
    #if we are evaluating SP, then BL is available
    #sectionName = 'Evaluation_'+tag+'PxProb'
    #fullEvalFile = os.path.join(eval_dir,evalName)
    #Precision,Recall,evalString = readEvaluation(fullEvalFile,sectionName,AlgoLabel)

    pylab.plot(100*recall, 100*precision, linewidth=linewidth, color=linecol[drawCol], label=textLabel)


    #writing out PrecRecall curves as graphic
    setFigLinesBW(Fig)
    if textLabel!= None:
        pylab.legend(loc='lower left',prop={'size':fontsize2})
    
    if title!= None:
        pylab.title(title, fontsize=fontsize1)
        
    #pylab.title(title,fontsize=24)
    pylab.ylabel('PRECISION [%]',fontsize=fontsize1)
    pylab.xlabel('RECALL [%]',fontsize=fontsize1)
    
    pylab.xlim(0,100)
    pylab.xticks( [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      ('0','','20','','40','','60','','80','','100'), fontsize=fontsize2 )
    pylab.ylim(0,100)
    pylab.yticks( [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      ('0','','20','','40','','60','','80','','100'), fontsize=fontsize2 )
    pylab.grid(True)
   
    # 
    if type(outFileName) != list:
        pylab.savefig( outFileName )
    else:
        for outFn in outFileName:
            pylab.savefig( outFn )
    if clearFig:
        pylab.close()
        Fig.clear()
   


def saveBEVImageWithAxes(data, outputname, cmap = None, xlabel = 'x [m]', ylabel = 'z [m]', rangeX = [-10, 10], rangeXpx = None, numDeltaX = 5, rangeZ = [7, 62], rangeZpx = None, numDeltaZ = 5, fontSize = 16):
    '''
    
    :param data:
    :param outputname:
    :param cmap:
    '''
    aspect_ratio = float(data.shape[1])/data.shape[0]
    fig = pylab.figure()
    Scale = 8
    # add +1 to get axis text
    fig.set_size_inches(Scale*aspect_ratio+1,Scale*1)
    ax = pylab.gca()
    #ax.set_axis_off()
    #fig.add_axes(ax)
    if cmap != None:
        pylab.set_cmap(cmap)
    
    #ax.imshow(data, interpolation='nearest', aspect = 'normal')
    ax.imshow(data, interpolation='nearest')
    
    if rangeXpx == None:
        rangeXpx = (0, data.shape[1])
    
    if rangeZpx == None:
        rangeZpx = (0, data.shape[0])
        
    modBev_plot(ax, rangeX, rangeXpx, numDeltaX, rangeZ, rangeZpx, numDeltaZ, fontSize, xlabel = xlabel, ylabel = ylabel)
    #plt.savefig(outputname, bbox_inches='tight', dpi = dpi)
    pylab.savefig(outputname, dpi = data.shape[0]/Scale)
    pylab.close()
    fig.clear()
    
def modBev_plot(ax, rangeX = [-10, 10 ], rangeXpx= [0, 400], numDeltaX = 5, rangeZ= [8,48 ], rangeZpx= [0, 800], numDeltaZ = 9, fontSize = None, xlabel = 'x [m]', ylabel = 'z [m]'):
    '''

    @param ax:
    '''
    #TODO: Configureabiltiy would be nice!
    if fontSize==None:
        fontSize = 8
 
    ax.set_xlabel(xlabel, fontsize=fontSize)
    ax.set_ylabel(ylabel, fontsize=fontSize)
        
    zTicksLabels_val = np.linspace(rangeZpx[0], rangeZpx[1], numDeltaZ)
    ax.set_yticks(zTicksLabels_val)
    #ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800])
    xTicksLabels_val = np.linspace(rangeXpx[0], rangeXpx[1], numDeltaX)
    ax.set_xticks(xTicksLabels_val)
    xTicksLabels_val = np.linspace(rangeX[0], rangeX[1], numDeltaX)
    zTicksLabels = map(lambda x: str(int(x)), xTicksLabels_val)
    ax.set_xticklabels(zTicksLabels,fontsize=fontSize)
    zTicksLabels_val = np.linspace(rangeZ[1],rangeZ[0], numDeltaZ)
    zTicksLabels = map(lambda x: str(int(x)), zTicksLabels_val)
    ax.set_yticklabels(zTicksLabels,fontsize=fontSize)
    
 