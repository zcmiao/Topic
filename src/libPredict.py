#!/usr/bin/env python
#encoding: utf8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import libWordCut as wc
import libTFIDF as ti
import libSQL as db
from gensim import corpora, models, similarities
import pickle
import operator
from multiprocessing import Pool
import multiprocessing
import time
import copy
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cosine
import numpy
import math
import os.path
import libHelper as helper



saveTemplateToFileFlag=True

class predictionParam:
    def __init__(self, PNSh=30, PNSd=240, PNSdLong=240, PperSlot=0.1, PsimTemplateThres=0.8, templateTopSimSize=5, Nh_tp=7, Nl_tp=7, PNShSim=60, PNstep=10, PNstepLong=30, combineCount=1):
        self.PNSh=PNSh
        self.PNSd=PNSd
        self.PNSdLong=PNSdLong
        self.PperSlot=PperSlot
        self.PNShSim=PNShSim
        self.PNstep=PNstep
        self.PNstepLong=PNstepLong
        #3600*0.1=360=6min
        self.PsimTemplateThres=PsimTemplateThres
        self.templateTopSimSize=templateTopSimSize
        self.Nh_tp=Nh_tp
        self.Nl_tp=Nl_tp
        self.SelectedID=[]
        self.comparePreviousRMSECoef=1
        self.step_tp=24
        self.predictCache=None
        self.template=None
        self.combineCount=combineCount


def getTemplate(util_gt, running_time, perSlot=3, step_tp=24, Nh_tp=7, Nl_tp=7, retCache=None, basetime=None, userList=set()):
    if len(userList)==0:
        filename='%s-Nh%s-lasts%s.template_all' %(helper.timestamp2str(running_time),Nh_tp,Nl_tp)
        if os.path.exists(filename):
            return pickle.load(open(filename,'r'))
        
    template_vecs={}
    outretEvent={}
    if basetime==None:
        basetime={}
    
    if len(userList)==0:
        for line in retCache:
            if line['event_id'] not in outretEvent:
                outretEvent[line['event_id']]=[]
            outretEvent.get(line['event_id']).append(line)
            
    else:
        for user in userList:
            if user in retCache:
                for line in retCache[user]:
                    if line['event_id'] not in outretEvent:
                        outretEvent[line['event_id']]=[]
                    outretEvent[line['event_id']].append(line)
    
    for event in util_gt:
        event_id=event.db_event_id
        # corpus_src_init, userInvolveList, outret = db.getText(timePoint=running_time, historySlots= Nh * perSlot, detectionSlots=Nd * perSlot, output_prefix='data/%s_pred_%s_his%sh_ev%s' %(jobID ,helper.timestamp2str(running_time), Nh*perSlot, event_id), userList= selUsers, event_id=event_id, cacheFlag=True)
        if event_id in outretEvent:
            eventContents=outretEvent[event_id]
            if basetime!=None:
                if event_id in basetime:
                    baseTimestamp=basetime[event_id]
                else:
                    baseTimestamp=min(eventContents, key=lambda x:x['dsttime'])['dsttime']
                    # basetime[event_id]=baseTimestamp
            else:
                baseTimestamp=min(eventContents, key=lambda x:x['dsttime'])['dsttime']
                # basetime[event_id]=baseTimestamp
            countDict={}
            templateVec=[]
            for every in eventContents:
                countDict[int((every['dsttime']-baseTimestamp)/(3600*perSlot))]=countDict.get(int((every['dsttime']-baseTimestamp)/(3600*perSlot)), 0)+1
            for i in range(0, int(Nl_tp*step_tp/perSlot)):
                templateVec.append(countDict.get(i, 0))
            # for i in range(0, NSd):
            #     templateVec.append(countDict.get(i, 0))
            # if sum(knownVec)>0:
            template_vecs[event_id]=templateVec
            # GT_vect[event_id]=GTVec
    if len(userList)==0:
        if saveTemplateToFileFlag:
            pickle.dump((template_vecs, basetime), open(filename, 'w'))
        return template_vecs, basetime
    else:
        return template_vecs
    

def getCountRecursive(srcmid, event_id, selUsers, src_dst_dict, user_src_dict, src_user_dict, deep=1):
    outret=[]
    if deep >2:
        return outret
    for dstmid, dsttime in src_dst_dict[(srcmid,event_id)]:
        if dstmid not in src_user_dict or dstmid==srcmid or src_user_dict[dstmid] in selUsers:
            #is leaf or head
            outret.append((dstmid, dsttime))
        # elif src_user_dict[dstmid] in selUsers:
            # outret.append(dsttime)
        else:
            outret.append((dstmid, dsttime))
            outret.extend(getCountRecursive(dstmid, event_id, selUsers, src_dst_dict, user_src_dict, src_user_dict, deep+1))
    return outret
    
#get predict part
def getKnownAndGTPart(util_gt, running_time, selUsers=set(), perSlot=3, NSh= 4, NSd=40, jobID='predict', inputCache=None, basetime=None):
    known_vect={}
    GT_vect={}
    # event_id=0
    if inputCache==None:
        print 'null'

    outretEvent={}
    outret=inputCache
    if outret:
        if isinstance(outret,dict):
            if len(selUsers)==0:
                for user in outret.keys():
                    for line in outret[user]:
                        if line['event_id'] not in outretEvent:
                            outretEvent[line['event_id']]=[]
                        outretEvent.get(line['event_id']).append(line)
            else:
                for user in selUsers:
                    if user in outret:
                        for line in outret[user]:
                            if line['event_id'] not in outretEvent:
                                outretEvent[line['event_id']]=[]
                            outretEvent.get(line['event_id']).append(line)
        else:
            if len(selUsers)==0:
                for line in outret:
                    if line['event_id'] not in outretEvent:
                        outretEvent[line['event_id']]=[]
                    outretEvent.get(line['event_id']).append(line)
            else:
                for line in outret:
                    if line['dstuid'] in selUsers:
                        if line['event_id'] not in outretEvent:
                            outretEvent[line['event_id']]=[]
                        outretEvent.get(line['event_id']).append(line)

    for event in util_gt:
        event_id=event.db_event_id
        # corpus_src_init, userInvolveList, outret = db.getText(timePoint=running_time, historySlots= Nh * perSlot, detectionSlots=Nd * perSlot, output_prefix='data/%s_pred_%s_his%sh_ev%s' %(jobID ,helper.timestamp2str(running_time), Nh*perSlot, event_id), userList= selUsers, event_id=event_id, cacheFlag=True)
        if event_id in outretEvent:
            eventContents=outretEvent[event_id]
            if basetime!=None:
                if event_id in basetime:
                    baseTimestamp=basetime[event_id]
                else:
                    baseTimestamp=min(eventContents, key=lambda x:x['dsttime'])['dsttime']
            else:
                baseTimestamp=min(eventContents, key=lambda x:x['dsttime'])['dsttime']
            countDict={}
            knownVec=[]
            GTVec=[]
            for every in eventContents:
                countDict[int((every['dsttime']-baseTimestamp)/(3600*perSlot))]=countDict.get(int((every['dsttime']-baseTimestamp)/(3600*perSlot)), 0)+1
            for i in range(0, NSh):
                knownVec.append(countDict.get(i, 0))
            for i in range(0, NSd):
                GTVec.append(countDict.get(i+NSh, 0))
            # if sum(knownVec)>0:
            known_vect[event_id]=knownVec
            GT_vect[event_id]=GTVec

    return known_vect, GT_vect

    
def getKnownAndGTAll(corpus_ret, util_gt, perSlot=3, NSh= 4, NSd=40):
    
    known_vect={}
    GT_vect={}
    basetime_all={}
    
    outretEvent={}
    if isinstance(corpus_ret, dict):
        # totalCount=len(corpus_ret)
        for userContent in corpus_ret.values():
            for line in userContent:
                if line['event_id'] not in outretEvent:
                    outretEvent[line['event_id']]=[]
                outretEvent[line['event_id']].append({'dsttime':line['dsttime']})
    else:
        for line in corpus_ret:
            if line['event_id'] not in outretEvent:
                outretEvent[line['event_id']]=[]
            outretEvent[line['event_id']].append({'dsttime':line['dsttime']})
    
    for event in util_gt:
        event_id=event.db_event_id
        # corpus_src_init, userInvolveList, outret = db.getText(timePoint=running_time, historySlots= Nh * perSlot, detectionSlots=Nd * perSlot, output_prefix='data/%s_pred_%s_his%sh_ev%s' %(jobID ,helper.timestamp2str(running_time), Nh*perSlot, event_id), userList= selUsers, event_id=event_id, cacheFlag=True)
        if event_id in outretEvent:
            eventContents=outretEvent[event_id]

            baseTimestamp=min(eventContents, key=lambda x:x['dsttime'])['dsttime']
            countDict={}
            knownVec=[]
            GTVec=[]
            for every in eventContents:
                countDict[int((every['dsttime']-baseTimestamp)/(3600*perSlot))]=countDict.get(int((every['dsttime']-baseTimestamp)/(3600*perSlot)), 0)+1
            for i in range(0, NSh):
                knownVec.append(countDict.get(i, 0))
            for i in range(0, NSd):
                GTVec.append(countDict.get(i+NSh, 0))
            # if sum(knownVec)>0:
            known_vect[event_id]=knownVec
            GT_vect[event_id]=GTVec
            basetime_all[event_id]=baseTimestamp
    return known_vect, GT_vect, basetime_all
    
    
def getVecs(util_gt, running_time, selUsers=set(), perSlot=3, NSh=4, NSd=40, jobID='predict', bypassTemplate=True, inputTemplate=None, outputCache=False, inputCache=None, inputKnownAllCache=None, inputGTAllCache=None, verbose=False, basetime=None):
    if bypassTemplate:
        template_vecs=inputTemplate
    # else:
        # template_vecs= getTemplate(util_gt=util_gt, running_time=running_time, perSlot=perSlot, step_tp=step_tp, Nh_tp=Nh_tp, Nl_tp=Nl_tp, retCache=inputCache)

    known_vecs, gt_vecs= getKnownAndGTPart(util_gt=util_gt, running_time=running_time, selUsers=selUsers, perSlot=perSlot, NSh=NSh , NSd=NSd, jobID=jobID, inputCache=inputCache, basetime=basetime)
    # print 'got known'
    if inputKnownAllCache !=None and inputGTAllCache!=None:
        known_vecs_all=inputKnownAllCache
        gt_vecs_all=inputGTAllCache

    dor= calcDor(gt_vecs, gt_vecs_all, known_vecs, known_vecs_all)
    if verbose:
    # print 'got dor'
        for event in gt_vecs_all.keys():
            print 'gtall',event,gt_vecs_all[event]
        for event in gt_vecs.keys():
            print 'gtpart',event,gt_vecs[event]
        for event in known_vecs_all.keys():
            print 'knownall', event,known_vecs_all[event]
        for event in known_vecs.keys():
            print 'knownpart', event,known_vecs[event]
        for event in dor.keys():
            print 'dor', event,dor[event]
    # if outputCache:
        # return known_vecs, gt_vecs, known_vecs_all, gt_vecs_all, dor, outret
    # else:
    return known_vecs, gt_vecs, dor
    
    
def calcDor(gt_vecs, gt_vecs_all, known_vecs, known_vecs_all):
    dor={}
    for event_id in gt_vecs.keys():
        gt_vec=gt_vecs[event_id]
        gtdor=[]
        gtdor_non_zero_count=0
        gtdor_non_zero_sum=0.0
        for i in range(0,len(gt_vec)):
            if gt_vec[i]==0:
                gtdor.append(-1)
            else:
                # print len(gt_vec), len(gt_vecs_all[event_id])
                this_dor=float(gt_vecs_all[event_id][i])/float(gt_vec[i])
                gtdor.append(this_dor)
                gtdor_non_zero_sum+=this_dor
                gtdor_non_zero_count+=1
        
        if gtdor_non_zero_count>0:
            gtdor_non_zero_avg=float(gtdor_non_zero_sum)/float(gtdor_non_zero_count)
        else:
            gtdor_non_zero_avg=1
        for i in range(0, len(gtdor)):
            if gtdor[i]==-1:
                gtdor[i]=gtdor_non_zero_avg
        
        # gtdor=list(numpy.divide(numpy.add(1e-10,gt_vecs_all[event_id]), numpy.add(1e-10,gt_vecs[event_id])))
        dor[event_id]=gtdor
        #kdor=list(numpy.divide(numpy.add(1e-10,known_vecs_all[event_id]), numpy.add(1e-10,known_vecs[event_id])))

    return dor

def calcDorSum(gt_vecs, gt_vecs_all, known_vecs, known_vecs_all):
    dor={}
    for event_id in gt_vecs.keys():
        gt_vec=gt_vecs[event_id]
        gt_vec_all=gt_vecs_all[event_id]
        gt_sum=[]
        gt_sum_all=[]
        
        for i in range(0, len(gt_vec)):
            pop_part=numpy.sum(gt_vec[:i+1])
            pop_all=numpy.sum(gt_vec_all[:i+1])
            gt_sum.append(pop_part)
            gt_sum_all.append(pop_all)
            
        this_dor=numpy.true_divide(gt_sum_all, gt_sum)
        avg=numpy.mean(numpy.ma.masked_invalid(this_dor))
        for i in range(0, len(gt_vec)):
            if this_dor[i]==numpy.nan or this_dor[i]==numpy.inf:
                this_dor[i]=avg
        
        dor[event_id]=this_dor
        # this_dor=numpy.true_divide(gt_sum_all, gt_sum)
    return dor
    

class perEvent:
    event_id=None
    full_known_vecs=None
    NSh=None
    NSd=None
    topSimSize=5
    simThres=0.8


def vcorrcoef(X,y):
    Xm = numpy.reshape(numpy.mean(X,axis=1),(X.shape[0],1))
    y=numpy.reshape(y, (y.shape[0], 1))
    ym = numpy.mean(y)
    r_num = numpy.sum((X-Xm)*(y-ym), axis=1)
    r_den = numpy.sqrt(numpy.sum(numpy.power((X-Xm),2),axis=1)*numpy.sum(numpy.power((y-ym),2),axis=0))
    r = numpy.true_divide(r_num, r_den)
    return r.T.getA()[0]


def vcosine(X, y):
    y=numpy.matrix(y)
    r=numpy.power(X*y.T, 2)
    d=numpy.sum(numpy.power(X,2), axis=1) * numpy.sum(y*y.T)
    return numpy.sqrt(numpy.true_divide(r, d).T.getA()[0])


def predictOnce(templates, known_vecs_part, known_vecs_all, NSh, NSd, topSimSize=5, simThres=0.8, select=False):
    template_vecs_all, template_vecs_part, basetime=templates
    predicted_vecs_part={}
    predicted_vecs_all={}
    # X, Xindex, Xsp_var, Xsp_sum, special_sum, special_var, X_dict = templates_X
    X=[]
    Xsp_var=[]
    Xsp_sum=[]

    # Xcnt=0
    X_dict={}
    for eid, template_vec_part in template_vecs_part.items():
        for i in range(0, len(template_vec_part)-NSh-NSd+1):
            ax=template_vec_part[i:i+NSh]
            tplax=tuple(ax)
            
            if numpy.sum(ax)==0:
                if tplax not in X_dict:
                    X_dict[tplax]=[(eid,i)]
                    # special_sum_count+=1
                    Xsp_sum.append(ax)
                else:
                    X_dict[tplax].append((eid,i))
            elif numpy.var(ax)==0:
                if tplax not in X_dict:
                    X_dict[tplax]=[(eid,i)]
                    # special_var_count+=1
                    Xsp_var.append(ax)
                else:
                    X_dict[tplax].append((eid,i))
            else:
                if tplax not in X_dict:
                    X_dict[tplax]=[(eid,i)]
                    # X_count+=1
                    X.append(ax)
                else:
                    X_dict[tplax].append((eid,i))
                  
            # Xcnt+=1
    # for x in X:
        # Xset.add(tuple(x))
    # print 'temp_vec_part size=%d, uniq size=%d' %(len(X), len(Xset))
    X_count=len(X)
    special_sum_count=len(Xsp_sum)
    special_var_count=len(Xsp_var)
    X=numpy.matrix(X)
    Xsp_var=numpy.matrix(Xsp_var)
        
    for event_id, known_vec_part in known_vecs_part.items():

        if len(known_vec_part)<NSh:
            NSh=len(known_vec_part)
        known_part = known_vec_part[-NSh:]
        known_all = known_vecs_all[event_id][-NSh:]
        
        y=numpy.array(known_part)
        yvar=numpy.var(y)
        ysum=numpy.sum(y)
        
        if ysum==0:
            corrcoefs=numpy.concatenate((numpy.zeros(X_count), numpy.ones(special_sum_count), numpy.zeros(special_var_count)))
        elif yvar==0:
            corrcoefs=vcosine(X, y)
            corrcoefs=numpy.concatenate((corrcoefs, numpy.zeros(special_sum_count), numpy.ones(special_var_count)))
        else:
            corrcoefs= vcorrcoef(X, y)
            if Xsp_var.shape[1]>0:
                corrcoefs=numpy.concatenate((corrcoefs, numpy.zeros(special_sum_count), vcosine(Xsp_var, y)))
            else:
                corrcoefs=numpy.concatenate((corrcoefs, numpy.zeros(special_sum_count)))
        
        #find the index of top-k corr, no sorting of other
        max_ind=numpy.argpartition(corrcoefs,-min(topSimSize, len(corrcoefs)))[-topSimSize:][::-1]
        # print corrs[max_ind]

        simCount=0
        predicts_all=[]
        predicts_part=[]
        sims=[]
        dors=[]
        
        # template_part_sums=0
        # template_all_sums=0
        # template_part_after_sums=0
        # template_all_after_sums=0
        
        for i in max_ind:
            if len(sims)>=topSimSize:
                break
            corrcoef=corrcoefs[i]
            if corrcoef<=simThres:
                continue
            # sims.append(corrcoef)
            if i >= X_count:
                if (i-X_count)>=special_sum_count:
                    tplax=tuple(numpy.array(Xsp_var)[i-X_count-special_sum_count].tolist())
                    # eid, index = special_var[i-X_count-special_sum_count]
                else:
                    tplax=tuple(Xsp_sum[i-X_count])
                    # eid, index= special_sum[i-X_count]
            else:
                tplax=tuple(numpy.array(X)[i].tolist())
                # eid, index = Xindex[i]
            # print 'i', i, 'corr', corrcoef,  'tuple', tplax
            # simCount += len(X_dict[tplax])
            for eid, index in X_dict[tplax]:
            
            # print corrcoef
                if len(sims)>=topSimSize:
                    break
                simCount +=1
                # print 'tuple +1'
                sims.append(corrcoef)
                template_all=template_vecs_all[eid][index:index+NSh]
                template_all_after=template_vecs_all[eid][index+NSh:index+NSh+NSd]
                template_part=template_vecs_part[eid][index:index+NSh]
                template_part_after=template_vecs_part[eid][index+NSh:index+NSh+NSd]

                known_part_sum=numpy.sum(known_part)
                template_part_sum=numpy.sum(template_part)
                
                # template_part_sums += numpy.sum(template_part)
                # template_all_sums += numpy.sum(template_all)
                # template_part_after_sums += numpy.sum(template_part_after)
                # template_all_after_sums += numpy.sum(template_all_after)
                
                
                if template_part_sum>0:
                    dor=numpy.true_divide(known_part_sum, template_part_sum)
                # if template_part_sum>0:
                #     dor=numpy.true_divide(template_all_after_sum, template_part_after_sum)
                #     dor=numpy.true_divide(template_all_sum, template_part_sum)
                else:
                    dor=1
                if dor==0:
                    dor=1

                predicted_all=numpy.multiply(template_all_after, dor)
                predicted_part=numpy.multiply(template_part_after, dor)
                predicts_all.append(predicted_all)
                predicts_part.append(predicted_part)
                # print 'predi+all', predicted_all

        if simCount > 1:
            predict_all_final=numpy.average(predicts_all, axis=0, weights=sims)
            predict_part_final=numpy.average(predicts_part, axis=0, weights=sims)
        elif simCount==1:
            predict_all_final=predicts_all[0]
            predict_part_final=predicts_part[0]
        else:
            predict_all_final=numpy.zeros(NSd)
            predict_part_final=numpy.zeros(NSd)
        

        predicted_vecs_part[event_id]=predict_part_final
        predicted_vecs_all[event_id]=predict_all_final

    return predicted_vecs_part, predicted_vecs_all
                
                

def predictResult(templates, known_vecs_part, known_vecs_all, gt_vecs_part, gt_vecs_all, NSh, NSd, NShSim=60, Nstep=30, topSimSize=5, simThres=0.8, select=False, combineCount=1):
    runTimes=NSd/Nstep
    known_vecs_all_updated=copy.deepcopy(known_vecs_all)

    # known_vecs_all_updated={}

    for i in range(0, int(runTimes)):
        # print 'round%d' %i
        predicted_vecs_part, predicted_vecs_all = predictOnce(templates, known_vecs_part, known_vecs_all_updated, NSh=NShSim, NSd=Nstep, topSimSize=topSimSize, simThres=simThres)
        # print predicted_vecs_part
        # print predicted_vecs_all
        # 
        for event_id in predicted_vecs_all.keys():
            # print 'partb%d, parte%d, allb%d, alle%d' %(len(known_vecs_part[event_id]), len(predicted_vecs_part[event_id]), len(known_vecs_all_updated[event_id]), len(predicted_vecs_all[event_id]))
            # print 'pred vec', predicted_vecs_all[event_id]
            known_vecs_part[event_id].extend(predicted_vecs_part[event_id])
            known_vecs_all_updated[event_id].extend(predicted_vecs_all[event_id])
    
    #the above contain known and predicted vec, split by NSh.
    rmses_final = RMSEcalc(known_vecs_part, known_vecs_all_updated, gt_vecs_part, gt_vecs_all, NSh, NSd, select, combineCount)
    return rmses_final
    
def RMSEcalc(known_vecs_part, known_vecs_all_updated, gt_vecs_part, gt_vecs_all, NSh, NSd, select=False, combineCount=1):
    rtols= [0.2, 0.8]
    
    errsPart=[]
    errsPartSum=[]
    errsAll=[]
    errsAllSum=[]
    
    inRangesPartDict={}
    inRangesPartSumDict={}
    inRangesAllDict={}
    inRangesAllSumDict={}
    # inRangePPs=[]
    

    bypassCnt=0
    blackEvent=[]
    # blackEvent=[608, 661,667,694,767,819]
    for event_id in known_vecs_part.keys():
        
        # if numpy.sum(known_vec_part[:NSh])<=1 or known_event_id in blackEvent:
            # bypassCnt += 1
            # continue
        # print 'predicted_all', len(predicted_all), 'predicted_part', len(predicted_part)
        # print
        predicted_all=known_vecs_all_updated[event_id][NSh:]
        predicted_part=known_vecs_part[event_id][NSh:]
        gt_part=gt_vecs_part[event_id]#[NSh:]
        gt_all=gt_vecs_all[event_id]#[NSh:]
        
        if combineCount>1:
            splitSize=NSd/combineCount
            predicted_all=numpy.sum(numpy.array_split(predicted_all, splitSize), axis=1)
            predicted_part=numpy.sum(numpy.array_split(predicted_part, splitSize), axis=1)
            gt_all=numpy.sum(numpy.array_split(gt_all, splitSize), axis=1)
            gt_part=numpy.sum(numpy.array_split(gt_part, splitSize), axis=1)

        predicted_part_sum=[]
        predicted_all_sum=[]
        gt_part_sum=[]
        gt_all_sum=[]
        
        for i in range(0,int(NSd/combineCount)):
            predicted_all_sum.append(numpy.sum(predicted_all[:(i+1)]))
            predicted_part_sum.append(numpy.sum(predicted_part[:(i+1)]))
            gt_all_sum.append(numpy.sum(gt_all[:(i+1)]))
            gt_part_sum.append(numpy.sum(gt_part[:(i+1)]))
            
        errsPart.append(numpy.subtract(predicted_part, gt_part))
        errsPartSum.append(numpy.subtract(predicted_part_sum, gt_part_sum))
        errsAll.append(numpy.subtract(predicted_all, gt_all))
        # print 'pre', predicted_all
        # print 'gt ', gt_all
        errsAllSum.append(numpy.subtract(predicted_all_sum, gt_all_sum))
        
        # if not select:
        #     print 'event id=', event_id
        #     print 'pred all', predicted_all
        #     print 'gt all', gt_all
        #     print 'pred allsum', predicted_all_sum
        #     print 'gt all sum', gt_all_sum
        #     print 'pred part', predicted_part
        #     print 'gt part', gt_part
        #     print 'pred partsum', predicted_part_sum
        #     print 'gt all part summ', gt_part_sum

        for rtol in rtols:
            if rtol not in inRangesPartDict:
                inRangesPartDict[rtol]=[]
            if rtol not in inRangesPartSumDict:
                inRangesPartSumDict[rtol]=[]
            if rtol not in inRangesAllDict:
                inRangesAllDict[rtol]=[]
            if rtol not in inRangesAllSumDict:
                inRangesAllSumDict[rtol]=[]

            inRangesPartDict[rtol].append(numpy.isclose(predicted_part, gt_part, rtol=rtol, atol=0))
            inRangesPartSumDict[rtol].append(numpy.isclose(predicted_part_sum, gt_part_sum, rtol=rtol, atol=0))
            inRangesAllDict[rtol].append(numpy.isclose(predicted_all, gt_all, rtol=rtol, atol=0))
            inRangesAllSumDict[rtol].append(numpy.isclose(predicted_all_sum, gt_all_sum, rtol=rtol, atol=0))


    rmsesPart=numpy.sqrt(numpy.mean(numpy.array(errsPart)*numpy.array(errsPart), axis=0))
    rmsesPartSum=numpy.sqrt(numpy.mean(numpy.array(errsPartSum)*numpy.array(errsPartSum), axis=0))
    rmsesAll=numpy.sqrt(numpy.mean(numpy.array(errsAll)*numpy.array(errsAll), axis=0))
    rmsesAllSum=numpy.sqrt(numpy.mean(numpy.array(errsAllSum)*numpy.array(errsAllSum), axis=0))
    
    if select:
        return [0, 0, rmsesAll, rmsesAllSum]
    
    ret=[]
    ret.append(rmsesPart)
    ret.append(rmsesPartSum)
    ret.append(rmsesAll)
    ret.append(rmsesAllSum)
    for rtol in rtols:
        if rtol in inRangesPartDict:
            ret.append(1- numpy.mean(inRangesPartDict[rtol], axis=0))
        if rtol in inRangesPartSumDict:
            ret.append(1- numpy.mean(inRangesPartSumDict[rtol], axis=0))
        if rtol in inRangesAllDict:
            ret.append(1- numpy.mean(inRangesAllDict[rtol], axis=0))
        if rtol in inRangesAllSumDict:
            ret.append(1- numpy.mean(inRangesAllSumDict[rtol], axis=0))
    
    return ret
    


def calcRR(template_vecs, known_vecs, gt_vecs, known_vecs_all, gt_vecs_all, dor, NSh=4, NSd=40, topSimSize=5, simThres=0.8):
    errs=[]
    errsPP=[]
    # above are event-wise
    for known_event_id, one_known_vec in known_vecs.items():
        RRs=[]
        for event_id, template in template_vecs.items():
            if NSh>=len(template):
                continue
            corr=numpy.correlate(template, one_known_vec)
            max_ind = numpy.argmax(corr)
            max_corr=float(corr[max_ind])
        
            max_remain_part=template[max_ind:max_ind+NSh+NSd]
            padding_part=numpy.lib.pad(max_remain_part, (0, NSh+NSd-len(max_remain_part)),'constant',constant_values=0)
            sim=max_corr/float(max(numpy.correlate(template[max_ind:max_ind+NSh], template[max_ind:max_ind+NSh])[0], numpy.correlate(one_known_vec, one_known_vec)[0]))
            #origin ver
            #sim=sim=max_corr**2/float((numpy.correlate(template, template)[0] * numpy.correlate(one_known_vec, one_known_vec)[0]))
            RRs.append([sim, padding_part, max_corr])
    

        RRs=sorted(RRs, key=lambda x:x[0], reverse=True)    
        pred=[]
        pred=numpy.lib.pad(pred, (0, NSd), 'constant',constant_values=0)
        
        simCount=0
        for i in range (0, topSimSize):
            
            w=float(numpy.correlate(one_known_vec, one_known_vec)[0])/float(RRs[i][2])
            pred=numpy.add(pred, numpy.multiply(RRs[i][1][NSh:], w))
            simCount+=1
            # print RRs[i][0],
            if RRs[i][0]<simThres:
                break
        # print ''
        predPP=numpy.divide(pred, simCount)
        pred=numpy.multiply(predPP, dor[known_event_id])
        
        errPP = numpy.subtract(predPP, gt_vecs[known_event_id])
        err = numpy.subtract(pred, gt_vecs_all[known_event_id])
        
        errsPP.append(errPP)
        errs.append(err)

    errPP_matrix=numpy.matrix(errsPP)
    err_matrix=numpy.matrix(errs)

    rmsesPP=[]
    rmses=[]
    rmsesPPSum=[]
    rmsesSum=[]
    
    rmsesSumSlots=[]
    m,n = err_matrix.shape
    # print 'shape is ', n
    for i in range(0, n):
        
        rmsePP=numpy.sqrt(numpy.mean(errPP_matrix[:,i].getA()**2))
        rmse=numpy.sqrt(numpy.mean(err_matrix[:,i].getA()**2))
        
        rmsePPSum=numpy.sqrt(numpy.mean(numpy.sum(errPP_matrix[:,:i+1], axis=1).getA()**2))
        rmseSum=numpy.sqrt(numpy.mean(numpy.sum(err_matrix[:,:i+1], axis=1).getA()**2))
        
        rmsesPP.append(rmsePP)
        rmses.append(rmse)
        rmsesPPSum.append(rmsePPSum)
        rmsesSum.append(rmseSum)
        
        
    return rmsesPP, rmses, rmsesPPSum, rmsesSum


