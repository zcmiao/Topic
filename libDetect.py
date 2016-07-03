#!/usr/bin/env python
#encoding: utf8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import libWordCut as wc
import libTFIDF as ti
import libSQL as db
import libPredict as pr
from gensim import corpora, models, similarities
import pickle
import operator
from multiprocessing import Pool,Value
import multiprocessing
import time
import copy
from scipy.stats.stats import pearsonr
import numpy
import math
import libHelper as helper

phonyRMSE=10000

class detectParam:
    def __init__(self, detectThres=3, isNewDocThres=0.3, userSimThres=0.999, userSimCountCoef=0.999, ergCoef=0.01, costCoef=0.7, baCoef= -1e-2, recallDetectedCountThres=3):
        self.ergCoef=ergCoef
        self.costCoef=costCoef
        self.baCoef=baCoef
        self.detectThres=detectThres
        self.isNewDocThres=isNewDocThres
        self.userSimThres=userSimThres
        self.userSimCountCoef=userSimCountCoef
        self.recallDetectedCountThres=recallDetectedCountThres


class detectParamOnline:
    def __init__(self, isNewDocThres=0.25, onlineFilterExistThres=0.3, onlineCombineThres=0.25, recallDetectedCountThres=3, precDetectedCountThres=3):
        self.isNewDocThres=isNewDocThres
        self.onlineFilterExistThres=onlineFilterExistThres
        self.onlineCombineThres=onlineCombineThres
        
        self.precDetectedCountThres=precDetectedCountThres
        self.recallDetectedCountThres=recallDetectedCountThres
        

#selType:
# em: eventEmerg / ev: eventCount
# c: user cost or not / b: cost boundary or not / d: doc or content match / p: predict when select

# cexv : swc
# embd : jnt with lambda =0 
# embdp: jnt with lambda >0


class TypeFlag:
    def __init__(self, selType):
        if selType.find('em')>-1:
            self.emFlag=True
        else:
            self.emFlag=False
        if selType.find('c')>-1:
            self.costFlag=True
        else:
            self.costFlag=False
        if selType.find('b')>-1:
            self.bondFlag=True
        else:
            self.bondFlag=False
        if selType.find('d')>-1:
            self.docFlag=True
        else:
            self.docFlag=False
        if selType.find('p')>-1:
            self.preFlag=True
        else:
            self.preFlag=False
        if selType.find('x')>-1:
            self.extFlag=True
        else:
            self.extFlag=False
        print 'em=%s cost=%s bond=%s docSelect=%s preSelect=%s extSelect=%s' %(self.emFlag, self.costFlag, self.bondFlag, self.docFlag, self.preFlag, self.extFlag)

def erg(util, detectParam):
    if util.detectCount>=util.detectThres:
        if detectParam.ergCoef == 0:
            return 0
        return float(detectParam.ergCoef) * float(1) / float(util.detectCount)
    else:
        return float(1-detectParam.ergCoef) * float(util.detectThres - util.detectCount)/float(util.detectThres)

def costBond(left_c, left_k, costCoef):
    if left_c<=0:
        return 0
    bond = float(left_c)/float((left_k**costCoef)) + float(left_c)/float(left_k)
    if left_c<=bond:
        return left_c
    else:
        return bond
    

def userSim(user, selUsers, thres=0.6, countThres=5):
    simCount=0
    for selUser in selUsers:
        if len(selUser[2])>1: #gt vector dim>1
            a=[]
            b=[]
            for i,event_gt in enumerate(selUser[2]):
                a.append(event_gt.detectCount)
                b.append(user[2][i].detectCount)
            sim = pearsonr(a,b)[0]
            # sim = pearsonr(zip(*selUser[2]).detectCount, zip(*user[2]).detectCount)[0]
            #print sim
            #is nan, so all count=0, not valid user, return sim=true to bypass it if all zero
            if numpy.isnan(sim):
                if sum(b)>0:
                    return False
                else:
                    return True 
            if sim>thres:
                simCount+=1
                if simCount>=countThres:
                    return True
    return False


def countUtilGT(util_gt, corpus_ret, userList):
    updated_util_gt_offline_new=copy.deepcopy(util_gt)
    gt_count={}
    for util in updated_util_gt_offline_new:
        gt_count[util.db_event_id]=0
        
    for userid, usercontent in corpus_ret.items():
        if userid not in userList:
            print '%s not in userlist' %userid
        else:
            for line in usercontent:
                gt_count[line['event_id']]= gt_count.get(line['event_id'], 0)+1
    for util in updated_util_gt_offline_new:
        util.detectCount=gt_count[util.db_event_id]
    return updated_util_gt_offline_new


#uitl_gt=[(event_count, event_erg, event_id)]
def detectUtilWithGT(corpus_detect, dictionary, util_gt, corpus_gt, tfidf_gt, index_gt, runFlags, user_cost, detectParam, countOnlyOnceFlag=True):
    delta_score=0.0    
    updated_util_gt=copy.deepcopy(util_gt)
    detected_event_index=set()
    for query_vec in corpus_detect:
        if query_vec:
            
            # print 'sim1:', query_vec
            sim = ti.query_tfidf(query_vec, dictionary, corpus_gt, tfidf_gt, index_gt, updateIndex=False, best=1, verbose=False)
            # print 'sim2:', sim
            if not ti.isNewDoc(sim, verbose=False, thres=detectParam.isNewDocThres):
                #print 'sim3:', sim
                event_index = sim[0][0]
                if countOnlyOnceFlag:
                    if event_index not in detected_event_index:
                        detected_event_index.add(event_index)
                        # event_count = updated_util_gt[event_index][0]
                        # event_erg = updated_util_gt[event_index][1]
                        if runFlags.emFlag:
                            event_erg = erg(updated_util_gt[event_index], detectParam=detectParam)
                            delta_score +=  float(event_erg)
                        else:
                            delta_score +=  1
                        updated_util_gt[event_index].detectCount += 1
                    else:
                        continue
                else:
                    updated_util_gt[event_index].detectCount += 1
    if runFlags.costFlag:
        if user_cost==0:
            delta_score = 1e5 #delta_score * 1e5 #or =0 ?
        else:
            delta_score=  delta_score / float(user_cost)
            
    return updated_util_gt, delta_score

def detectUtilWithDoc(user_corpus, util_gt, runFlags, user_cost, detectParam, countOnlyOnceFlag=True):
    delta_score=0.0    
    updated_util_gt=copy.deepcopy(util_gt)

    gt_index={}
    detected_event_index=set()
    for i, event in enumerate(updated_util_gt):
        gt_index[event.db_event_id]=i
    
    for row in user_corpus:
        event_id=row['event_id']
        if event_id not in gt_index:
            continue
            
        event_index=gt_index[event_id]
        
        if countOnlyOnceFlag:
            if event_index in detected_event_index:
                continue
            # event_count = updated_util_gt[event_index][0]
        detected_event_index.add(event_index)
        
        if runFlags.emFlag:
            event_erg = erg(updated_util_gt[event_index], detectParam=detectParam)
            delta_score +=  float(event_erg)
        else:
            delta_score +=  1
        updated_util_gt[event_index].detectCount += 1


    if runFlags.costFlag:
        if user_cost==0:
            delta_score = 1e5 #delta_score * 1e5 #or =0 ?
        # else:
        delta_score=  delta_score / float(user_cost)
    return updated_util_gt, delta_score


def updateScore(user_util, util_gt, base_util, user_cost, runFlags, detectParam):
    delta_score=0.0
    # updated_util=copy.deepcopy(util_gt)
    for i,event_gt in enumerate(util_gt):
        if runFlags.extFlag:
            if event_gt.detectCount <=0:
                if (user_util[i].detectCount-base_util[i].detectCount)>0:
                    if runFlags.emFlag:
                        delta_score += float(erg(event_gt, detectParam=detectParam))
                    else:
                        delta_score += 1
                    user_util[i].detectCount=1
            else:
                user_util[i].detectCount=1
                        
        else:
            if (user_util[i].detectCount-base_util[i].detectCount)>0:
                if runFlags.emFlag:
                    delta_score += float(erg(event_gt, detectParam=detectParam))
                else:
                    delta_score += 1
            if (event_gt.detectCount-base_util[i].detectCount)>0:
                user_util[i].detectCount +=1
            # updated_util[i][1] = erg(updated_util[i][0], detectParam=detectParam)
            # user_util[i][0] -= 1
    if runFlags.costFlag:
        if user_cost==0:
            user_cost=1e5
            # delta_score = 0 #delta_score * 1e5 #or =0 ?
        # else:
        delta_score=  delta_score / float(user_cost)
    return user_util, delta_score

    
def parSelectInit(*args):
    global parCount,totalCount,leftCount
    parCount,totalCount,leftCount=args
    
def parSelectInitP(*args):
    global parCount,totalCount,leftCount
    global known_vecs_all_user,gt_vecs_all_user,basetime_all_user,PParam_user
    parCount,totalCount,leftCount,known_vecs_all_user,gt_vecs_all_user,basetime_all_user,PParam_user=args
    
def parSelect(user):
    global parCount,totalCount,leftCount
   

    if len(user.userInv)<1:
        return user.uid, -1.0

    if not user.runFlags.docFlag:
        user.dictionary, corpus_detect, metas = ti.corpus_init(user.corpus_src_detect, user.dictionary, prefix='model_sel/%s_user_%s_%s_his%sh_evall' %(user.jobID, user.uid, helper.timestamp2str(user.running_time), user.Nh*user.step), verbose=False, outFlag=False, updateDict=False)
        tmp_util_gt, user_score = detectUtilWithGT(corpus_detect, user.dictionary, user.util_gt, user.corpus_gt, user.tfidf_gt, user.index_gt, user.runFlags, user.cost, user.detectParam)
    else:
        tmp_util_gt, user_score = detectUtilWithDoc(user.corpus_src_detect, user.util_gt, user.runFlags, user.cost, user.detectParam)


    countPoint=200
    if user.runFlags.preFlag:
        #if nothing is detected, no need to calc preRMSE
        if user_score<=0:
            # countPoint=10
            currentPredictionRMSES=[[phonyRMSE], [phonyRMSE], [phonyRMSE], [phonyRMSE]]
        else:
            global known_vecs_all_user,gt_vecs_all_user,basetime_all_user
            global PParam_user

            known_vecs_part_user, gt_vecs_part_user= pr.getKnownAndGTPart(util_gt=user.util_gt, running_time=user.running_time, selUsers=PParam_user.SelectedID.union(set([user.uid])), perSlot=PParam_user.PperSlot, NSh=PParam_user.PNSh , NSd=PParam_user.PNSd, jobID='predict', inputCache=PParam_user.predictCache, basetime=basetime_all_user)
            template_part_user = pr.getTemplate(user.util_gt, user.running_time, perSlot=PParam_user.PperSlot, step_tp=PParam_user.step_tp, Nh_tp=PParam_user.Nh_tp, Nl_tp=PParam_user.Nl_tp, retCache=PParam_user.predictCache, userList=PParam_user.SelectedID.union(set([user.uid])), basetime=PParam_user.template[2])
            rmses= pr.predictResult((PParam_user.template[0], template_part_user, PParam_user.template[2]), known_vecs_part_user, known_vecs_all_user, gt_vecs_part_user, gt_vecs_all_user, NSh=PParam_user.PNSh, NSd=PParam_user.PNSd, NShSim=PParam_user.PNShSim, Nstep=PParam_user.PNstep, topSimSize=PParam_user.templateTopSimSize, simThres=PParam_user.PsimTemplateThres, select=True, combineCount=PParam_user.combineCount)
            currentPredictionRMSES= rmses

    parCount.value +=1
    if(parCount.value % countPoint == 0):
        if user.runFlags.preFlag:
            print '%s/%s, left %s. user %s score %s rmse %s' %(parCount.value,totalCount.value,leftCount.value, user.uid,user_score, getRMSEMap(currentPredictionRMSES))
        else:
            print '%s/%s, left %s. user %s score %s' %(parCount.value,totalCount.value,leftCount.value, user.uid,user_score)

        sys.stdout.flush()
    if user.runFlags.preFlag:
        return user.uid, user_score, tmp_util_gt, currentPredictionRMSES
    else:
        return user.uid, user_score, tmp_util_gt


class perUser:
    uid=None
    # count=None
    # info=''
    running_time=None
    Nh=0
    # Nl=0
    step=1
    jobID='data_sel/tmp'
    # event_id=0
    # corpus_ret=None
    corpus_src_detect=None
    userInv=None
    dictionary=None
    cost=0
    # userInfoDict={}
    util_gt=[]
    corpus_gt=None
    tfidf_gt=None
    index_gt=None
    runFlags=None
    detectParam=None
    # previousPredictionRMSE=None
    PParam=None
    # PselectID=None
    # template_vecs=None

def getRMSEMap(previousPredictionRMSES, index=2):
    rmse_indexed=numpy.array(previousPredictionRMSES[index])
    if rmse_indexed.size>1:
        return rmse_indexed
    else:
        return rmse_indexed

def getRMSEMapWithSplit(previousPredictionRMSES, splitTimes=1, index=3, outSize=-1):
    # splitTime=6
    result=[]
    sumRMSE=numpy.array(previousPredictionRMSES[index])
    # print sumRMSE
    mean=numpy.mean(sumRMSE)
    if numpy.isnan(mean) or sumRMSE.size<splitTimes:
        if sumRMSE.size==1:
            return [sumRMSE]
        else:
            return sumRMSE
        for i in range(0,splitTimes):
            result.append(phonyRMSE)
        return result
   
    splitPoints=[]
   
    if splitTimes>1:
        for i in range(1,splitTimes):
            splitPoints.append(int(len(sumRMSE)*i/splitTimes))
        splitSumRMSE=numpy.split(sumRMSE,splitPoints)
    else:
       splitSumRMSE=[numpy.array(sumRMSE)]
   
    for i in range(0,splitTimes):
        mean=numpy.mean(splitSumRMSE[i])
    #     if numpy.isnan(mean):
    #         result.append(phonyRMSE)
    #     else:
        result.append(mean)
    #
    if outSize<0:
        return result
    else:
        print result[0]
        return result[0]

def selectUserOneStep(left_c, left_k, involveUsers, runFlags, userInfoDict, corpus_ret, dictionary, corpus_gt, tfidf_gt, index_gt, util_gt, running_time, step=1, Nh=0, Nl=0, jobID='tmp_user', eventlist=set(), selectSize=1, detectParam=None, predictionParam=None):
    base_util=copy.deepcopy(util_gt)
    if runFlags.preFlag:
        PParam=predictionParam
        # PselectID=set(PParam.SelectedID)
        PParam.predictCache=corpus_ret

        known_vecs_all, gt_vecs_all, basetime_all = pr.getKnownAndGTAll(corpus_ret, util_gt, perSlot=PParam.PperSlot, NSh= PParam.PNSh, NSd=PParam.PNSd)

    all_users=[]
    userScores=[]
    cnt = 0
    total_cnt = 0
    split_size=10000
    for userID in involveUsers:
        cnt += 1
        total_cnt += 1
        user = perUser()
        user.uid=userID
        user.running_time=running_time
        user.Nh=Nh
        # user.Nl=Nl
        user.step=step
        user.cost=userInfoDict[user.uid]['cost']
        # user.corpus_ret=corpus_ret
        user.util_gt=util_gt        
        user.runFlags=runFlags
        user.detectParam=detectParam
        if not runFlags.docFlag:
            user.jobID=jobID
            # user.eventlist=eventlist
            uset=set()
            uset.add(user.uid)
            user.corpus_src_detect, user.userInv = db.getText(timePoint=running_time, historySlots= Nh * step, timelastsSlots=Nl * step, output_prefix='data_sel/%s_user_%s_%s_his%sh_lasts%sh_evl0' %(jobID, user.uid, helper.timestamp2str(running_time), Nh*step, Nl*step), userList=uset, event_id=set(), verboseFlag=False,cacheRet=corpus_ret)
            user.dictionary=dictionary
            user.corpus_gt=corpus_gt
            user.tfidf_gt=tfidf_gt
            user.index_gt=index_gt
        else:
            # user.jobID=jobID
            uset=set()
            uset.add(user.uid)
            user.userInv=uset
            user.corpus_src_detect=corpus_ret.get(user.uid,[])

        all_users.append(user)

        
        if runFlags.extFlag:
            if total_cnt< len(involveUsers):
                continue
            else:
                # ext_cnt=0
                global parCount,totalCount,leftCount
                parCount = Value('i', 0)
                totalCount = Value('i', len(involveUsers))
                leftCount = Value('i', left_k)
            
                for us in all_users:
                    ret=parSelect(us)
                    userScores.append(ret)

        else:
            if cnt <split_size and total_cnt < len(involveUsers):
                continue
            else:
            
                cnt=0
                parCount = Value('i', len(userScores))
                totalCount = Value('i', len(involveUsers))
                leftCount = Value('i', left_k)
    
                workerThres=20
                workerCount=multiprocessing.cpu_count()
                if workerCount > workerThres:
                    workerCount = workerThres
                if runFlags.preFlag:
                    pool = Pool(processes = workerCount, maxtasksperchild = 1600, initializer = parSelectInitP, initargs = (parCount, totalCount, leftCount, known_vecs_all, gt_vecs_all, basetime_all, PParam, ))
                    chunk, extra= divmod(len(all_users), 40)
                    if extra:
                        chunk+=1
                    it=pool.imap_unordered(parSelect, all_users, chunk)
                else:
                    pool = Pool(processes = workerCount, maxtasksperchild = 4800, initializer = parSelectInit, initargs = (parCount, totalCount, leftCount ))
                    chunk, extra= divmod(len(all_users), 8)
                    if extra:
                        chunk+=1
                    it=pool.imap_unordered(parSelect, all_users, chunk)#


                for ret in it:
                    userScores.append(ret)
                pool.close()
                pool.join()
                all_users=[]
                pool=None

    
    blackUsers=set()
    for user in userScores:
        if user[1]<=0:
            blackUsers.add(user[0])
        elif runFlags.preFlag:
            tempRMSE=numpy.mean(getRMSEMap(user[3]))
            if tempRMSE>phonyRMSE:
                blackUsers.add(user[0])
            # if numpy.isnan(tempRMSE):
                # user[3][3]=phonyRMSE
    userScores=[user for user in userScores if user[0] not in blackUsers]
    
    if runFlags.preFlag:
        userScoresSorted= sorted(userScores, key = lambda x:x[1]+detectParam.baCoef*numpy.mean(getRMSEMap(x[3])), reverse=False)
    else:
        userScoresSorted= sorted(userScores, key = lambda x:x[1], reverse=False)

    userScores=userScoresSorted
    
    selSet=[]
    selCost=0
    selID=set()
    selScore=0.0
    
    similarCount=0
    noSmallScoreCount=1
    predErrCount=0
    
    
    
    candidateUser=None
    while len(selSet)<min(selectSize, left_k) and len(userScores)>0 and selCost<left_c:
        candidateUser=userScores.pop()
        if runFlags.extFlag:
            if candidateUser[1]<=0:
                break
        if candidateUser[0] in selID:
            continue
        if len(selSet)>10:
            if candidateUser[1] < 0.0001 * float(selScore)/float(len(selSet)):
                break
        if userSim(candidateUser, selSet, thres=detectParam.userSimThres, countThres=math.ceil(selectSize * detectParam.userSimCountCoef)):
            similarCount+=1
            # print 'user %s similar, unselect' %user[0]
            continue
        
        selSet.append(candidateUser)
        selCost+=userInfoDict[candidateUser[0]]['cost']
        selID.add(candidateUser[0])
        selScore+=candidateUser[1]
        
        eCount=0
        tCount=0
        for i, aut in enumerate(candidateUser[2]):
            if aut.detectCount>base_util[i].detectCount:
                eCount+=1
                if aut.detectThres<=aut.detectCount:
                    tCount+=1
                    
                    
        if runFlags.preFlag:
            print 'user %s selected, score %s, count %s, overcount %s, rmse %s' %(candidateUser[0], candidateUser[1], eCount, tCount, getRMSEMap(candidateUser[3]))
            sys.stdout.flush()
        else:
            print 'user %s selected, score %s, count %s, overcount %s' %(candidateUser[0], candidateUser[1], eCount, tCount)
            sys.stdout.flush()
        
        if len(selSet) >= selectSize:
            break
        
        newUserScores=[]
        for user in userScores:
            updated_gt, updated_score = updateScore(user[2], candidateUser[2], base_util, userInfoDict[user[0]]['cost'], runFlags, detectParam)
            if runFlags.preFlag:
                newUserScores.append((user[0], updated_score, updated_gt, user[3]))
            else:
                newUserScores.append((user[0], updated_score, updated_gt))

        if runFlags.preFlag:
            userScores= sorted(newUserScores, key = lambda x:x[1]+detectParam.baCoef*numpy.mean(getRMSEMap(x[3])), reverse=False)
        else:
            userScores= sorted(newUserScores, key = lambda x:x[1], reverse=False)
        base_util=copy.deepcopy(candidateUser[2])
            

            
    print 'this round selected %s users: consider selecting %s, bypass similar %s' %(len(selID), len(involveUsers)-len(blackUsers), similarCount)
    sys.stdout.flush()

    if not runFlags.docFlag:
        corpus_src_detect, userInv = db.getText(timePoint=running_time, historySlots= Nh * step, output_prefix='data_sel/%s_userstep%s-%s_%s_his%sh_evall' %(jobID, left_k, len(selID), helper.timestamp2str(running_time), Nh*step), userList=selID, event_id=set(), cacheRet=corpus_ret)
        if len(userInv)<1:
            return selID, None, None
        dictionary, corpus_detect,metas = ti.corpus_init(corpus_src_detect, dictionary, prefix='model_sel/%s_userstep%s-%s_%s_his%sh_evall' %(jobID, left_k, len(selID), helper.timestamp2str(running_time), Nh*step), verbose=False, updateDict=False)
        step_util_gt, user_score = detectUtilWithGT(corpus_detect, dictionary, util_gt, corpus_gt, tfidf_gt, index_gt, runFlags, selCost, detectParam, countOnlyOnceFlag=False)
    else:
        corpus_src_detect=[]
        for uID in selID:
            corpus_src_detect.extend(corpus_ret.get(uID,[]))
        if runFlags.extFlag:
            step_util_gt, user_score = detectUtilWithDoc(corpus_src_detect, util_gt, runFlags, selCost, detectParam, countOnlyOnceFlag=True)
        else:
            step_util_gt, user_score = detectUtilWithDoc(corpus_src_detect, util_gt, runFlags, selCost, detectParam, countOnlyOnceFlag=False)
        step_util_gt=candidateUser[2]
#
    
    if runFlags.preFlag:
        known_vecs_part_step, gt_vecs_part_step= pr.getKnownAndGTPart(util_gt=util_gt, running_time=running_time, selUsers=PParam.SelectedID.union(selID), perSlot=PParam.PperSlot, NSh=PParam.PNSh , NSd=PParam.PNSd, jobID='predict', inputCache=PParam.predictCache, basetime=basetime_all)
        template_part_step = pr.getTemplate(util_gt, running_time, perSlot=PParam.PperSlot, step_tp=PParam.step_tp, Nh_tp=PParam.Nh_tp, Nl_tp=PParam.Nl_tp, retCache=PParam.predictCache, userList=PParam.SelectedID.union(selID), basetime=PParam.template[2])
        rmses= pr.predictResult((PParam.template[0], template_part_step, PParam.template[2]), known_vecs_part_step, known_vecs_all, gt_vecs_part_step, gt_vecs_all, NSh=PParam.PNSh, NSd=PParam.PNSd, NShSim=PParam.PNShSim, Nstep=PParam.PNstep, topSimSize=PParam.templateTopSimSize, simThres=PParam.PsimTemplateThres, combineCount=PParam.combineCount)
        PredictionRMSES = rmses
        return selID, step_util_gt, user_score, blackUsers, PredictionRMSES
    else:
        return selID, step_util_gt, user_score, blackUsers


def selectUserHeu(c, k, userInvolveList, runFlags, dictionary, corpus_gt, tfidf_gt, index_gt, util_gt, running_time, step=1, Nh=0, Nl=0, jobID='tmp_user', eventlist=set(), selectSize=1, detectParam=None, predictParam=None):
    # print 'user before in init %d' %len(userInvolveList)
    userInfo = db.getUserInfoAll(userInvolveList)
    userInfoDict={}
    for user in userInfo:
        userInfoDict[user['uid']]=user
    #sort by fo desc, and cost asce
    userInfo=sorted(userInfo, key= lambda x: (x['followers_count'], -x['cost']), reverse=True)

    cur_c=0
    cur_k=0
    corpus_src_detect, userInv, corpus_ret = db.getText(timePoint=running_time, historySlots= Nh * step, timelastsSlots= Nl * step, output_prefix='data_sel/%s_heu_%s_his%sh_evall' %(jobID, helper.timestamp2str(running_time), Nh*step), userList=userInvolveList, event_id=eventlist, cacheFlag=True)
    # print 'user after in init=%d' %len(userInv)
    print 'corpus detect size=%d' %len(corpus_ret)
    # print 'onestep ', type(corpus_ret), len(corpus_ret)

    # for cor in corpus_ret:
    #     print type(cor),cor
    #
    selected_uid = []
    if runFlags.extFlag:
        updated_util_gt_ext=copy.deepcopy(util_gt)
    updated_util_gt=copy.deepcopy(util_gt)
    blackUsers=set()
    
    PParam=predictParam
 
    
    for i in range(0,k):
        left_c = c- cur_c
        left_k = k- cur_k
        
        if runFlags.extFlag:
        # updated_util_gt=copy.deepcopy(updated_util_gt_ext)
            for i, event in enumerate(updated_util_gt_ext):
                if updated_util_gt[i].detectCount >=1:
                    event.detectCount += 1#event.detectCount
                
                if event.detectCount>=event.detectThres:
                    updated_util_gt[i].detectThres=0
                    updated_util_gt[i].detectCount=0
                else:
                    updated_util_gt[i].detectThres=1
                    updated_util_gt[i].detectCount=0
        
        print 'selecting round for %d users, left_k=%s, left_c= %s' %(selectSize, left_k, left_c)
        if left_c <= 0 or left_k <= 0:
            print 'selected done!, cur_c=%s, cur_k=%s' %(cur_c, cur_k)
            return selected_uid
        else:
            userCand=set()
            bondCount=0
            for user in userInfo:
                if user['uid'] in blackUsers:
                    continue
                if user['uid'] in userInvolveList and user['uid'] not in selected_uid:
                    if user['avr_rp_count']>0:
                        if runFlags.bondFlag:
                            if user['cost'] > costBond(left_c, left_k, detectParam.costCoef) or user['cost']<=0:
                                bondCount+=1
                                continue
                        userCand.add(user['uid'])
            if len(userCand)>0:
                print 'selecting from all_user %d' %len(userCand)
                if runFlags.bondFlag:
                    print 'has bypassed %d bigger than bond' %bondCount
                
                if runFlags.preFlag:
                    PParam.SelectedID=set(copy.deepcopy(selected_uid))
                    selSet, updated_util_gt, user_score, blackUsersStep, stepPredictionRMSES = selectUserOneStep(left_c, left_k, userCand, runFlags, 
                        userInfoDict, corpus_ret, dictionary, corpus_gt, tfidf_gt, index_gt, updated_util_gt, running_time,
                        step=step, Nh=Nh, Nl=Nl, jobID=jobID, eventlist=eventlist, selectSize=selectSize, detectParam=detectParam, predictionParam=PParam)
                    if len(selSet)==0:
                        break
                    print 'selected user %s with score %f, RMSE %s' %(str(selSet), user_score, getRMSEMap(stepPredictionRMSES))
                else:
                    selSet, updated_util_gt, user_score, blackUsersStep = selectUserOneStep(left_c, left_k, userCand, runFlags, 
                    userInfoDict, corpus_ret, dictionary, corpus_gt, tfidf_gt, index_gt, updated_util_gt, running_time,
                    step=step, Nh=Nh, Nl=Nl, jobID=jobID, eventlist=eventlist, selectSize=selectSize, detectParam=detectParam, predictionParam=PParam)
                    if len(selSet)==0:
                        break
                    print 'selected user %s with score %f' %(str(selSet), user_score)
                for user in selSet:
                    selected_uid.append(user)
                    cur_c +=userInfoDict[user]['cost']
                cur_k += len(selSet)
                for user in blackUsersStep:
                    blackUsers.add(user)
            else:
                break

    print 'selected done!, cur_c=%d, cur_k=%s' %(cur_c, cur_k)
    return selected_uid


def combined_index_extract(corpus_filtered, metas_filtered, combine_index, dictionary):
    events_result=[]
    events_corpus=[]
    metas_result=[]
    from collections import Counter
    for event in combine_index:
        if event:
            event_words=Counter()
            meta_result=[]
            # test_set=[]
            event_words_count=0
            for doc_id in event:
                if doc_id:
                    for vec in corpus_filtered[doc_id]:
                        if vec[0] not in event_words:
                            event_words[vec[0]]=0
                        event_words[vec[0]]=event_words[vec[0]]+1
                        event_words_count += 1
                    meta_result.append(metas_filtered[doc_id])

            avg_size=int(event_words_count/len(event))
            event_result=[]
            
            for word in event_words.most_common(avg_size):
                event_result.append(dictionary[word[0]])
            events_result.append((len(event),event_result,meta_result))
            events_corpus.append(dictionary.doc2bow(list(event_result)))
            metas_result.append(meta_result)
            
    # events_result.sort(key=lambda x:x[0], reverse=True)
    print 'event combined into %d groups' %len(events_result)
    return events_result, events_corpus, metas_result

def second_combined_index_extract(events_result, events_corpus, metas_result, second_combine_index, dictionary):
    second_events_result=[]
    second_events_corpus=[]
    second_metas_result=[]
    from collections import Counter
    for event in second_combine_index:
        if event:
            event_words=Counter()
            event_meta=[]# meta_result=[]
            event_total_count=0
            # test_set=[]
            event_words_count=0
            for doc_id in event:
                if doc_id:
                    # print 'content',events_result[doc_id]
                    event_count, event_result, meta_result = events_result[doc_id]
                    event_result=events_corpus[doc_id]
                    for vec in event_result:
                        if vec[0] not in event_words:
                            event_words[vec[0]]=0
                        event_words[vec[0]]=event_words[vec[0]]+1
                        event_words_count += 1
                    event_total_count += event_count
                    event_meta.extend(meta_result)
                    # meta_result.extend(second_metas[i])#metas_filtered[doc_id])

            avg_size=int(event_words_count/len(event))
            event_result=[]
            
            for word in event_words.most_common(avg_size):
                event_result.append(dictionary[word[0]])
            second_events_result.append((event_total_count, event_result, event_meta))
            second_events_corpus.append(dictionary.doc2bow(list(event_result)))
            second_metas_result.append(event_meta)
            
    second_events_result.sort(key=lambda x:x[0], reverse=True)
    print 'event combined size %d' %len(events_result)
    return second_events_result, second_events_corpus, second_metas_result
    

def online_detect_v3(corpus_filtered, metas_filtered, combine_index, dictionary, corpus_gt_online, tfidf_gt_online, index_gt_online, util_gt_online, detectParam):
    updated_util_gt_online=copy.deepcopy(util_gt_online)
    
    combine_index=combine_doc_v2(corpus_filtered, dictionary, thres=detectParam.onlineCombineThres, prefix='model/first')
    events_result, events_corpus, metas_result=combined_index_extract(corpus_filtered, metas_filtered, combine_index, dictionary)
    
    #secondary cluster
    second_combine_index = combine_doc_v2(events_corpus, dictionary, thres=0.2, prefix='model/second')#=detectParam.onlineCombineThres, prefix='model/second')
    events_result, events_corpus, metas_result =second_combined_index_extract(events_result, events_corpus, metas_result, second_combine_index, dictionary)

    is_gt_metas={}
    result_gt_index={}
    count_less_thres_event_combine=0
    count_not_gt_event=0
    events_result_index=0
    for event_count, event_result, meta_result in events_result:
        # print 'combined eventsize=%d' %event_count
        # for word in event_result:
        #     print word,
        # print ''
        
        if event_count>=detectParam.precDetectedCountThres:
            query_vec= dictionary.doc2bow(list(event_result))
            sim = ti.query_tfidf(query_vec, dictionary, corpus_gt_online, tfidf_gt_online, index_gt_online, updateIndex=False, verbose=True)
            if not ti.isNewDoc(sim, verbose=False, thres=detectParam.isNewDocThres):
                gt_event_id = sim[0][0]
                updated_util_gt_online[gt_event_id].detectCount+=event_count
                if gt_event_id not in is_gt_metas:
                    is_gt_metas[gt_event_id]=[]
                is_gt_metas.get(gt_event_id).extend(meta_result)
                result_gt_index[events_result_index]=gt_event_id
            else:
                count_not_gt_event+=1
        else:
             count_less_thres_event_combine+=1
        events_result_index+=1
    
    print 'event combined result: match gt=%d, no match=%d, less than thres=%d' %(len(is_gt_metas), count_not_gt_event, count_less_thres_event_combine)
    return updated_util_gt_online, events_result, is_gt_metas, result_gt_index
    #begin recall/prec calc
    
    
def combine_doc_v2(corpus_to_combine, dictionary, thres=0.5, prefix='tmpon_new'):
    tfidf_online_combine, index_online_combine = ti.sims_init(corpus_to_combine, prefix=prefix)
    print 'tfidf done, %s combine online' %prefix
    sys.stdout.flush()
    
    clustered_corpus_index=set()
    combined_index=[]
    point = len(corpus_to_combine) / 20 + 1
    for current_index, query_vec in enumerate(corpus_to_combine):
        if (current_index % point) == 0:
            print ('%d%%'%(5*current_index/point)),
            sys.stdout.flush()
         
        if current_index not in clustered_corpus_index:
            same_event=set()
            same_event.add(current_index)
            
            sims = ti.query_tfidf(query_vec, dictionary, corpus_to_combine, tfidf_online_combine, index_online_combine, updateIndex=False, best=len(corpus_to_combine), verbose=False)

            for sim in list(sims):
                if not ti.isNewDoc(sim,thres=thres,verbose=False) and sim[0] not in clustered_corpus_index:
                    #print 'sim: ',sim[1], vec2text(corpus_vec_online[sim[0]])
                    same_event.add(sim[0])
                    clustered_corpus_index.add(sim[0])
            combined_index.append(same_event)
            # print 'GROUP size %d' %len(same_event)
#             for corpus_id in same_event:
#                 print 'SAME GROUP...',
#                 for word in corpus_to_combine[corpus_id]:
#                     print dictionary[word[0]],
#                 print ''
#             print ''
    print '%d corpuses combined into %d groups' %(len(corpus_to_combine), len(combined_index))
    # print combined_index
    sys.stdout.flush()
    return combined_index
    
def calcRec(util_gt, detectRecallThres=5):
    detected_events_count_recall=0
    total_detect_count=0
    for event in util_gt:
        total_detect_count+=event.detectCount
        if event.detectCount>= detectRecallThres:
            detected_events_count_recall+=1
    if len(util_gt)==0:
        recall=-1
        avg_doc=0
    else:
        recall= float(detected_events_count_recall)/float(len(util_gt))
        avg_doc=float(total_detect_count)/float(len(util_gt))
    print '====== recall thres= %s ======' %detectRecallThres
    print 'recall %s, %d / %d' %(recall, detected_events_count_recall, len(util_gt))
    print 'avg doc %f' %avg_doc
    return recall

def fscore(p, r, beta=1):
    return float((1+beta**2)*p*r)/float(beta**2*p+r)


def calcPR(util_gt, new_events, detect_metas=None, result_gt_index=None, detectRecallThres=5, detectPrecThres=5, cut=-1):
    detected_events_count_recall=0
    detected_events_count_recall_new=0
    # _count_prec=0
    total_time_gain=0
    total_events_count=0
    
    
    gt_set=set()
    event_index=-1
    time_gain_dict={}
    for event_count, event_result, meta_result in new_events:
        event_index+=1
        if event_count< detectRecallThres:
            continue
        if event_index in result_gt_index:
            gt_event_id=result_gt_index.get(event_index)
            if gt_event_id not in gt_set:
                gt_set.add(gt_event_id)
                detected_events_count_recall_new+=1
            
            times_meta=sorted(meta_result)
            time_gain = util_gt[gt_event_id].topicReportTime-times_meta[detectRecallThres-1]
            if gt_event_id not in time_gain_dict:
                time_gain_dict[gt_event_id]=time_gain
            else:
                if time_gain > time_gain_dict[gt_event_id]:
                    time_gain_dict[gt_event_id]=time_gain
            
            total_events_count += event_count
        
    
    for evid in time_gain_dict.keys():
        total_time_gain += time_gain_dict[evid]
    if len(util_gt)>0:
        avg_time_gain=total_time_gain/len(util_gt)
    else:
        avg_time_gain=0
        
    
    
    detected_new_events_count=0
    detected_events_in_gt=0
    event_index=-1
    gt_set=set()
    if cut<0:
        cut=len(new_events)
    for event_count, event_result, meta_result in new_events:
        event_index+=1
        if event_count< detectPrecThres:
            continue
        if event_index>=cut:
            continue
        if event_index in result_gt_index:
##          if result_gt_index.get(event_index) not in gt_set:
##              gt_set.add(result_gt_index[event_index])
            detected_events_in_gt+=1
        
        detected_new_events_count+=1
    
    if len(util_gt)==0 or detected_new_events_count==0:
        recall=-1
        precision=-1
        avg_doc=0
    else:
        recall=float(detected_events_count_recall_new)/float(len(util_gt))
        precision=float(detected_events_in_gt)/float(detected_new_events_count)
        avg_doc=float(total_events_count)/float(len(util_gt))
    
    print '====== recall thres=%s, prec thres=%s, cut=@%s ======' %(detectRecallThres, detectPrecThres, cut)
    print 'recall %s, %d / %d' %(recall, detected_events_count_recall_new, len(util_gt))
    print 'precis %s, %d / %d' %(precision, detected_events_in_gt, detected_new_events_count)
    print 'f1-socre %s' %(fscore(p=precision, r=recall, beta=1))
    print 'f2-socre %s' %(fscore(p=precision, r=recall, beta=2))
    print 'avg time gain %d' %avg_time_gain
    print 'avg doc %f' %avg_doc
    # for cut in [10,20,30,50,100,150,200]:
    #     calcPrecCut(util_gt, new_events, detectPrecThres, cut, recall)
    return recall, precision, avg_time_gain, avg_doc, cut
    

