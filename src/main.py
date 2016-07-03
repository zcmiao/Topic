#!/usr/bin/env python
#encoding: utf8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os.path
import os
import libWordCut as wc
import libTFIDF as ti
import libSQL as db
import libDetect as de
import libPredict as pr
from gensim import corpora, models, similarities
import pickle
import argparse
import numpy
import libHelper as helper
import time


if __name__ == '__main__':
    program_start_time_init = time.time()
    parser = argparse.ArgumentParser(description="select heu")
    parser.add_argument('-r', '--range',
                        dest    = 'rg',
                        default = '1',
                        nargs   = '?',
                        type    = int,
                        help    = "run times total")
    parser.add_argument('-step', '--step',
                         dest    = 'step',
                         default = '24',
                         nargs   = '?',
                         type    = int,
                         help    = "run update frequency in hour")
    parser.add_argument('-t', '--type',
                        dest    = 'selType',
                        default = 'cevx',
                        nargs   = '?',
                        help    = "cevx: swc / embd: jnt / embdp: jnt p")
    parser.add_argument('-i', '--ID',
                         dest    = 'jobID',
                         default = 'test',
                         nargs   = '?',
                         help    = "jobID prefix")
    parser.add_argument('-Nh', '--Nh',
                         dest    = 'Nh',
                         default = '15',
                         nargs   = '?',
                         type    = int,
                         help    = "training historical in days")
    parser.add_argument('-Nd', '--Nd',
                         dest    = 'Nd',
                         default = '15',
                         nargs   = '?',
                         type    = int,
                         help    = "online test in days")
    parser.add_argument('-Nl', '--Nl',
                         dest    = 'Nl',
                         default = '7',
                         nargs   = '?',
                         type    = int,
                         help    = "topic lifetime lasts in days")
    parser.add_argument('-k', '--userCount',
                         dest    = 'k',
                         default = '500',
                         nargs   = '?',
                         type    = int,
                         help    = "selected users size")
    parser.add_argument('-c', '--cost',
                         dest    = 'c',
                         default = '30000',
                         nargs   = '?',
                         type    = int,
                         help    = "selected total cost")
    parser.add_argument('-ba', '--baCoef',
                         dest    = 'baCoef',
                         default = '1',
                         nargs   = '?',
                         type    = float,
                         help    = "balance coeff")
    parser.add_argument('-X', '--DocThres',
                         dest    = 'DocThres',
                         default = '40',
                         nargs   = '?',
                         type    = int,
                         help    = "Doc Threshold for each topic")
    parser.add_argument('-userSim', '--userSimThres',
                         dest    = 'userSimThres',
                         default = '0.999',
                         nargs   = '?',
                         type    = float,
                         help    = "user similar check in each selectsize")
    parser.add_argument('-costCoef', '--costBoundaryCoef',
                         dest    = 'costBoundaryCoef',
                         default = '0.7',
                         nargs   = '?',
                         type    = float,
                         help    = "cost Boundary Coef")
    parser.add_argument('-ergCoef', '--ergCoef',
                         dest    = 'ergCoef',
                         default = '0.01',
                         nargs   = '?',
                         type    = float,
                         help    = "erg Coef")
    parser.add_argument('-size', '--selSize',
                         dest    = 'selSize',
                         default = '1',
                         nargs   = '?',
                         type    = int,
                         help    = "user select count for each step")
    parser.add_argument('-l', '--log',
                         dest    = 'logFlag',
                         action  = 'store_true',
                         help    = "save to log file")
    parser.add_argument('-offtest', '--offtest',
                         dest    = 'offtestFlag',
                         action  = 'store_true',
                         help    = "evaluate offline")
    parser.add_argument('-ontest', '--ontest',
                         dest    = 'ontestFlag',
                         action  = 'store_true',
                         help    = "evaluate online")
    parser.add_argument('-predict', '--predict',
                         dest    = 'predictFlag',
                         action  = 'store_true',
                         help    = "evaluate prediction")
    parser.add_argument('-select', '--select',
                         dest    = 'selectFlag',
                         action  = 'store_true',
                         help    = "user selection")
    parser.add_argument('-baseline', '--baselineselect',
                         dest    = 'baselineFlag',
                         action  = 'store_true',
                         help    = "user selection baseline")
    parser.add_argument('-pagerank', '--pagerank',
                         dest    = 'selectPageRankFlag',
                         action  = 'store_true',
                         help    = "user selection pagerank")
    parser.add_argument('-PNSh', '--PNSh',
                         dest    = 'PNSh',
                         default = '60',
                         nargs   = '?',
                         type    = int,
                         help    = "known length")
    parser.add_argument('-PNSd', '--PNSd',
                         dest    = 'PNSd',
                         default = '60',
                         nargs   = '?',
                         type    = int,
                         help    = "to predict length in selection")
    parser.add_argument('-PNSdLong', '--PNSdLong',
                         dest    = 'PNSdLong',
                         default = '240',
                         nargs   = '?',
                         type    = int,
                         help    = "to predict length in final prediction")
    parser.add_argument('-PperSlot', '--PperSlot',
                         dest    = 'PperSlot',
                         default = '0.1',
                         nargs   = '?',
                         type    = float,
                         help    = "per slot size, unit is 1 hour")
    parser.add_argument('-PNstep', '--PNstep',
                         dest    = 'PNstep',
                         default = '10',
                         nargs   = '?',
                         type    = int,
                         help    = "per step in each prediction in selection")
    parser.add_argument('-PNstepLong', '--PNstepLong',
                         dest    = 'PNstepLong',
                         default = '30',
                         nargs   = '?',
                         type    = int,
                         help    = "per step in each prediction in final prediction")
    parser.add_argument('-PNShSim', '--PNShSim',
                         dest    = 'PNShSim',
                         default = '30',
                         nargs   = '?',
                         type    = int,
                         help    = "similarity calc length")
    parser.add_argument('-PsimThres', '--PsimThres',
                         dest    = 'PsimThres',
                         default = '0.5',
                         nargs   = '?',
                         type    = float,
                         help    = "similarity threshold")
    parser.add_argument('-PsimCount', '--PsimCount',
                         dest    = 'PsimCount',
                         default = '5',
                         nargs   = '?',
                         type    = int,
                         help    = "similarity top count")
    parser.add_argument('-combineCount', '--combineCount',
                         dest    = 'combineCount',
                         default = '10',
                         nargs   = '?',
                         type    = int,
                         help    = "combine RMSE count")
    
    for directory in ['result', 'model', 'data', 'model_sel', 'data_sel', 'detect']:
        if not os.path.exists(directory):
            os.makedirs(directory)

    args = parser.parse_args()
    
    if args.ontestFlag:
        args.offtestFlag=True
    
    if args.logFlag:
        sys.stdout = open('%s_%s_c%s_k%s_s%s_size%s.log' %(args.jobID, args.selType, args.c, args.k, args.step, args.selSize), 'w')
    
    for i in range(0,args.rg):
        
        # 2012-09-25 1348502400
        init_time_string='2012-09-25-00'
        init_time = helper.str2timestamp(init_time_string)
        step = args.step
        running_time=init_time + 3600 * step * i
        jobID='%s_%s' %(args.jobID, args.selType)
        
        #training start, in day
        Nh = args.Nh
        #test start, in day
        Nd = args.Nd
        #training lasts, in day
        Nl = args.Nl
        
        
        #patch offset, in day
        gt_Nh_patch=0
        gt_Nd_patch=0
        
        #ground truth for offline test
        gt_Nh = Nh+gt_Nh_patch
        gt_Nd = gt_Nd_patch
        
        #ground truth for online teset
        gt_on_Nh = gt_Nh_patch
        gt_on_Nd = Nd+gt_Nd_patch


        offline_userSimThres = args.userSimThres
        
        offline_sim_thres=0.3
        offline_detectThres=args.DocThres
        
        online_filter_thres=0.3
        online_combine_thres=0.25
        online_sim_thres=0.25
        recallDetectedCountThres=3
        precDetectedCountThres=3
        
        ergCoef=args.ergCoef
        costCoef=args.costBoundaryCoef
        baCoef=args.baCoef
        lambda_s=-100
        baCoef=float(baCoef)/float(lambda_s)
        
        PNSh=args.PNSh
        PNSd=args.PNSd
        PNSdLong=args.PNSdLong
        PperSlot=args.PperSlot
        PsimTemplateThres=args.PsimThres
        PtemplateTopSimSize=args.PsimCount
        PNh_tp=gt_Nh
        PNl_tp=Nl
        
        PNShSim=args.PNShSim
        PNstep=args.PNstep
        PNstepLong=args.PNstepLong
        combineCount=args.combineCount
        
        wc.init_lib('../python-nlpir')
        dictionary= corpora.Dictionary()
        runFlags= de.TypeFlag(args.selType)
        
        
        DParam=de.detectParam(isNewDocThres=offline_sim_thres, detectThres=offline_detectThres, userSimThres= offline_userSimThres, ergCoef=ergCoef, costCoef=costCoef, baCoef= baCoef)
        
        if args.offtestFlag:
            offlineTestParam=de.detectParam(isNewDocThres=offline_sim_thres, recallDetectedCountThres=recallDetectedCountThres)
        
        if args.ontestFlag:
            onlineTestParam=de.detectParamOnline(isNewDocThres=online_sim_thres, onlineFilterExistThres=online_filter_thres, onlineCombineThres=online_combine_thres, recallDetectedCountThres=recallDetectedCountThres,precDetectedCountThres=precDetectedCountThres)

        PParam=pr.predictionParam(PNSh=PNSh, PNSd=PNSd, PNSdLong=PNSdLong, PperSlot=PperSlot, PsimTemplateThres=PsimTemplateThres, templateTopSimSize=PtemplateTopSimSize, Nh_tp=PNh_tp, Nl_tp=PNl_tp, PNShSim=PNShSim, PNstep=PNstep, PNstepLong=PNstepLong, combineCount=combineCount)

        
        print '====== init %s %s ======' %(jobID, helper.timestamp2str(running_time))

        dictionary, corpus_gt, tfidf_gt, index_gt, util_gt = ti.loadEventGT(dictionary, timePoint=running_time, step=step, gt_Nh=gt_Nh, gt_Nd=gt_Nd, jobID=jobID, inputEvent= set(), detectParam= DParam)
        eventlist_gt=set()
        for event in util_gt:
            eventlist_gt.add(event.db_event_id)
        corpus_src_init, userInvolveList_init, corpus_init_ret = db.getText(timePoint=running_time, historySlots= Nh * step, timelastsSlots = Nl * step, output_prefix='data/%s_init_%s_%sh%sh%sh_ev%s' %(jobID ,helper.timestamp2str(running_time), Nh*step, Nd*step, Nl*step, len(eventlist_gt)), event_id=eventlist_gt, util_gt=util_gt, cacheFlag=True)
        template_vecs_all, basetime_all_init = pr.getTemplate(util_gt, running_time, perSlot=PParam.PperSlot, step_tp=PParam.step_tp, Nh_tp=PParam.Nh_tp, Nl_tp=PParam.Nl_tp, retCache=corpus_init_ret)
        PParam.template=(template_vecs_all, None, basetime_all_init)
        delta_time=time.time() - program_start_time_init
        print("--- init using %s seconds, or %s minutes ---" % (delta_time, delta_time/60.0))

        program_start_time_select=time.time()
        # alg. swc jnt
        if args.selectFlag:
            print '====== selecting heu %s %s ====== ' %(jobID, helper.timestamp2str(running_time))
            if not os.path.exists('%s-Nh%s-lasts%s.dict' %(helper.timestamp2str(running_time),Nh,Nl)):
                dictionary, corpus_init,metas_init = ti.corpus_init(corpus_src_init, dictionary, prefix='model/%s_init_%s_his%sh_lasts%sh_evl%s' %(jobID, helper.timestamp2str(running_time), Nh * step, Nl * step, len(eventlist_gt)), filtDict=False)
                del corpus_init
                dictionary.save('%s-Nh%s-lasts%s.dict' %(helper.timestamp2str(running_time),Nh, Nl))
            else:
                dictionary=dictionary.load('%s-Nh%s-lasts%s.dict' %(helper.timestamp2str(running_time),Nh,Nl))
                print 'load dictionary size: %s' %len(dictionary)
            
            dictionary, corpus_gt, tfidf_gt, index_gt, util_gt = ti.loadEventGT(dictionary, timePoint=running_time, step=step, gt_Nh=gt_Nh, gt_Nd=gt_Nd, jobID=jobID, inputEvent= eventlist_gt, detectParam= DParam, verbose=True)

            selected= de.selectUserHeu(c=args.c, k=args.k, userInvolveList=userInvolveList_init, runFlags=runFlags, dictionary=dictionary, corpus_gt=corpus_gt, tfidf_gt=tfidf_gt, index_gt=index_gt, util_gt=util_gt, running_time=running_time, step=step, Nh=Nh, Nl=Nl, jobID=jobID, eventlist=eventlist_gt, selectSize=args.selSize, detectParam=DParam, predictParam=PParam)
            db.markUser(selected, method='%s_%s_c%s_k%s_s%s_size%s' %(args.jobID, args.selType, int(args.c), args.k, args.step, args.selSize), overwrite=True)

        #alg. fm ecm
        elif args.baselineFlag:
            print '====== selecting baseline %s %s ====== ' %(jobID, helper.timestamp2str(running_time))
            selected= db.selectUserbaseline(c=args.c, k=args.k, method=args.selType, userInvolveList=userInvolveList_init)
            db.markUser(selected, method='%s_%s_c%s_k%s_s%s_size%s' %(args.jobID, args.selType, int(args.c), args.k, args.step, args.selSize), overwrite=True)
        
        #alg. pr
        elif args.selectPageRankFlag:
            print '====== selecting pagerank %s %s ====== ' %(jobID, helper.timestamp2str(running_time))
            args.selType='pagerank'
            selected= db.selectUserPageRank(c=args.c, k=args.k, method=args.selType, userInvolveList=userInvolveList_init, corpus= corpus_init_ret)
            db.markUser(selected, method='%s_%s_c%s_k%s_s%s_size%s' %(args.jobID, args.selType, int(args.c), args.k, args.step, args.selSize), overwrite=True)
        
        userList=db.getUserList(method='%s_%s_c%s_k%s_s%s_size%s' %(args.jobID, args.selType, int(args.c), args.k, args.step, args.selSize))
        print '====== has read %d users ======' %len(userList)
        delta_time=time.time() - program_start_time_select
        print("--- selecting using %s seconds, or %s minutes ---" % (delta_time, delta_time/60.0))
        
        #offline dataset evaluation
        program_start_time_off=time.time()
        corpus_src_offline, userInvolveList_offline, corpus_ret_offline= db.getText(timePoint=running_time, historySlots= Nh * step, output_prefix='data/%s_off_%s_his%sh_evall' %(jobID, helper.timestamp2str(running_time), Nh*step), userList=userList, event_id=set(), cacheFlag=True)
        template_vecs_part = pr.getTemplate(util_gt, running_time, perSlot=PParam.PperSlot, step_tp=PParam.step_tp, Nh_tp=PParam.Nh_tp, Nl_tp=PParam.Nl_tp, retCache=corpus_ret_offline, userList=userList, basetime=basetime_all_init)
        PParam.template=(template_vecs_all, template_vecs_part, basetime_all_init)
        print '====== has read %d template  =====' %len(template_vecs_all)
        
        if args.offtestFlag:
            print '====== compare offline using direct-gt (for debug only) %s %s ====== ' %(jobID, helper.timestamp2str(running_time))
            corpus_src_offline, userInvolveList_offline, corpus_offline_ret_full = db.getText(timePoint=running_time, historySlots= Nh * step, output_prefix='data/%s_off_%s_his%sh_evall' %(jobID, helper.timestamp2str(running_time), Nh*step), userList=userList, event_id=set(), cacheFlag=True)
            
            updated_util_gt_offline_gt=de.countUtilGT(util_gt, corpus_offline_ret_full, userList)
            for i in [-2,-1,1,2,5,0,7,12,17,27,offline_detectThres-6, offline_detectThres-5, offline_detectThres-4, offline_detectThres-3, offline_detectThres-2, offline_detectThres-1]:
                offline_recall=de.calcRec(updated_util_gt_offline_gt, offlineTestParam.recallDetectedCountThres+i)
            
            print '====== compare offline using content matching %s %s ====== ' %(jobID, helper.timestamp2str(running_time))
            corpus_src_offline, userInvolveList_offline = db.getText(timePoint=running_time, historySlots= Nh * step, output_prefix='data/%s_off_%s_his%sh_evall' %(jobID, helper.timestamp2str(running_time), Nh*step), userList=userList, event_id=set())

            if corpus_src_offline!=None:
                dictionary, corpus_offline, metas_offline = ti.corpus_init(corpus_src_offline, dictionary, prefix='model/%s_off_%s_his%sh_evall' %(jobID, helper.timestamp2str(running_time), Nh * step), filtDict=False)
                tfidf_offline, index_offline = ti.sims_init(corpus_offline, prefix='model/%s_off_%s_his%sh_evall' %(jobID, helper.timestamp2str(running_time), Nh * step))
                print 'offline user_size=%d corpus_size=%d' %(len(userInvolveList_offline), len(corpus_offline))
                updated_util_gt_offline, delta_score = de.detectUtilWithGT(corpus_offline, dictionary, util_gt, corpus_gt, tfidf_gt, index_gt, runFlags, user_cost=1, detectParam=offlineTestParam, countOnlyOnceFlag=False)
                for i in [-2,-1,1,2,5,0,7,12,17,27,offline_detectThres-6, offline_detectThres-5, offline_detectThres-4, offline_detectThres-3, offline_detectThres-2, offline_detectThres-1]:
                    offline_recall=de.calcRec(updated_util_gt_offline, offlineTestParam.recallDetectedCountThres+i)
            else:
                print 'selected user dont have offline'
            
            # print '====== compare offline count======'
            # for i, util in enumerate(updated_util_gt_offline):
            #     print util.db_event_id, util.detectCount, updated_util_gt_offline_gt[i].detectCount
            #
            
            known_vecs_all_offline, gt_vecs_all_offline, basetime_all_offline =  pr.getKnownAndGTAll(corpus_init_ret, util_gt, perSlot=PParam.PperSlot, NSh= PParam.PNSh, NSd=PParam.PNSdLong)
            known_vecs_part_offline, gt_vecs_part_offline = pr.getKnownAndGTPart(util_gt, running_time, selUsers=userList, perSlot=PParam.PperSlot, NSh= PParam.PNSh, NSd=PParam.PNSdLong, jobID='predict', inputCache=corpus_init_ret, basetime=basetime_all_offline)
            rmses= pr.predictResult(PParam.template, known_vecs_part_offline, known_vecs_all_offline, gt_vecs_part_offline, gt_vecs_all_offline,  NSh=PParam.PNSh, NSd=PParam.PNSdLong, NShSim=PParam.PNShSim, Nstep=PParam.PNstepLong, topSimSize=PParam.templateTopSimSize, simThres=PParam.PsimTemplateThres, combineCount=PParam.combineCount)
            for j in range(0, len(rmses)):
                if j<4:
                    print 'predict', de.getRMSEMap(rmses, index=j)
                else:
                    print de.getRMSEMap(rmses, index=j)
        delta_time=time.time() - program_start_time_off
        print("--- offline using %s seconds, or %s minutes ---" % (delta_time, delta_time/60.0))

        #online testing detect
        if args.ontestFlag:
            print '====== compare online %s %s ====== ' %(jobID, helper.timestamp2str(running_time))
            corpus_src_online_full, userInvolveList_online_full, corpus_online_ret_full = db.getText(timePoint=running_time, detectionSlots= Nd * step, output_prefix='data/%s_on_%s_det%sh_evall' %(jobID, helper.timestamp2str(running_time), Nd*step), userList=set(), event_id=set(), cacheFlag=True)
            corpus_online_events=set()
            if corpus_online_ret_full!=None and len(corpus_online_ret_full)>0 :
                for line in corpus_online_ret_full:
                    corpus_online_events.add(line['event_id'])
            for event in util_gt:
                if event.db_event_id in corpus_online_events:
                    print 'overlap in gt off and on', event.db_event_id
                    corpus_online_events.remove(event.db_event_id)
                    
        program_start_time_on=time.time()
        if args.ontestFlag:
            dictionary, corpus_gt_online, tfidf_gt_online, index_gt_online, util_gt_online = ti.loadEventGT(dictionary, timePoint=running_time, step=step, gt_Nh=gt_on_Nh, gt_Nd=gt_on_Nd, jobID=jobID, inputEvent= corpus_online_events)
            print '====== compare online with direct-gt (for debug only) ====='
            corpus_src_online, userInvolveList_online, corpus_online_ret_user= db.getText(timePoint=running_time, detectionSlots= Nd * step, output_prefix='data/%s_on_%s_det%sh_evall' %(jobID, helper.timestamp2str(running_time), Nd*step), userList=userList, event_id=set(), cacheFlag=True)
            updated_util_gt_online_gt=de.countUtilGT(util_gt_online, corpus_online_ret_user, userList)
            for i in [-2,-1,1,2,5,7]:
                offline_recall=de.calcRec(updated_util_gt_online_gt, onlineTestParam.recallDetectedCountThres+i)

            print '====== compare online with content matching ====='
            corpus_src_online, userInvolveList_online= db.getText(timePoint=running_time, detectionSlots= Nd * step, output_prefix='data/%s_on_%s_det%sh_evall' %(jobID, helper.timestamp2str(running_time), Nd*step), userList=userList, event_id=set())
            corpus_filtered,metas_filtered = ti.filter_known_topic(corpus_src_online, dictionary, corpus_gt, tfidf_gt, index_gt, prefix='model/%s_on_%s_det%sh_evall' %(jobID, helper.timestamp2str(running_time),  Nd*step), filterThres= onlineTestParam.onlineFilterExistThres)
            combine_index=de.combine_doc_v2(corpus_filtered, dictionary, thres=onlineTestParam.onlineCombineThres, prefix='model/%s_on_%s_det%sh_evall')
            updated_util_gt_online,events_result,metas_result,result_gt_index = de.online_detect_v3(corpus_filtered, metas_filtered, combine_index, dictionary, corpus_gt_online, tfidf_gt_online, index_gt_online, util_gt_online, detectParam=onlineTestParam)
            print 'selected usersize=%d, online usersize=%d' %(len(userList), len(userInvolveList_online))

            # print '====== compare online count gt and content======'
            # for i, util in enumerate(updated_util_gt_online):
            #     print util.db_event_id, util.detectCount, updated_util_gt_online_gt[i].detectCount
            
            print '====== compare online result======='
            for i in  [-2,-1,1,2,5,7,0]: #[-2,-1,1,2,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,0]:
                for cut in [-1, 30, 50, 100]:
                    online_recall, online_precision, times_gain, avg_doc, cut= de.calcPR(updated_util_gt_online, events_result, metas_result, result_gt_index, detectRecallThres=onlineTestParam.recallDetectedCountThres+i, detectPrecThres=onlineTestParam.precDetectedCountThres+i, cut=cut)
            
            else:
                print 'selected user dont have online'
                
            
        delta_time=time.time() - program_start_time_on
        print("--- online using %s seconds, or %s minutes ---" % (delta_time, delta_time/60.0))
        
        #online predict
        program_start_time_predict=time.time()
        if args.predictFlag:
            print '====== compare prediction %s %s ====== ' %(jobID, helper.timestamp2str(running_time))

            PuserList=userList
            corpus_src_online, userInvolveList_online, corpus_online_ret= db.getText(timePoint=running_time, detectionSlots= Nd * step, output_prefix='data/%s_on_%s_det%sh_evall' %(jobID, helper.timestamp2str(running_time), Nd*step), userList=userList, event_id=set(), cacheFlag=True)

            known_vecs_all_online, gt_vecs_all_online, basetime_all_online =  pr.getKnownAndGTAll(corpus_online_ret_full, util_gt_online, perSlot=PParam.PperSlot, NSh= PParam.PNSh, NSd=PParam.PNSdLong)
            known_vecs_part_online, gt_vecs_part_online = pr.getKnownAndGTPart(util_gt_online, running_time, selUsers=PuserList, perSlot=PParam.PperSlot, NSh= PParam.PNSh, NSd=PParam.PNSdLong, jobID='predict', inputCache=corpus_online_ret, basetime=basetime_all_online)
            rmses= pr.predictResult(PParam.template, known_vecs_part_online, known_vecs_all_online, gt_vecs_part_online, gt_vecs_all_online,  NSh=PParam.PNSh, NSd=PParam.PNSdLong, NShSim=PParam.PNShSim, Nstep=PParam.PNstepLong, topSimSize=PParam.templateTopSimSize, simThres=PParam.PsimTemplateThres, combineCount=PParam.combineCount)
            for j in range(0, len(rmses)):
                if j<4:
                    print 'predict', de.getRMSEMap(rmses, index=j)
                # else:
                    # print de.getRMSEMap(rmses, index=j)
            print 'predicted rmse per slot', numpy.mean(de.getRMSEMap(rmses))/10 #10 slots per hour
        delta_time=time.time() - program_start_time_predict
        print("--- predict using %s seconds, or %s minutes ---" % (delta_time, delta_time/60.0))

        