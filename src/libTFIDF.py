#!/usr/bin/env python
#encoding: utf8

import libWordCut as wc
import libSQL as db
from gensim import corpora, models, similarities
import sys
import pickle
import copy
reload(sys)
sys.setdefaultencoding('utf-8')
import libHelper as helper

class gt_util_param:
    def __init__(self, db_event_id, detectThres, topicReportTime, topicStartTime=0):
        # [detectCount, previousErgency, db_event_id]
        self.detectCount = 0
        self.previousErgency = 1.0
        self.db_event_id = db_event_id
        self.detectThres = detectThres
        self.topicReportTime = topicReportTime
        self.topicStartTime = topicStartTime
        
    
def corpus_init(srcfile, dictionary, prefix='tmp_', seg=True, verbose=True, outFlag=False, filtDict=False, updateDict=True):
    if seg:
        texts, metas = fileSeg(srcfile, countLine=verbose)
        if verbose:
            print '%s wordcut done' %prefix
            sys.stdout.flush()
    else:
        texts=srcfile
        metas=None

    if updateDict:
        dictionary.add_documents(texts)
        print 'dict size:%d' %len(dictionary)
        sys.stdout.flush()
        
    if filtDict:
        print 'raw dict size:%d' %len(dictionary)
        dictionary=filter_dict(dictionary)
        print 'filtered dict size:%d' %len(dictionary)
        sys.stdout.flush()
    # dictionary=make_dict(texts,dictionary, filtDict=filtDict)

    if outFlag:
        dictionary.save('%s.dict' %prefix)
    if verbose:
        print '%s dict done, len %d' %(prefix, len(dictionary))
    # print metas, seg
    corpus,metas = make_corpus_vec(texts,dictionary,metas,metasFlag=seg)
    if outFlag:
        corpora.MmCorpus.serialize('%s.mm'%prefix, corpus)
    if verbose:
        print '%s corpus done' %prefix
        sys.stdout.flush()
    # print metas
    return dictionary, corpus, metas

def sims_init(corpus, prefix='tmp_', verbose=False):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf= tfidf[corpus]
    tfidf.save('%s.model'%prefix)
    if verbose:
        print '%s tfidf model done' %prefix
    
    index = similarities.Similarity(output_prefix='%s.chunk'%prefix, corpus=corpus_tfidf, num_features=len(corpora.Dictionary.from_corpus(corpus)))
    #index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))
    index.save('%s.index'%prefix)
    if verbose:
        print '%s index done' %prefix
    
    return tfidf, index

def fileSeg(filename, countLine=True):
    wordseg = wc.file2seglist(filename, tag = True, countLine = countLine)
    # wordsegfile=open('wordseg','w')
    # pickle.dump(wordseg, wordsegfile)
    # wordsegfile.close()
    texts, metas = wc.seglist4filter(wordseg, srcTag = True, filterLow=False, fromFile=True)
    # textfile=open('textfile','w')
    # pickle.dump(texts, textfile)
    # textfile.close()
    return texts, metas

def paraSeg(query):
    query = wc.para2seglist(query, tag=True)
    query = wc.seglist4filter(query, srcTag = True, filterLow = False, fromFile=True)
    return query

def make_dict(texts, dictionary, filtDict=False):
    dictionary.add_documents(texts)
    print 'dict size:%d' %len(dictionary)
    if filtDict:
        dictionary=filter_dict(dictionary)
        print 'filtered dict size:%d' %len(dictionary)
    return dictionary

def filter_dict(dictionary, lowFreqThres=1):
    lowFreqID=[]
    for ele in dictionary.iteritems():
        if dictionary.dfs.get(ele[0],0)<=lowFreqThres:# or ele[1].startswith('@'):
            lowFreqID.append(ele[0])
    dictionary.filter_tokens(bad_ids=lowFreqID)
    return dictionary


def make_corpus_vec(texts, dictionary, metas, metasFlag=True):
#    corpus = [dictionary.doc2bow(text) for text in texts if dictionary.doc2bow(text)]
    corpus=[]
    # metasFlag=False
    # if not metas:
        # matasFlag=True
    outmetas=[]
    for i, text in enumerate(texts):
        if dictionary.doc2bow(text):
            corpus.append(dictionary.doc2bow(text))
            # if metasFlag:
            if metasFlag:
                # print metas[i]
                outmetas.append(metas[i])
    ##print corpus with dict
    # count = 0
    # for doc in corpus:
    #     print count,
    #     for bow in doc:
    #         print dictionary[bow[0]],
    #     print ''
    #     count += 1
    if metasFlag:
        return corpus, outmetas
    else:
        return corpus, None


def query_tfidf(query_vec, dictionary, corpus, tfidf, index, updateIndex=False, best=1, verbose=False):
    index.num_best = best
    sims = index[tfidf[query_vec]]
    #print index
    
    if updateIndex:
        index.add_documents([tfidf[query_vec]])
        corpus.append(query_vec)
    
    
    if verbose and len(sims)>0:
        if sims[0][1]>=0.25:
            count=0
            print 'input query:',
            for word in query_vec:
                if word[0] in dictionary:
                    print dictionary[word[0]],
                else:
                    print 'NOT_IN_DICT',
            print ''
            # print ''
            for sim in sims:
                if count<2:
                    print sim,
                    for word in corpus[sim[0]]:
                        if word[0] in dictionary:
                            print dictionary[word[0]],
                        else:
                            print 'NOT_IN_DICT',
                    print ''
                count+=1
    
    return sims


def isNewDoc(sim, thres=0.8, verbose=True):
    if not sim:
        sim=(0,0)
    if isinstance(sim, list):
        sim=sim[0]
    thres_not_new=thres
    
    if sim[1] < thres_not_new:
        if verbose:
            print 'new, not in offline %s \n'%sim[1]
        return True
    if verbose:
        print 'not new, in offline'
    return False


def vec2text(query_vec):
    return ' '.join([dictionary[int(word[0])] for word in query_vec])
 
    

    
def filter_known_topic(corpus_src_to_filter, dictionary, corpus_offline, tfidf_offline, index_offline, prefix='tmpfil_', filterThres=0.5):
    dictioanry, corpus_to_filter, metas = corpus_init(corpus_src_to_filter, dictionary, prefix, filtDict=False)
    print 'before filter corpus size= %d' %len(corpus_to_filter)
    filtered_doc_id=[]
    filtered_corpus=[]
    filtered_metas=[]

    #first, compare to_filter with known offline
    print '%s filter known' %prefix
    # current_doc_id=0
    point = len(corpus_to_filter) / 20 + 1
    for i, query_vec in enumerate(corpus_to_filter):
        if (i % point) == 0:
            print ('%d%%'%(5*i/point)),
            sys.stdout.flush()
            
        if query_vec:
            sim = query_tfidf(query_vec, dictionary, corpus_offline, tfidf_offline, index_offline, updateIndex=False, verbose=False)
            if isNewDoc(sim, verbose=False):
                # filtered_doc_id.append(current_doc_id)
                filtered_corpus.append(query_vec)
                filtered_metas.append(metas[i])
            else:
                for word in query_vec:
                    print dictionary[word[0]],
                print 'SIMILIAR to', sim[0]
                for word in corpus_offline[sim[0][0]]:
                    print dictionary[word[0]],
                print ''
        # current_doc_id += 1
    print ''
    print 'filtered corpus size= %d' %len(filtered_corpus)
    return filtered_corpus, filtered_metas

 

def loadOffline(prefix='tmp_test'):
    dictionary = corpora.Dictionary.load('%s.dict' %prefix)
    corpus = list(corpora.MmCorpus('%s.mm' %prefix))
    tfidf = models.TfidfModel.load('%s.model' %prefix)
    index = similarities.Similarity.load('%s.index' %prefix)
    
    return dictionary, corpus, tfidf, index


def loadEventGT(dictionary, timePoint=1347724800, step = 1, gt_Nh=0, gt_Nd=0, event_id=0, jobID='tmpgt_', inputEvent=set(), detectParam=None, verbose=False):
    ret= db.getEventInfo(startTime=timePoint, historySlots=gt_Nh*step, detectionSlots=gt_Nd*step, event_id=event_id)
    corpus_src_gt=[]
    util_gt=[]
    event_gt=[]
    if ret:
        # print 'gt_event:'
        for event in ret:
            if len(inputEvent)>0:
                if event['event_id'] not in inputEvent:
                    continue
            if event['event_id_time']<(timePoint-gt_Nh*24*3600):
                continue
            
            # keyword=[word for word in event['split_words'].split(';') if word]
            # keyword=list(set(keyword))
            # gt.append(keyword)
            # print 'ID=%s: ABS=%s TIT=%s' %(event['event_id'], event['abstract'], event['title'])
            keywords= str(event['title'])
            wordseg = wc.para2seglist(keywords, tag=True)
            seglist = wc.seglist4filter(wordseg, srcTag = True, filterLow=False, fromFile=False)
            # print seglist
            seglist=list(set(seglist))
            # for element in seglist:
                # print element,
            # print ''
            corpus_src_gt.append(seglist)
            # print event['event_id'],
            # for word in seglist:
            #     print word,
            # print ''
            # 
            # [detectCount, previousErgency, db_event_id]
            if detectParam:
                detectThres= detectParam.detectThres
            else:
                detectThres=0
            # eventReportTime=event['time_get']
            
            # d = datetime.date(2015,1,5)
            # unixtime = time.mktime(d.timetuple())
            
            a_gt_util = gt_util_param(event['event_id'], detectThres, int(event['reportTime']), int(event['event_id_time']))
            util_gt.append(a_gt_util)
        print 'total gt size: %d' %len(corpus_src_gt)

    dictionary, corpus_gt, metas_gt = corpus_init(corpus_src_gt, dictionary, prefix='detect/%s_gt_%s_his%s_det%s_ev%s' %(jobID, helper.timestamp2str(timePoint), gt_Nh*step, gt_Nd*step, event_id), seg=False)
    tfidf_gt, index_gt = sims_init(corpus_gt, prefix='model/%s_gt_%s_his%sh_det%sh_ev%s' %(jobID, helper.timestamp2str(timePoint), gt_Nh * step, gt_Nd*step, event_id))
    # print 'dict size2=%s'%len(dictionary)
    if verbose:
        for i, corpus in enumerate(corpus_gt):
            print 'gtid=%s'%util_gt[i].db_event_id
            for word in corpus:
                print dictionary[word[0]],
            print ''
    return dictionary, corpus_gt, tfidf_gt, index_gt, util_gt


