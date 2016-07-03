#!/usr/bin/env python
#encoding: utf8

import pymysql as mdb
from datetime import datetime
import time
import sys
import networkx as nx
import operator
reload(sys)
sys.setdefaultencoding('utf-8')
import libHelper as helper
 
IP='localhost'
USERNAME='root'
PASSWORD='password'
DATABASE='database'

def new_conn():
    conn = mdb.connect(IP, USERNAME, PASSWORD, DATABASE, charset="utf8");
    cur= conn.cursor(mdb.cursors.DictCursor)
    return conn,cur
  

def getUserInfoAll(userInvolveList=''):
    conn,cur=new_conn()
    with conn:
        cur.execute("select * from `user_info_all` ")# limit 0,1000")
        rows = cur.fetchall()
        # conn.close()
        if userInvolveList:
            rows=[row for row in rows if row['uid'] in userInvolveList]
            return rows
        else:
            return rows
        
def getUserList(method=''):
    conn,cur=new_conn()
    with conn:
        if method:
            cur.execute("select uid from `uid_weibo_info` where method_%s=1" %method)
        else:
            cur.execute("select uid from `uid_weibo_info`  ")
        rows = cur.fetchall()
        # conn.close()
        userList=set()
        for user in rows:
            userList.add(user['uid'])
        #userList = [str(user['uid']) for user in rows]
        return userList

def markUser(userList, method='test', overwrite=False):
    conn,cur=new_conn()

    with conn:
        cur.execute("select * from information_schema.COLUMNS where TABLE_SCHEMA='%s' "\
        "and TABLE_NAME='uid_weibo_info' and COLUMN_NAME='method_%s'" %(DATABSE, method))
        existFlag=False
        if cur.fetchone():
            existFlag=True
        if existFlag:
            if overwrite:
                db_name='method_%s' %method
                cur.execute("ALTER TABLE uid_weibo_info DROP COLUMN %s" %db_name)
                cur.execute("ALTER TABLE uid_weibo_info ADD %s bit default 0" %db_name)
            else:
                db_name='method_%s_%s' %(method, time.strftime("%Y_%m_%d_%H_%M_%S"))
                cur.execute("ALTER TABLE uid_weibo_info ADD %s bit default 0" %db_name)
        else:
            db_name='method_%s' %method
            cur.execute("ALTER TABLE uid_weibo_info ADD %s bit default 0" %db_name)
        for user in userList:
            cur.execute("UPDATE uid_weibo_info set %s=1 where uid=%s" %(db_name, user))
    conn.close()


def getEventInfo(startTime=1347724800, perSlot=3600, historySlots=12, detectionSlots=12, event_id=0, ct=750):
    conn,cur=new_conn()
    with conn:
        if event_id!=0:
            cur.execute("select *,UNIX_TIMESTAMP(time_get) as reportTime from dataset_2_event_type_raw_database where event_id in \
            (select event_id from \
            (select event_id,count(*) as ct from dataset_2_weibo_spread where rttime between %s and %s group by event_id) as tmp \
            where ct>%s) and valid=1 and event_id = %s" %(startTime-perSlot*historySlots, startTime+perSlot*detectionSlots, ct, event_id) )
        else:
            cur.execute("select *,UNIX_TIMESTAMP(time_get) as reportTime from dataset_2_event_type_raw_database where event_id in \
            (select event_id from \
            (select event_id,count(*) as ct from dataset_2_weibo_spread where rttime between %s and %s group by event_id) as tmp \
            where ct>%s) and valid=1" %(startTime-perSlot*historySlots, startTime+perSlot*detectionSlots, ct))

        rows = cur.fetchall()
        # conn.close()
        return rows


def getInvolveUser(ret):
    involveUser=set()
    for line in ret:
        involveUser.add(line['dstuid'])
    return involveUser


#2012-09-25 0:0:0 1348502400
def getText(timePoint=1348502400, perSlot=3600, historySlots=0, detectionSlots=0, timelastsSlots=0, output_prefix='data/test_', event_id=set(), util_gt=None, userList=set(), cacheFlag=False, cacheRet=None, keys=['rttext','srctext','dsttext','dsttime'], verboseFlag=True, singleUserID=0):
    conn,cur=new_conn()
    with conn:
        filename=output_prefix
        if cacheRet == None:
            ret=[]
            if util_gt!=None and len(event_id)>0:
                filename = '%s_start%sh_lasts%sh_evl%s' %(filename, historySlots, timelastsSlots, len(event_id))
                for event in util_gt:
                # event_query="and (event_id=" + ' or event_id='.join(event_id_str) + ') '
                    cur.execute("select event_id,dstuid,dsttime,rttext,srctext,dsttext,srcmid,dstmid,srcuid from dataset_2_weibo_spread where event_id = %s and dsttime between %s and %s order by dsttime" %(event.db_event_id, event.topicStartTime, event.topicStartTime+perSlot*timelastsSlots)) 
                    beforelen=len(ret)
                    ret.extend(cur.fetchall())
                    print 'topic %s has total %s posts' %(event.db_event_id, len(ret)-beforelen)
            else:
                cur.execute("select event_id,dstuid,dsttime,rttext,srctext,dsttext,srcmid,dstmid,srcuid from dataset_2_weibo_spread where dsttime between %s and %s order by dsttime" %(timePoint-perSlot*historySlots, timePoint+perSlot*detectionSlots)) 
                filename = '%s_start%sh_end%sh_ev%s' %(filename, historySlots, detectionSlots, len(event_id))
                ret= cur.fetchall()
                # print len(ret)

            if len(ret)>0:
                if len(userList)==0:
                    #if userlist = full, don't return ret_dict_key_by_user
                    outputUserSet=set()
                    for line in ret:
                        outputUserSet.add(line['dstuid'])
                    if not cacheFlag:
                        writeQuery2File(filename, ret, keys=keys, verbose=verboseFlag)
                        return filename, outputUserSet
                    else:
                        writeQuery2File(filename, ret, keys=keys, verbose=verboseFlag)
                        return filename, outputUserSet, ret

                #else userlist = part, return ret_dict_key_by_user
                else:
                    outret={}
                    outputret=[]
                    outputUserSet=set()
                    for line in ret:
                        if line['dstuid'] in userList:
                            outputUserSet.add(line['dstuid'])
                            if not cacheFlag:
                                outputret.append(line)
                            else:
                                if line['dstuid'] not in outret:
                                    outret[line['dstuid']]=[]
                                outret.get(line['dstuid']).append(line)
                                # print type(outret.get(line['dstuid']))
                    if not cacheFlag:
                        if len(outputret)>0:
                            # outputret=[]
                            # for line in outret.values():
                            #     outputret.extend(line)
                            writeQuery2File(filename, outputret, keys=keys, verbose=verboseFlag)
                            return filename, outputUserSet
                        else:
                            print '%s is empty' %filename
                            return None, set()
                    else:
                        if len(outret)>0:
                            return filename, outputUserSet, outret
                        else:
                            print '%s is empty' %filename
                            return None, set(), None

            else:
                if not cacheFlag:
                    print '%s is empty' %filename
                    return None, set()
                else:
                    return None, set(), None


        else:
            outret=cacheRet
            verboseFlag=False
            outputret=[]
            outputUserSet=set()

            if len(userList)==0:
                outputret=[]
                for user in outret.keys():
                    outputret.extend(outret.get(user, []))
                    outputUserSet.add(user)
                writeQuery2File(filename, outputret, keys=keys, verbose=verboseFlag)
                return filename, outputUserSet

            else:
                for user in userList:
                    outputret.extend(outret.get(user, []))
                    outputUserSet.add(user)
                writeQuery2File(filename, outputret, keys=keys, verbose=verboseFlag)
                return filename, outputUserSet
                
    # conn.close()
            

def writeQuery2File(filename, ret, keys,  verbose=True):
    with open(filename, 'w') as file:

        for line in ret:
 
            output=[]
            for key in keys:
                if key=='dsttime':
                    output.append('DSTTIME%d' %line[key])
                else:
                    output.append(line[key].decode('utf-8'))
            file.write('%s\n' %('\t'.join(output)))
        if verbose:
            print '%s writing done' %(filename)
    file.close()

    
def selectUserPageRank(c, k, method, userInvolveList, corpus):
    userInfo = getUserInfoAll(userInvolveList)
    userSet=set()
    userCosts={}
    for user in userInfo:
        userSet.add(user['uid'])
        userCosts[user['uid']]=user['cost']
    
    bypassCnt = 0
    G = nx.MultiDiGraph()
    for line in corpus:
        if line['dstuid'] not in userSet or line['srcuid'] not in userSet:
            bypassCnt += 1
            continue
        G.add_edge(line['dstuid'], line['srcuid'])
        
    print 'Graph has %d nodes and %d edges' %(len(G.nodes()), len(G.edges()))
    pagerank = nx.pagerank_scipy(G, alpha=0.9)
    pageranks = sorted(pagerank.items(), key=operator.itemgetter(1), reverse=True)
    
    cur_c=0
    cur_k=0
    selected_uid = []
    for uid, pr in pageranks:
        if cur_k>=k or cur_c>=c:
            break
            # print 'selected done!, cur_c=%d, cur_k=%d' %(cur_c, cur_k)
            # return selected_uid
        else:
            if uid in pagerank:
            # if userInfo[i]['avr_rp_count']>0.2:
                selected_uid.append(uid)
                cur_c +=  userCosts[uid]
                cur_k += 1
                print 'select %s, pr %s' %(uid, pr)
            else:
                continue
    print 'selected done!, cur_c=%d, cur_k=%d' %(cur_c, cur_k)
    return selected_uid
    
def selectUserbaseline(c, k, method='followers', userInvolveList=''):
    # method= followers_count, friends_count, event_count, mid_count, rp_count, avg_rp_count, cm_count, avg_cm_count...
    method=method + '_count'
    userInfo = getUserInfoAll(userInvolveList)
    #sort by fo desc, and cost asce
    userInfo=sorted(userInfo, key= lambda x: (x[method], -x['cost']), reverse=True)
    G = nx.MultiDiGraph()
    
    cur_c=0
    cur_k=0
    selected_uid = []
    for i in range(0,len(userInfo)):
        if cur_k>=k or cur_c>=c:
            print 'selected done!, cur_c=%d, cur_k=%d' %(cur_c, cur_k)
            return selected_uid
        else:
            if userInfo[i]['avr_rp_count']>0.2:
                selected_uid.append(userInfo[i]['uid'])
                cur_c +=  userInfo[i]['cost']
                cur_k += 1
            else:
                continue
    print 'selected done!, cur_c=%d, cur_k=%d' %(cur_c, cur_k)
    return selected_uid



