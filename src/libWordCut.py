#!/usr/bin/env python
#encoding: utf8

from PyNLPIR import *
import os
import sys
import codecs
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf-8')


def fileLineCount(filename):
    lines = 0
    for line in open(filename):
        lines += 1
    return lines


def init_lib(init_dir = '.'):
    #init_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
    if init(init_dir, Constants.CodeType.UTF8_CODE):
        print 'NLPIR initialization succeed.'
    else:
        raise 'NLPIR initialization fail.'


#return list: [ [wordseg tag], ... ]
def para2seglist(srcPara, tag = True):
    word_segs=paragraph_process(srcPara, tag)
    words=word_segs.split(' ')
    output=[]
    for word in words:
        if word:
            if tag:
                result=word.rsplit('/',1)
                output.append(result)
            else:
                output.append([word])
    return output


#return list: [[[[wordseg, tag],[wordseg, tag] ... ], DSTTIME], ... ]
def file2seglist(srcFile, tag = True, countLine = True):    
    file = open (srcFile)
    output = []
    count = 0
    if countLine:
        count = 0
        lineCount = fileLineCount(srcFile)
        point = lineCount / 20 + 1
        print '%s segmenting %d lines' %(srcFile, lineCount)

    for line in file:
        if countLine:
            count += 1
            if (count % point) == 0:
                print ('%d%%' %(5*count/point)),
                sys.stdout.flush()
        
        if line.strip():
            lines=line.strip().split('\t')
            DSTTIME=''
            contentPart=''
            for part in lines:
                if part.startswith('DSTTIME'):
                    DSTTIME=part.replace('DSTTIME', '')
                else:
                    contentPart=contentPart+'\t'+part
            output.append([para2seglist(contentPart.strip(), tag), DSTTIME])
    #print ''
    return output


def seglist4filter(srcList, srcTag = True, filterLow = True, fromFile = True):
    output = []
    outputTime=[]
    #test whether srcList is from para2list or file2list
    # fileFlag = True
    if fromFile:
        for para in srcList:
            if para[0]:
                outpara=[]
                for word in para[0]:
                    if word:
                        if srcTag:
                            if len(word)<2:
                                continue
                            else:
                                if not stopWordType(word[1]):
                                    outpara.append(word[0].decode('utf-8', 'ignore'))
                        else:
                            outpara.append(word[0].decode('utf-8', 'ignore'))
            outpara=filterStopWord(outpara)
            if len(outpara)>0:
                if not para[1]=='':
                    output.append(outpara)
                    outputTime.append(int(para[1]))
    else:
        for word in srcList:
            if word:
                # output=[]
                if srcTag:
                    if len(word) < 2:
                        continue
                    else:
                        # print word[0],word[1]
                        if not stopWordType(word[1]):
                            output.append(word[0].decode('utf-8', 'ignore'))
                else:
                    output.append(para[0].decode('utf-8', 'ignore'))
            
        output = filterStopWord(output)
        # output.append(outputpara)

    if filterLow:
        output = filterLowFreq(output)
    # output = [ele for ele in output if ele]
    if fromFile:
        return output, outputTime
    else:
        return output


def stopWordType(wordType):
    stopType = set('w u m y'.split())
    for atype in stopType:
        if wordType.startswith(atype):
            return True
    return False


def filterStopWord(src):
    stoplist = set('的 地 得 了 吧 //'.split())
    if isinstance(src, list):
        output=[word for word in src if word not in stoplist and not word.startswith('http') and not word.startswith('@')]
        return output
    else:
        if src in stoplist or src.startswith('http') or src.startswith('@'):
            return ''
    return src


def filterLowFreq(srcList):
    FreqThres = 1
    all_tokens = sum(srcList, [])
    tokenCount=defaultdict(int)
    for i in all_tokens:
        tokenCount[i] += 1
    
    output = []
    for text in srcList:
        outputInner=[word for word in text if tokenCount[word] >= FreqThres ]
        if outputInner:
            output.append(outputInner)
    
    # output = [[word for word in text if tokenCount[word] >= FreqThres] for text in srcList]
    # tokens_low = set(word for word in set(all_tokens) if all_tokens.count(word) <= FreqThres)
    # output = [[word for word in text if word not in tokens_low] for text in srcList]
    return output


#the following add/delete will NOT be effectiven, unless you save it!
def saveUserWord():
    return save_user_dict()


#return bool: add success or not
def addUserWord(word, tag):
    if word.find(' '):
        return add_user_word('[%s]\t%s' %(word, tag))
    return add_user_word('%s\t%s' %(word, tag))


#return bool: delete success or not
def deleteUserWord(word, tag):
    if word.find(' '):
        return delete_user_word('[%s]\t%s' %(word, tag))
    return delete_user_word('%s\t%s' %(word, tag))


#return int: added FieldDict count
def importDict(dictFile):
    return import_user_dict(dictFile)


#return int: 0 for reset success
#input a null file, so userDict will be overwrite (so it's reset!)
def resetDict():
    return import_user_dict(os.devnull)


