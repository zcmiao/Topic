#!/usr/bin/env python
#encoding: utf8

import libWordCut as wc


if __name__ == '__main__':
    
    wc.init_lib('../python-nlpir')

    text = '24小时降雪量24小时降雨量863计划ABC防护训练APEC会议BB机BP机C2系统C3I系统C3系统C4ISR系统C4I系统CCITT建议'
    print wc.para2list(text, True)
 