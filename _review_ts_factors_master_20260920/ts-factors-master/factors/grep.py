# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 23:52:02 2020

@author: NJU
"""


import os

grep_str = 'data-kyts'
grep_str2 = ''
no_grep_str3 = ''

grep_str = grep_str.lower()
grep_str2 = grep_str2.lower()


flist = os.listdir()
for fname in flist:
    if fname=='grep.py': continue
    #if '.bak' in fname: continue
    #if 'strategy_traders' in fname: continue
    #if not '.py' in fname: continue
    #if 'strategy_traders.py' not in fname: continue
    for e in ['utf-8', 'ANSI']:
        try:
            with open(fname, 'r', encoding=e) as fp:
                tag_print_fname = False
                for sline in fp:
                    ss = sline.lower()
                    if grep_str in ss and grep_str2 in ss and (no_grep_str3=='' or no_grep_str3 not in ss):
                        if not tag_print_fname:
                            print('file -----> ', fname)
                            tag_print_fname = True
                        print('    ', sline)
        except:
            #print('fail to process ', fname)
            pass
