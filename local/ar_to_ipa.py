import fst
import codecs
import os
import cPickle as pickle
import sys
from kitchen.text.converters import getwriter
#import subprocess

DATA_DIR = sys.argv[1]
ALT_AR_FST = '/export/ws15-pt-data/rsloan/Arabic_ref_orthography_fst.txt.1'
AR_FST_FILE = '/export/ws15-pt-data/rsloan/phonetisaurus-0.8a/phonetisaurus/script/ar_data/ar_pron.fst'
OUT_DIR = "/export/ws15-pt/data/rsloan/arabic_ipa"
CALLHOME_DICT = "/export/ws15-pt-data/rsloan/ar_pron.p"
CALLHOME_FST = "/export/ws15-pt-data/rsloan/callhome_to_ipa.txt"
QAC_DICT = '/export/ws15-pt-data/rsloan/qac.p'
OOV_DICT = '/export/ws15-pt-data/rsloan/new_oov_dict.p'

def create_ar_fst():
    '''creates fst for converting callhome dictionary pronunciations to arabic'''
    ar_fst = fst.Transducer()
    fst_file = codecs.open(ALT_AR_FST, 'r', encoding='utf-8')
    for l in fst_file:
        l = l.replace(u'\ufeff', '')
        rule = l.split()
        if len(rule)==4:
            if rule[2] == 'eps':
                rule[2] = u'\u03b5'
            if rule[3] == 'eps':
                rule[3] = u'\u03b5'
            ar_fst.add_arc(int(rule[0]), int(rule[1]), rule[2], rule[3])
    ar_fst[1].final = True
    fst_file.close()
    return ar_fst

def create_ipa_fst():
    '''creates fst for converting callhome dictionary pronunciations to arabic'''
    ipa_fst = fst.Transducer()
    fst_file = codecs.open(CALLHOME_FST, 'r', encoding='utf-8')
    for l in fst_file:
        l = l.replace(u'\ufeff', '')
        rule = l.split()
        if len(rule)==4:
            ipa_fst.add_arc(int(rule[0]), int(rule[1]), rule[2], rule[3])
    ipa_fst[1].final = True
    fst_file.close()
    return ipa_fst

#ipa_fst = create_ipa_fst()
utt_names = open(sys.argv[2], 'r')
#ar_dict = pickle.load(open(CALLHOME_DICT, 'rb'))
qac_dict = pickle.load(open(QAC_DICT, 'rb'))
oov_dict = pickle.load(open(OOV_DICT, 'rb'))
#ar_fst = fst.read(AR_FST_FILE)
alt_ar_fst = create_ar_fst()
#ar_fst.write('qac_test.fst')
extra_oovs = codecs.open('extra_oov.txt', 'w', encoding='utf-8')

for name in utt_names:
    name = name.rstrip()
    f = codecs.open(DATA_DIR+"/"+name+".txt", 'r', encoding='utf-8')
    outstring = ''
    for l in f:
        words = l.split()
        for w in words:
            extra_chars = [' ', u'\ufeff', '-', u'\u060c', u'\u060d', u'\u060e', u'\u060f', u'\ufd3e', u'\ufd3f',
                               ':','[',']', '(', ')', '\"',u'\u0640']
            for c in extra_chars:
                w = w.replace(c, '')
            end_sil = False
            end_punc = [u'\u061f', '!','.']
            for c in end_punc:
                if c in w:
                    end_sil = True
                    w = w.replace(c, '')
            if len(w) < 1:
                if end_sil:
                    outstring += ' SIL'
                continue
            w = w.encode('utf-8')
            try:
                pron = qac_dict[unicode(w,'utf-8')]
                #ipa_pron = True
                #print 'word found in QAC dict'
            except KeyError:
                try:
                    '''if unicode(w, 'utf-8') == u'\u0621':
                        pron = 'C'
                    else:
                        pron = ar_dict[w]
                    #print 'word in callhome dict'
                    ipa_pron = False'''
                    pron = oov_dict[unicode(w,'utf-8')]
                except KeyError:
                    '''ipa_pron = False
                    #print 'word is OOV, attempting phonetisaurus lookup'
                    w = unicode(w, 'utf-8')
                    pron = subprocess.check_output(['phonetisaurus-g2p', '--model='+AR_FST_FILE,'--input='+w.encode('utf-8')])
                    pron = pron.rstrip()
                    endnum = pron.index('\t')
                    pron = pron[endnum+1:]
                    pron = ''.join(pron.split())
                    pron = pron.decode('utf-8')
                    #manually handle characters not in my phonetisaurus fst
                    pron = pron.replace(u'\u0644', u'\006c')
                    pron = pron.replace(u'\u0622', 'CA')
                    print 'reverting to phonetisaurus' '''
                    wfst = fst.linear_chain(unicode(w,'utf-8'), syms=alt_ar_fst.isyms)
                    ppath = wfst.compose(alt_ar_fst).shortest_path(1)
                    pron = ''
                    for path in ppath:
                        pron += ''.join(ppath.osyms.find(arc.olabel) for arc in path)
            #try:
            pron = pron.encode('utf-8')
            pron = pron.replace('\x06', '')
            #if ipa_pron:
            pron = unicode(pron, 'utf-8')
            longcs = [] #long consonants to be replaced
            extra_long = [] #long consonants that would be replaced by two colons if this didn't exist
            for i in range(len(pron)-1):
                if pron[i] == pron[i+1] and pron[i] not in [u'a', u'u', u'i',u"\u02D0"]:
                    longcs.append(pron[i])
                    if i < len(pron)-2 and pron[i+2] == u"\u02D0":
                        extra_long.append(pron[i])
            for c in longcs:
                pron = pron.replace(c+c, c+u"\u02D0")
            for c in extra_long:
                pron = pron.replace(c+u'\u02D0'+u'\u02D0', c+c+u'\u02D0')
            outstring += ' ' + ' '.join(pron)
            '''    else:
                    pfst = fst.linear_chain(pron,syms=ipa_fst.isyms)
                    bestpath = pfst.compose(ipa_fst).shortest_path(1)
                    for path in bestpath.paths():
                        pathout = ' '.join(bestpath.osyms.find(arc.olabel) for arc in path)
                        outstring += ' ' + pathout
            except NameError:
                #print "word has character not found in phonetisaurus dict: " + w
                pfst.write('pfst.fst')
                wfst = fst.linear_chain(w, syms=alt_ar_fst.isyms)
                ppath = wfst.compose(alt_ar_fst).shortest_path(1)
                for path in ppath:
                    pathout = ' '.join(ppath.osyms.find(arc.olabel) for arc in path)
                    outstring += ' ' + pathout'''
            if end_sil:
                outstring += ' SIL'
        outstring = outstring.replace(u'\u03b5', '')
        #make sure multi-character phones aren't being split
        outstring = outstring.replace(u' \u02d0', u'\u02d0')
        outstring = outstring.replace(u' \u02e4', u'\u02e4')
        outstring = outstring.replace(u'\u0064 \u0292', u'\u0064\u0292')
        outstring = outstring.replace(u'\u0061 \u026a', u'\u0061\u026a')
        outstring = outstring.replace(u'\u0061 \u028a', u'\u0061\u028a')
        outstring = ' '.join(outstring.split())
        UTF8Writer = getwriter('utf8')
        sys.stdout = UTF8Writer(sys.stdout)
        print outstring
    f.close()
