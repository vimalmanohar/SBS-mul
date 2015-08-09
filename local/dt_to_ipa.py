import fst
import codecs
import sys
import cPickle as pickle
import subprocess
from kitchen.text.converters import getwriter

DATA_DIR = sys.argv[1]
FILE_LIST = sys.argv[2]
DUTCH_FST_FILE = '/export/ws15-pt-data/rsloan/Dutch_ref_orthography_fst.txt'
DUTCH_DICT = '/export/ws15-pt-data/rsloan/dt_pron.p'
EN_DICT = '/export/ws15-pt-data/rsloan/en_pron.p'
PHONETISAURUS_FILE = '/export/ws15-pt-data/rsloan/phonetisaurus-0.8a/phonetisaurus/script/dt_data/dt_pron.fst'
#OUTFILE = '/export/ws15-pt-data/rsloan/dt_oov.txt'

def create_dt_fst():
    dt_fst = fst.Transducer(isyms=fst.SymbolTable('eps'), osyms=fst.SymbolTable('eps'))
    fst_file = codecs.open(DUTCH_FST_FILE, 'r', encoding='utf-8')
    
    for l in fst_file:
        l = l.replace(u'\ufeff', '')
        entry = l.split()
        if len(entry) == 4:
            if entry[3] == 'ks':
                entry[3] = 'k s'
            dt_fst.add_arc(int(entry[0]), int(entry[1]), entry[2], entry[3])

        dt_fst[1].final = True
        dt_fst[2].final = True
    return dt_fst

def process_numerals(l):
    '''replaces numerals with their Dutch translations and treats each numeral as a separate word
    also has the separate effect of standardizing spacing, but that shouldn't matter in this script'''
    l = l.replace('1', ' een ')
    l = l.replace('9', ' negen ')
    l = l.replace('3', ' drie ')
    l = l.replace('4', ' vier ')
    l = l.replace('2', ' twee ')
    l = l.replace('5', ' vijf ')
    l = l.replace('6', ' zes ')
    l = l.replace('7', ' zeven ')
    l = l.replace('8', ' acht ')
    l = l.replace('0', ' zero ')
    return l

dt_fst = create_dt_fst()
dt_dict = pickle.load(open(DUTCH_DICT, 'rb'))
en_dict = pickle.load(open(EN_DICT, 'rb'))
#outfile = codecs.open(OUTFILE, 'w', encoding='utf-8')

def word_to_pron(word):
    '''given a Dutch word, outputs a pronunciation (with phonemes separated by spaces)'''
    try: #look in dict
        if word == '<silence>':
            return 'SIL'
        else:
            return dt_dict[word]
    except KeyError:
        #outfile.write(word + '\n')
        #return word
       # if len(word) < 15:
        #print "using phonetisaurus on word: " + word
        pron = subprocess.check_output(['phonetisaurus-g2p', '--model='+PHONETISAURUS_FILE,'--input='+word])
        pron = pron.rstrip()
        endnum = pron.index('\t')
        pron = pron[endnum+1:]
        pron = unicode(pron, 'utf-8')
        pron = pron.replace(u"\u0279", 'r')
        pron = pron.replace('h', u"\u0266")
        pron = pron.replace(u"\u0289", u"\u0259")
        pron = pron.replace(u"\u0264", u"\u0263")
        #if len(word) <= 2 or len(pron.split()) > len(word)/2:
        return pron
        #try: #check if English
        #    pron = en_dict[word]
       #     return pron
        #except KeyError: #OOV word, revert to fst

          #phonetisaurus probably couldn't get through word, use FST instead
        #print "reverting to FST for " + word
        #wfst = fst.linear_chain(word, syms=dt_fst.isyms)
        #bestpath = wfst.compose(dt_fst).shortest_path(1)
        #for path in bestpath.paths():
        #    pron = ' '.join(bestpath.osyms.find(arc.olabel) for arc in path)
        #return pron

utts = open(FILE_LIST, 'r')

for name in utts:
    name = name.strip()
    utt = codecs.open(DATA_DIR+'/'+name+'.txt', 'r', encoding='utf-8')
    for l in utt:
        outstring = ''
        l = l.replace('-', ' ') #treat hyphens as spaces when splitting words
        l = l.replace('/', ' ')
        l = process_numerals(l)
        words = l.split()
        for w in words:
            w = w.replace(u"\u00EB", 'e')
            w = w.replace(u"\u00EF", 'i')
            w = w.replace('&', 'en')
            extra_chars = [',', '\'',u'\ufeff',':','\"','(',')']
            end_punc = ['.', '?', '!']
            for c in extra_chars:
                w = w.replace(c, '')
            end_sil = False
            for c in end_punc:
                if c in w:
                    w = w.replace(c, '')
                    end_sil = True
            
            if len(w) > 0:
                outstring += word_to_pron(w.lower()) + ' '
            if end_sil:
                outstring += 'SIL '
        outstring = outstring.replace(u'eps', '')
        outstring = ' '.join(outstring.split()) #remove extra space, just in case
        UTF8Writer = getwriter('utf8')
        sys.stdout = UTF8Writer(sys.stdout)
        print outstring
