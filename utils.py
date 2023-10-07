"""Utilities code

Author - Ximi
License - MIT
"""
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# -------------------- Utils ------------------------ #
def path_to_mp4(fname):
    return fname.split('.csv')[0] + '.mp4'
def path_to_csv(fname):
    return fname.split('.mp4')[0] + '.csv'

def extract_subject(fname):
    return fname.split('-')[0].split('/')[-1]

from scipy import stats

def majority_voting(labels):
    return stats.mode(labels)[0][0]
def read_file(path):
    try:
        with open(path) as f:
            dat = [i.strip('\n') for i in f.readlines()]
    except:
        return []
    return dat

def cleanXy(Xy):
    return [x for x in Xy if type(x[1])!=tuple]
# --------------------------------------------------- #