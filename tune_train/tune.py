# -*- coding: utf-8 -*-

from glob import glob
from os import chcwd, getcwd, makedirs
from os.path import isdir
from subprocess import check_call
import sys

def mkdirp(d):
    if not isdir(d):
        makedirs(d)

def get_best_valid_cer(f):
    if isinstance(f, str):
        f = open(f, 'r')
    best_cer_valid = 100000000000
    best_line = []
    for line in f:
        line = line.split()
        if len(line) != 6 or line[0] == 'EPOCH':
            continue
        valid_cer = float(line[-1])
        if valid_cer < best_cer_valid:
            best_cer_valid = valid_cer
            best_line = line
    f.close()
    return best_cer_valid, best_line

def main(job_id, params):
    # CWD to main project dir
    cwd = getcwd()
    chcwd('..')
    # Prepare torch arguments
    args = ['th', 'train.lua']
    # Extend arguments list with the given parameters
    for (p,v) in params.iteritems():
        args.extend(['-%s' % p, str(v)])
    # Fixed options and arguments
    args.extend(['-max_no_improv_epochs', '10'])
    args.extend(['-output_path', '%s/%08d' % (cwd, job_id)])
    args.append('%s/%08d/init.t7' % (cwd, job_id))
    args.append('data/bentham_iclef16/train.h5')
    args.append('data/bentham_iclef16/valid.h5')
    # Call torch and parse CSV file with results
    print >> sys.stderr, ' '.join(args)
    mkdirp('%s/%08d' % (cwd, job_id))
    check_call(args)
    best_cer_valid, best_line = \
        get_best_valid_cer(glob('%s/%08d/*.csv' % (cwd, job_id)))
    print >> sys.stderr, 'EPOCH BEST LOSS_TRAIN LOSS_VALID CER_TRAIN CER_VALID'
    print >> sys.stderr, best_line
    # CWD to previous dir
    chcwd(cwd)
    return best_valid_cer
