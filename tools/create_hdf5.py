#!/usr/bin/env python

import argparse
import common
import h5py
import numpy
import scipy.misc

def sym2int(x, table):
    assert isinstance(x, list) or isinstance(x, tuple)
    assert isinstance(table, dict)
    return numpy.asarray(map(lambda s: table[s], x), dtype=numpy.int32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('transcripts', type=argparse.FileType('r'),
                        help='File containing the transcript of each sample, '
                        'one per line')
    parser.add_argument('symbols_table', type=argparse.FileType('r'),
                        help='File containing the mapping between symbols and '
                        'integers')
    parser.add_argument('imgs_dir', type=common.DirType(),
                        help='Directory containing the PNG images of each '
                        'of the samples in the transcripts file')
    parser.add_argument('output', type=str,
                        help='Output HDF5 file containing the dataset')
    args = parser.parse_args()

    assert args.output, 'Output filename cannot be empty!'

    # Read symbols table
    sym2int_table = {}
    ln = 0
    for line in args.symbols_table:
        ln += 1
        line = line.split()
        assert len(line) == 2, 'File %s has a wrong format at line %d' % \
            (args.symbols_table.name, ln)
        sym2int_table[line[0]] = int(line[1])
    args.symbols_table.close()

    h5f = h5py.File(args.output, 'w')
    ln = 0
    for line in args.transcripts:
        ln += 1
        line = line.split()
        assert len(line) > 0, 'File %s has a wrong format at line %d' % \
            (args.transcripts.name, ln)
        sid = line[0]
        grp = h5f.create_group(sid)
        grp['transcript'] = sym2int(line[1:], sym2int_table)
        grp['image'] = (255.0 - \
                        scipy.misc.imread('%s/%s.png' % (args.imgs_dir, sid),
                                          flatten=True, mode='L')) / 255.0
    h5f.close()
