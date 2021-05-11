#!/usr/bin/env python
# Miten Jain
# make_profile_from_fasta.py

from __future__ import print_function
import sys, time, glob, os
import numpy
import collections
from optparse import OptionParser

class Fastaseq():
    '''
    fasta reader
    '''
    def __init__(self):
        self.id = None
        self.seq = ''
        self.length = ''
      
    @staticmethod 
    def readline(infile):
        seqobj = Fastaseq()
        for line in infile:
            if len(line)==0: 
                print('empty line', file=sys.stderr)
                continue
            if line.startswith('>'):
                if seqobj.id is None:
                    seqobj.id = line.rstrip()
                    continue
                else:
                    yield seqobj
                    seqobj = Fastaseq()
                    seqobj.id = line.rstrip()
            else:
                seqobj.seq += line.rstrip('\n\r').upper()
        yield seqobj    

########################################################################
# Main
# Here is the main program
########################################################################

def main(myCommandLine=None):

    t0 = time.time()

    parser = OptionParser(usage='usage: ./fastq_length.py --in ./fastq --low 0 --high  \
                                        1000000', version='%prog 0.0.2')

    #Options
    parser.add_option('--in', dest='infile', help='fasta file', default='')
    parser.add_option('--out', dest='outdir', help='outpit dir', default='.')
    parser.add_option('--name', dest='name', help='model name', default='model')
    parser.add_option('--type', dest='type', help='data type (dna/rna)', default='rna')

    #Parse the options/arguments
    options, args = parser.parse_args()
    #Print help message if no input
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    print(options, file=sys.stderr)

    inFile = options.infile
    outDir = options.outdir
    modelName = options.name
    dataType = options.type

    if dataType == 'dna':
        kmerModel = './kmer_models/r9.4_180mv_450bps_6mer/template_median68pA.model'
        kmerLen = 6
    else: # rna
        kmerModel = './kmer_models/r9.4_180mv_70bps_5mer_RNA/template_median69pA.model'
        kmerLen = 5

    # read kmer model
    file = open(kmerModel, 'r')
    kmer_dict = collections.defaultdict()
    # read header
    # kmer, level_mean, level_stdv, sd_mean, sd_stdv, ig_lambda, weight
    header = file.readline()
    for line in file:
        line = line.strip().split()
        kmer_dict[line[0]] = float(line[1]), float(line[2])
    file.close()    

    ### Important for RNA ###
    # string 5'->3'
    # >fMet
    # AGCAAGAAGAAGCCTGGTCGCGGGGTGGAGCAGCCTGGTAGCTCGTCGGGCTCATAACCCGAAGGTCGTCGGTTCAAATCCGGCCCCCGCAACCAGGCTTC
    # inSeq = 'AGCAAGAAGAAGCCTGGTCGCGGGGTGGAGCAGCCTGGTAGCTCGTCGGGCTCATAACCCGAAGGTCGTCGGTTCAAATCCGGCCCCCGCAACCAGGCTTC'
    # invert the string so it is 3'->5'
    # inSeq_rev = inSeq[::-1]
    ### ###

    # read fasta file
    file = open(inFile, 'r')
    for seq in Fastaseq.readline(file):
        header = seq.id
        fileHeader = header.replace('>', '')
        if dataType == 'rna':
            inSeq = seq.seq[::-1]
        else:
            inSeq = seq.seq
        counter = 0
        # write a model per fasta record
        outfile = outDir + '/' + modelName + '_' + fileHeader + '.txt'
        outFile = open(outfile, 'w')
        for i in range(0,len(inSeq)-kmerLen+1):
            kmer = inSeq[i:i+kmerLen]
            kmer_mean = str(kmer_dict[kmer][0])
            kmer_stdv = str(kmer_dict[kmer][1])
            print('\t'.join([str(counter), kmer_mean, kmer_stdv, kmer]), file=outFile)
            counter += 1
        outFile.close()
    file.close()

    print('\ntotal time for the program %.3f' % (time.time()-t0), file=sys.stderr)

if (__name__ == '__main__'):
    main()
    raise SystemExit

