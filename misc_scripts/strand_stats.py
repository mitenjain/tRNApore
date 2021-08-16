#!/usr/bin/env python
# Miten Jain
# strand_stats.py


##################################################################################
# Imports
##################################################################################
import sys, argparse, time, os, glob, h5py, numpy, collections
from optparse import OptionParser
##################################################################################

##################################################################################
# fast5 reader
##################################################################################
def fast5reader(in_path, file, count, error):

    file = os.path.join(in_path, file)
    try:
        hdf = h5py.File(file, 'r')
        count += 1
    except:
        hdf = None
        error += 1

    return hdf, count, error
##################################################################################

##################################################################################
# Fetch read duration times from read or base called  fast5 files
##################################################################################
def read_times(in_path, files):
    print >> sys.stderr, 'sequencing time from reads'
    raw_meta = '/Raw/Reads/'
    channel_meta = '/UniqueGlobalKey/channel_id/'

    count = 0
    error = 0
    read_durations = {}
    for file in files:
        file = os.path.join(in_path, file)
        try:
            hdf = h5py.File(file, 'r')
            # Get attributes for this channel, used to convert current to pA
            attributes = hdf[channel_meta].attrs.keys()
            digitisation = hdf[channel_meta].attrs['digitisation']
            sample_rate = hdf[channel_meta].attrs['sampling_rate']
            range = hdf[channel_meta].attrs['range']
            offset = hdf[channel_meta].attrs['offset']
            for key in hdf[raw_meta]:
                raw_signal_meta = raw_meta + key
                raw_signal = raw_signal_meta + '/Signal'
                raw_time = hdf[raw_signal_meta].attrs['duration']
                read_time = raw_time / sample_rate
                if not file in read_durations:
                    read_durations[file] = 0.0
                read_durations[file] = read_time
            hdf.close()
        except:
            if hdf:
                hdf.close()
            error += 1
            continue
        count += 1
    print >> sys.stderr, 'Files completed ', count
    print >> sys.stderr, 'Files with error ', error
    print >> sys.stderr, 'Total Files ', len(files)

    return read_durations
##################################################################################

##################################################################################
# Iterate through a folder and get files in a list
##################################################################################
def get_files(in_path):
    for root, dirnames, filenames in os.walk(in_path):
        all_files = filenames

    # iterate through folder to create a list of fast5 files
    files = []
    for file in all_files:#[:50]:
        if file.endswith('.fast5'):
            files.append(file)
    
    # ensure fast5 files are present in the input folder
    if len(files) == 0:
        print >> sys.stderr, 'No fast5 files found, check input folder'
        sys.exit()

    return in_path, files
##################################################################################

##################################################################################
# Main
# Here is the main program
##################################################################################

def main():

    t0 = time.time()

    #Parse the inputs args/options
    parser = OptionParser(usage='usage: ./strand_stats.py -f ./fast5 --o out/', 
                          version='%prog 0.0.1')
    #Options
    parser.add_option('--files', dest='fast5Files', help='path to fast5 files')
    parser.add_option('--dwell', dest='readTimes', default = True, action='store_true', \
                       help='plot read times')

    #Parse the options/arguments
    options, args = parser.parse_args()

    #Print help message if no input
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    print options

    #parse the options/arguments
    fast5_path = options.fast5Files
    
    print options
    
    in_path, files = get_files(fast5_path)
    if options.readTimes:
        read_durations = read_times(in_path, files)

    # print dwell times
    print >> sys.stdout, 'File\tDuration (s)'
    for fileName in read_durations:
        print >> sys.stdout, fileName, '\t', read_durations[fileName]

    print >> sys.stderr, '\n', 'total time for the program %.3f' % (time.time()-t0)

if (__name__ == '__main__'):
    main()
    raise SystemExit
