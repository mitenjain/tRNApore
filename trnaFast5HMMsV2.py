#!/usr/bin/env python
# Bryan Thornlow
# trnaFast5HMMs.py
# Uses Adam Novak's yahmm.py and Jacob Schreiber's PyPore
# Adapted from Miten Jain's trnaHMMs.py

import sys, string, time, math, glob, os
import argparse, itertools, collections
import numpy
import h5py
import colorsys
import pyximport
pyximport.install(setup_args={'include_dirs':numpy.get_include()})
from yahmm import *
from PyPore.core import Segment
from PyPore.parsers import *
from Fast5Types import *
import matplotlib.pyplot as plt
import seaborn as sns

########################################################################
# This program is designed to perform segmentation in parallel and make 
# decision dynamically on which segment to keep
# This is to be used with tRNA events
# 
#    1->2->3->4->5->6->7->8->9->10
# 
########################################################################

########################################################################
# Argument parser for specifying they type of HMM to be constructed
########################################################################

class CommandLine(object) :    
    '''
    Argument parser class. The arguments that can be given are:
    1. Input file/files - Whole path for file or folder containing a list of files
    2. Processing type - Specifies whether to perform segmentation and post-processing \
                        or parallel segmentation 
    '''
    def __init__(self) :
        self.parser = argparse.ArgumentParser(description = 'This program segments phi29 \
                                        DNAP events to be used by hmm scripts downstream', 
                        epilog = 'The program needs arguments to perform segmentation',
                        add_help = True, #default is True 
                        prefix_chars = '-', 
                        usage = '%(prog)s [options] -option1(default) -option2(required) \
                                 <input(required) >output(optional)')
        self.parser.add_argument('-i', '--path_to_files', action = 'store', \
                                dest='inpath', nargs='?', required = True, \
                                help='input folder with path')
        self.parser.add_argument('-m', '--path_to_models', action = 'store', \
                                dest='modelpath', nargs='?', required = True, \
                                help='models folder with path')
        self.args=self.parser.parse_args()


########################################################################
# Create all possible kmers
########################################################################

def kmer_current_map(file):
    # Read kmer current mean, sd, and time measurements
    kmer_current_dict = {}
    kmer_table = open(file, 'r')
    for line in kmer_table:
        line = line.strip().split('\t')
        key = line[0].strip()
        meanCurrent = float(line[1].strip())
        stdevCurrent = float(line[2].strip())
        if not key in kmer_current_dict.keys():
            kmer_current_dict[key] = 0
        kmer_current_dict[key] = [meanCurrent, stdevCurrent]

    kmer_table.close()

    return kmer_current_dict

########################################################################
# HMM Constructor class constructing input-specific HMMs
########################################################################

class HMM_Constructor():

    def __init__(self):
        pass

    def HMM_linear_model(self, kmer_list, kmer_current_dict, model_name=None):
        '''
        This HMM models the segments corresponding to the context and the label
        Each state will have the following transitions:
        1-step forward (expected) - 1 possible transition
        1-step back-slip - 1 possible transition
        1-step forward skip - 1 possible transition
        Self-loop
        '''
        # Create model and add states
        model = yahmm.Model(name=model_name) if not model_name is None \
                                                else yahmm.Model(name='HMM_linear_model')
        previous_skip = None
        previous_short_slip = None
        current_short_slip = None
        previous_mean = 0
        previous_sd = 0
        abasic = False # Abasic flag to trigger allowing re-read
        abasic_kmer = None # Track the position of abasic XXXX
        states_list = []

        for index in range(len(kmer_list)):
        
            # State name, mean and stdev for the kmer 
            kmer = kmer_list[index]
            current_mean = kmer_current_dict[kmer][0]
            current_sd = kmer_current_dict[kmer][1]
            # Transition probabilities for a match state
            # Self-loop to itself
            self_loop = 0.001
            # End from any state, i.e. reach model.end
            end = 0.005
            # Transitions for Drop-off State
            drop = 0.001
            # Transitions for going to Blip State
            blip = 0.005
            blip_self = 0.1
            # Back Slips, short and long
            slip = 0.05 if index > 0 else 0.00
            # Only short backslips possible
            short_slip = slip
            long_slip = 0.0
            # Transitions from silent slip states
            # Short slip from silent short slip state
            step_back = 0.60
            # Skip that accounts for a missed segment
            skip = 0.2
            # Transitions from current skip silent state to the previous match state or 
            # previous silent skip states
            long_skip = 0.05
            # Transitions for Insert state between two neighboring match states
            insert = 0.02 if index > 0 else 0.00
            # Self loop for an insert state
            ins_self = 0.05
            # Transition to the next match state (Forward Transition)
            # Each match state has transitions out to self_loop, end, drop, blip, slip, 
            # skip, insert, re_read, and forward
            forward = 1 - (self_loop + end + blip + slip + skip + insert)

            # Create and Add State
            current_state = yahmm.State(yahmm.NormalDistribution(current_mean, \
                                        current_sd), \
                                        name = 'M_' + kmer + '_' + str(index))
            model.add_state(current_state)

            # Transitions for the match state 
            # Self-loop to itself
            model.add_transition(current_state, current_state, self_loop)
            # The model could end from any match state
            if index < len(kmer_list) - 1:
                model.add_transition(current_state, model.end, end)

            # Each Match State can go to a silent drop-off state, and then to model.end
            drop_off = yahmm.State(None, name = 'S_DROPOFF_' + kmer + '_' + str(index))
            model.add_state(drop_off)
            # Transition to drop_off and back, from drop_off to end
            model.add_transition(current_state, drop_off, drop)
            model.add_transition(drop_off, current_state, 1.0 - blip_self)

            model.add_transition(drop_off, model.end, 1.00)

            # Each Match State can go to a Blip State that results from a voltage blip
            # Uniform Distribution with Mean and Variance for the whole event
            blip_state = yahmm.State(yahmm.UniformDistribution(15.0, 120.0), \
                                            name = 'I_BLIP_' + kmer + '_' + str(index))
            model.add_state(blip_state)
            # Self-loop for blip_staet
            model.add_transition(blip_state, blip_state, blip_self)
            # Transition to blip_state and back
            model.add_transition(current_state, blip_state, blip)
            model.add_transition(blip_state, current_state, 1.0 - blip_self)

            # Short Backslip - can go from 1 to the beginning but favors 1 > ...
            # Starts at state 1 when the first short slip silent state is created
            if index >= 1:
                # Create and add silent state for short slip
                current_short_slip = yahmm.State(None, name = 'B_BACK_SHORT_' + kmer + \
                                                            '_' + str(index))
                model.add_state(current_short_slip)
                # Transition from current state to silent short slip state
                model.add_transition(current_state, current_short_slip, short_slip)
                if index >= 2:
                    # Transition from current silent short slip state to previous 
                    # match state
                    model.add_transition(current_short_slip, states_list[index-1], \
                                            step_back)
                    # Transition from current silent short slip state to previous silent 
                    # short slip state
                    model.add_transition(current_short_slip, previous_short_slip, \
                                            1 - step_back)
                else:
                    model.add_transition(current_short_slip, states_list[index-1], 1.00)

            # Create and Add Skip Silent State
            current_skip = yahmm.State(None, name = 'S_SKIP_' + kmer + '_' + str(index))
            model.add_state(current_skip)
            
            if not previous_skip is None:
                # From previous Skip Silent State to the current Skip Silent State
                model.add_transition(previous_skip, current_skip, long_skip)
                # From previous Skip Silent State to the current match State
                model.add_transition(previous_skip, current_state, 1 - long_skip)

            # From previous match State to the current Skip Silent State
            if index == 0:
                model.add_transition(model.start, current_skip, 1.0 - forward)
            else:
                model.add_transition(states_list[index-1], current_skip, skip)

            # Insert States
            if index > 0:
                # Mean and SD for Insert State
                # Calculated as a mixture distribution
                insert_mean = (previous_mean + current_mean) / 2.0
                insert_sd =  numpy.sqrt(1/4 * ((previous_mean - current_mean) ** 2) \
                                        + 1/2 * (previous_sd ** 2 + current_sd ** 2))
                # Create and Add Insert State
                # Normal Distribution with Mean and Variance that represent 
                # neighboring states                
                insert_state = yahmm.State( yahmm.NormalDistribution(insert_mean, \
                                                insert_sd ), \
                                                name = 'I_INS_' + kmer + '_' + str(index))
                model.add_state(insert_state)
                # Self-loop
                model.add_transition(insert_state, insert_state, ins_self)
                # Transition from states_list[index-1]
                model.add_transition(states_list[index-1], insert_state, insert)
                # Transition to current_state
                model.add_transition(insert_state, current_state, 1.0 - ins_self)

            # Transition to the next match state
            if index == 0:
                # Only transitions from start to skip silent state or first match state
                model.add_transition(model.start, current_state, forward)
            elif index == 1:
                # Since I add match transitions from the previous match state to current
                # match state, I have to make sure the sum of outgoing edges adds to 1.0
                # For index 0, there is no slip, addition of M_0 -> M_1 happens at 1, 
                # which means add this slip probability to the forward transition for M_0
                model.add_transition(states_list[index-1], current_state, forward + slip)
            else:
                model.add_transition(states_list[index-1], current_state, forward)
    
            # Append the current state to states list            
            states_list.append(current_state)

            # Re-assign current states to previous states
            previous_skip = current_skip
            previous_short_slip = current_short_slip if not current_short_slip is None \
                                                        else None
            previous_mean = current_mean
            previous_sd = current_sd

            # Model end case
            if index == len(kmer_list) - 1:
                skip = 0.0
                insert = 0.0
                forward = 1 - (self_loop + end + blip + slip + skip + insert)
                # End cases
                model.add_transition(states_list[index], model.end, forward + end)
                model.add_transition(previous_skip, model.end, 1.00)

        model.bake()
        return model

def model_maker(kmer_current_dict, model_name=None):
    kmer_list = map(str, range(len(kmer_current_dict)))
    model = HMM_Constructor().HMM_linear_model(kmer_list, kmer_current_dict, \
                                                model_name)
    return model

def prediction(model, sequences, sequence_from_hmm, algorithm = 'forward-backward'):

    # Predict sequence from HMM using a user-specified algorithm
    # Forward-Backward (default) or Viterbi

    for i in range(len(sequences)):
        if algorithm == 'viterbi':
            sequence_from_hmm.append(model.viterbi(sequences[i]))
        elif algorithm == 'forward-backward':
            sequence_from_hmm.append(model.forward_backward(sequences[i]))


def plot_event(filename, event, model):
    # Plots, top plot is segmented event colored in cycle by segments, bottom 
    # subplot is segmented event aligned with HMM, colored by states
#    plt.figure(figsize=(20, 8))
    plt.figure()
    plt.subplot(211)
    plt.grid()
    event.plot(color='cycle')
    plt.subplot(212)
    plt.grid()
    event.plot(color='hmm', hmm=model, cmap='Set1')
    plt.show()
#    fig_name = filename + '_' + str(event.start) + '_' + str(event.end) + '.png'
#    plt.tight_layout()
#    plt.savefig(fig_name, format='png')
#    plt.close()
    

########################################################################
# Main
# Here is the main program
########################################################################

def main(myCommandLine=None):

    t0 = time.time()

    if myCommandLine is None:
        myCommandLine = CommandLine()
    else :
        myCommandLine = CommandLine(['-i', '-m'])

    filePath = myCommandLine.args.inpath
    # ^ fast5 file
    modelPath = myCommandLine.args.modelpath
    # ^ tRNA_profiles folder (index/meancurrent/stddev)

    print >> sys.stderr, 'creating kmer current map'
    # CREATE CURRENT MAPS OUT OF MODELPATH
    # kmer_current_dict[index] = [meancurrent, stddev]
    # one dictionary per file where each entry is one line in the file

    #kmer_current_dict_trnaT5 = kmer_current_map(os.path.join(modelPath, 'T5.txt'))
    kmer_current_dict_trnaT6b = kmer_current_map(os.path.join(modelPath, 'T6b.txt'))
    kmer_current_dict_trnaT12b = kmer_current_map(os.path.join(modelPath, 'T12b.txt'))
    #kmer_current_dict_trnaT7 = kmer_current_map(os.path.join(modelPath, 'T7.txt'))
    #kmer_current_dict_trnaT8 = kmer_current_map(os.path.join(modelPath, 'T8.txt'))
    #kmer_current_dict_trnaT12 = kmer_current_map(os.path.join(modelPath, 'T12.txt'))
    #kmer_current_dict_trnaT15 = kmer_current_map(os.path.join(modelPath, 'T15.txt'))
    #kmer_current_dict_trnaT19 = kmer_current_map(os.path.join(modelPath, 'T19.txt'))
    #kmer_current_dict_trnaT21 = kmer_current_map(os.path.join(modelPath, 'T21.txt'))
    #kmer_current_dict_trnaT22 = kmer_current_map(os.path.join(modelPath, 'T22.txt'))
    #kmer_current_dict_trnaT23 = kmer_current_map(os.path.join(modelPath, 'T23.txt'))

    '''
    Construct models: trnaT2, trnaT3, trnaT4
    '''
    # Build one model for each current_dict / filename in modelpath
    # model_maker takes kmer_list, which is just a list of str numbers that are the keys in that dict
    # then model_maker passes the list, dict, name into HMM_linear_model, which gets kmer (just an int),
    # and mean and stddev for each entry in the kmer_dict.
    # Make HMMs for every file and then add them all to a list called models.

    #trnaT5_model = model_maker(kmer_current_dict_trnaT5, model_name = 'T5')
    trnaT6b_model = model_maker(kmer_current_dict_trnaT6b, model_name = 'T6b')
    trnaT12b_model = model_maker(kmer_current_dict_trnaT12b, model_name = 'T12b')
    #trnaT7_model = model_maker(kmer_current_dict_trnaT7, model_name = 'T7')
    #trnaT8_model = model_maker(kmer_current_dict_trnaT8, model_name = 'T8')
    #trnaT12_model = model_maker(kmer_current_dict_trnaT12, model_name = 'T12')
    #trnaT15_model = model_maker(kmer_current_dict_trnaT15, model_name = 'T15')
    #trnaT19_model = model_maker(kmer_current_dict_trnaT19, model_name = 'T19')
    #trnaT21_model = model_maker(kmer_current_dict_trnaT21, model_name = 'T21')
    #trnaT22_model = model_maker(kmer_current_dict_trnaT22, model_name = 'T22')
    #trnaT23_model = model_maker(kmer_current_dict_trnaT23, model_name = 'T23')
    models = [trnaT6b_model, trnaT12b_model]

#    models[0].write(sys.stdout)
    print >> sys.stderr, 'models done'

    # Create blank templates for every model file
    viterbi_prediction = []
    # printing template
    column_output_template = '{0:>} {1:>5}'
    data_output_template = '{0:>d} {1:>5d}'

    #t5 = 0
    t6b = 0
    t12b = 0
    #t7 = 0
    #t8 = 0
    #t12 = 0
    #t15 = 0
    #t19 = 0
    #t21 = 0
    #t22 = 0
    #t23 = 0
    accuracy = 0.0

    matrix = {'T6b':{}, 'T12b':{}}
    #matrix['T5'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    matrix['T6b'] = {0:0, 1:0}
    matrix['T12b'] = {0:0, 1:0}
    #matrix['T7'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    #matrix['T8'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    #matrix['T12'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    #matrix['T15'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    #matrix['T19'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    #matrix['T21'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    #matrix['T22'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    #matrix['T23'] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}

    num_events = 0
    fileCount = 0
    current_run_id = 0
    filesets = []

    # Read in each .fast5 file
    for filename in glob.glob(os.path.join(filePath, '*.fast5')):
        fileCount += 1
        f5File = h5py.File(filename, 'r')

        # Find read number
        readInd = filename.index('read')
        readEnd = filename.index('_', readInd)
        readNum = 'Read_'+str(filename[readInd+4:readEnd])

        # File will have either Raw read or Events
        try:
            read = (f5File['/Raw/Reads/'+readNum])
            signal = (read['Signal']).value
        except:
            read = (f5File['/Analyses/EventDetection_000/Reads/'+readNum])
            signal = (read['Events']).value

        # Find values to be used for converting current to picoAmperes
        # Create current - numpy array of floats in units pA
        uniqueKey = (f5File['/UniqueGlobalKey'])
        digitisation = (uniqueKey['channel_id']).attrs['digitisation']
        offset = (uniqueKey['channel_id']).attrs['offset']
        f5range = (uniqueKey['channel_id']).attrs['range']
        sampling_rate = (uniqueKey['channel_id']).attrs['sampling_rate']

        adjusted_signal = (signal+offset)*(f5range/digitisation)
        current = numpy.array(adjusted_signal, dtype=numpy.float)

        # timestep was fADCSequenceInterval * 1e-3 = .01 for .abf
        # Different for .fast5? standard or make use of sampling interval?
        # This is fed into Segmenter as second = 1000/timestep
        timestep = 0.05
        #timestep = 1/sampling_rate
        # ??? sampling_rate = 3012 so would be 0.000332
        # looks weirdly smooth using that, horizontal axis VERY small
        # real value probably somewhere on the order of .01

        # Because each .fast5 file is an event, group events together in 
        # one fileset object that contains all files with same run_id.
        # For new run_id, new fileset, otherwise add to current fileset.

        current_run_id = (uniqueKey['tracking_id']).attrs['run_id']
        fileset = Fast5FileSet(filename, timestep, current)
        filesets.append(fileset)
        fileset.parse(Segment(current=current, start=0, end=(len(current)-1),
            duration=len(current), second=1000/timestep))


    min_gain_per_sample = 0.15
    sequences = []
    fine_segmentation = None

    # Iterate across filesets and analyze each event by fileset
    for fileset in filesets:
        i = 0
        while i < len(fileset.events):
            event = fileset.events[i]
            filename = fileset.filenames[i]
            timestep = fileset.timesteps[i]
            current = fileset.currents[i]
            second = fileset.seconds[i]

            # Example filename: 
            # MinION205A_20161101_FNFAB42656_MN18771_sequencing_run_MA_1033_R9_4_tRNA_T6b_11_01_16_91388_ch84_read433_strand.fast5
            className = filename[filename.find('_tRNA_'):]
            className = className[className.find('tRNA'):]
            # Example className: tRNA_T6b_11_01_16_91388_ch84_read433_strand.fast5
            fileType = className.split('_')[1]
            # Example fileType: T6b

            ####################################################################
            # NEW TO THIS VERSION:
            # segmenting and prediction are now coupled
            ####################################################################
            #

            sequence_from_hmm = []
            for model in models:

                if model == trnaT6b_model:
                    min_width = 250
                    max_width = 5000000
                    min_gain_per_sample = 0.14
                    window_width = 2000

                    if event.duration > 0:
                        event.filter(order=1, cutoff=2000)
                        event.parse(SpeedyStatSplit(min_width=min_width, max_width=max_width, \
                                            min_gain_per_sample=min_gain_per_sample, \
                                            window_width=window_width))

                        segment_means = []
                        count = 0
                        writeFile = open('F5'+str(min_gain_per_sample)+'/'+str(fileType)+'.txt', 'w')
                        for segment in event.segments:
                            segment_means.append(segment.mean)
                            writeFile.write(str(count)+'\t'+str(segment.mean)+'\t'+str(segment.std)+'\n')
                            count += 1
                        writeFile.close()
                        sequences.append(segment_means)
                        sequences = [segment_means]
                        prediction(model, sequences, sequence_from_hmm, algorithm='viterbi')

                elif model == trnaT12b_model:
                    min_width = 250
                    max_width = 5000000
                    min_gain_per_sample = 0.15
                    window_width = 2000

                    if event.duration > 0:
                        event.filter(order=1, cutoff=2000)
                        event.parse(SpeedyStatSplit(min_width=min_width, max_width=max_width, \
                                            min_gain_per_sample=min_gain_per_sample, \
                                            window_width=window_width))

                        segment_means = []
                        count = 0
                        writeFile = open('F5'+str(min_gain_per_sample)+'/'+str(fileType)+'.txt', 'w')
                        for segment in event.segments:
                            segment_means.append(segment.mean)
                            writeFile.write(str(count)+'\t'+str(segment.mean)+'\t'+str(segment.std)+'\n')
                            count += 1
                        writeFile.close()
                        sequences.append(segment_means)
                        sequences = [segment_means]
                        prediction(model, sequences, sequence_from_hmm, algorithm='viterbi')

            pred = sequence_from_hmm
            scores = [float(pred[0][0]), float(pred[1][0])]
            #
            classified_model = scores.index(max(scores))
            label = fileType
            if classified_model == 0 and label == 'T6b':
                t6b += 1
            if classified_model == 1 and label == 'T12b':
                t12b += 1
            num_events += 1

            matrix[label][classified_model] += 1

            #for k in pred[classified_model][1]:
                #print k[1].name

            # plot event according to model using plot_event:
            # top plot is segmented event colored in cycle by segments, bottom 
            # subplot is segmented event aligned with HMM, colored by states
            #plot_event(filename.strip().split('/')[-1], event, \
            #            model=models[classified_model])

            # iterate through rest of fileset
            i = i + 1


    # now print stuff
    assert (fileCount > 0), 'ERROR: empty directory, no .fast5 files found'
    accuracy = round((t6b + t12b)*100.0/(num_events), 2)
    print >> sys.stdout, 'Alignment results\n'
    print >> sys.stdout, column_output_template.format('T6b', 'T12b')
    print >> sys.stdout, data_output_template.format(t6b, t12b) 
    print >> sys.stdout, '# events', num_events
    print >> sys.stdout, '% accuracy = ', accuracy

    classes = ['T6b', 'T12b']
    for type in classes:
        print >> sys.stdout, type, matrix[type]

    print >> sys.stderr, '\n', 'total time for the program %.3f' % (time.time()-t0)



if (__name__ == '__main__'):
    main()
    raise SystemExit