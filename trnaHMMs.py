#!/usr/bin/env python
# Miten Jain
# trnaHMMs.py
# Uses Adam Novak's yahmm.py and Jacob Schreiber's PyPore

import sys, string, time, math, glob, os
import argparse, itertools, collections
import numpy
import colorsys
import pyximport
pyximport.install(setup_args={'include_dirs':numpy.get_include()})
from yahmm import *
from PyPore.core import Segment
from PyPore.parsers import *
from PyPore.DataTypes import *
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
            self_loop = 0.02
            # End from any state, i.e. reach model.end
            end = 0.02

            # Transitions for Drop-off State
            drop = 0.01
            # Transitions for going to Blip State
            blip = 0.02
            blip_self = 0.05

            # Back Slips, short and long
            slip = 0.05 if index > 0 else 0.00
            # Only short backslips possible
            short_slip = slip
            long_slip = 0.0
            # Transitions from silent slip states
            # Short slip from silent short slip state
            step_back = 0.80

            # Skip that accounts for a missed segment
            skip = 0.01
            # Transitions from current skip silent state to the previous match state or 
            # previous silent skip states
            long_skip = 0.10
            
            # Transitions for Insert state between two neighboring match states
            insert = 0.10 if index > 0 else 0.00
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

def prediction(models, sequences, algorithm = 'forward_backward'):

    # Predict sequence from HMM using a user-specified algorithm
    # Forward-Backward (default) or Viterbi
    sequence_from_hmm = []
    for i in range(len(sequences)):
        for model in models:
            if algorithm == 'viterbi':
                sequence_from_hmm.append(model.viterbi(sequences[i]))
            elif algorithm == 'forward_backward':
                sequence_from_hmm.append(model.forward_backward(sequences[i]))
    return sequence_from_hmm

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
    
# using generator and yield function to read 4 lines at a time
def getStanza (infile):
    while True:
        fasta_id = infile.readline().rstrip()
        fasta_seq = infile.readline().rstrip()
        qual_id = infile.readline().rstrip()
        qual_scr = infile.readline().rstrip()
        if fasta_id != '':
            yield [fasta_id, fasta_seq, qual_id, qual_scr]
        else:
            print >> sys.stderr, 'Warning: End of Sequence'
            break

# read the text files with coordinates and make a hash
def event_hash(filePath):
    class_event_hash = {}
    coordsFile = open(os.path.join(filePath, 'adptTRNA_coords.txt'))
    print >> sys.stderr, 'Reading coordinates from ', os.path.join(filePath, \
                                                    'adptTRNA_coords.txt')
    for stanza in getStanza(coordsFile):
        start_name = stanza[0]
        start_times = map(float, stanza[1].strip().split(','))
        end_name = stanza[2]
        end_times = map(float, stanza[3].strip().split(','))
        filename = start_name.strip().split('>')[-1].split('_start')[0] + '.abf'
        fileClass = filename.strip().split('_')[0]
        filenameKey = filename.strip().split('_')[1]
        if not filenameKey in class_event_hash.keys():
            class_event_hash[filenameKey] = {'type':None, 'start_times':[], 'end_times':[]}
        class_event_hash[filenameKey]['type'] = fileClass
        for start, end in zip(start_times, end_times):
            if start / 1000 >= 1:
                start = start / 1000.0
                end = end / 1000.0
                                
            class_event_hash[filenameKey]['start_times'].append(start)
            class_event_hash[filenameKey]['end_times'].append(end)

    coordsFile.close()

    return class_event_hash

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
    modelPath = myCommandLine.args.modelpath

    print >> sys.stderr, 'creating kmer current map'
    kmer_current_dict_trnaT5 = kmer_current_map(os.path.join(modelPath, \
                                                             'trnaT5_current_map.txt'))
    kmer_current_dict_trnaT6 = kmer_current_map(os.path.join(modelPath, \
                                                             'trnaT6_current_map.txt'))
    kmer_current_dict_trnaT7 = kmer_current_map(os.path.join(modelPath, \
                                                             'trnaT7_current_map.txt'))
    kmer_current_dict_trnaT8 = kmer_current_map(os.path.join(modelPath, \
                                                             'trnaT8_current_map.txt'))
    kmer_current_dict_trnaT22 = kmer_current_map(os.path.join(modelPath, \
                                                             'trnaT22_current_map.txt'))
    kmer_current_dict_adapter_bg = kmer_current_map(os.path.join(modelPath, \
                                                             'adapter_bg_current_map.txt'))

    '''
    Construct models: trnaT2, trnaT3, trnaT4
    '''
    trnaT5_model = model_maker( kmer_current_dict_trnaT5, model_name = 'trnaT5' )
    trnaT6_model = model_maker( kmer_current_dict_trnaT6, model_name = 'trnaT6' )
    trnaT7_model = model_maker( kmer_current_dict_trnaT7, model_name = 'trnaT7' )
    trnaT8_model = model_maker( kmer_current_dict_trnaT8, model_name = 'trnaT8' )
    trnaT22_model = model_maker( kmer_current_dict_trnaT22, model_name = 'trnaT22' )
    adapter_bg_model = model_maker( kmer_current_dict_adapter_bg, \
                                                       model_name = 'adapter_bg' )
    models = [    trnaT5_model, trnaT6_model, trnaT7_model, trnaT8_model, \
                                                trnaT22_model, adapter_bg_model ]

#    models[0].write(sys.stdout)
    print >> sys.stderr, 'models done'

    viterbi_prediction = []
    # printing template
    column_output_template = '{0:>} {1:>25} {2:>15} {3:>15} {4:>15} {5:>15}'
    data_output_template = '{0:>} {1:>20.2f} {2:>15.2f} {3:>15.2f} {4:>15.2f} {5:>15.2f}'
#    print >> sys.stdout, column_output_template.format('file', 'start (s)', 'end (s)', \
#                                                        'trnaT2', 'trnaT3', 'tRNAT4')

    t5 = 0
    t6 = 0
    t7 = 0
    t8 = 0
    
    num_events = 0
    fileCount = 0
    class_event_hash = event_hash(filePath)
    for filename in glob.glob(os.path.join(filePath, '*.abf')):
        fileCount += 1
        filenamekey = filename.strip().split('/')[-1]
        fileType = class_event_hash[filenamekey]['type']
        label = fileType.split('_')[0]
        start_list, end_list = class_event_hash[filenamekey]['start_times'], \
                                class_event_hash[filenamekey]['end_times']
        if not len(start_list) > 0 and len(end_list) > 0:
            continue
        print >> sys.stderr, 'Reading file ', filenamekey, ' of class ', fileType, \
                            ' with # events = ', len(start_list)

        file = File(filename)
        event_startpoints_list = [i * 100000 for i in start_list]
        event_endpoints_list = [i * 100000 for i in end_list]

        file.parse(MemoryParse(event_startpoints_list, event_endpoints_list))
        min_gain_per_sample = 0.25

        sequences = []
        fine_segmentation = None
        for event in file.events:
            if event.duration > 0:
                event.filter(order=1, cutoff=2000)
                event.parse(SpeedyStatSplit(min_width=250, max_width=5000000, \
                                            min_gain_per_sample=min_gain_per_sample, \
                                            window_width=2000))

            segment_means = []
            count = 0
            for segment in event.segments:
                segment_means.append(segment.mean)
#                if filenamekey == '16427002-s06.abf':
#                    print count, '\t', segment.mean, '\t', segment.std
                count += 1
            sequences.append(segment_means)
            sequences = [segment_means]

            # Align event to HMM
            pred = prediction (models, sequences, algorithm = 'viterbi')
            scores = [    float(pred[0][0]), float(pred[1][0]), float(pred[2][0]), \
                        float(pred[3][0])	] 

#            print fileType, event.start, event.end, scores

            classified_model = scores.index(max(scores))
            if classified_model == 0 and label == 'T5':
                t5 += 1
            if classified_model == 1 and label == 'T6':
                t6 += 1
            if classified_model == 2 and label == 'T7':
                t7 += 1
            if classified_model == 3 and label == 'T8':
                t8 += 1
            num_events += 1

            # plot event according to model
#            plot_event(filename.strip().split('/')[-1], event, \
#                        model=models[classified_model])
    assert (fileCount > 0), "ERROR: empty directory, no ABF files found"
            
    print num_events, t5, t6, t7, t8
    print 'Accuracy = ', round((t5+t6+t7+t8)*100.0/num_events, 2), ' %'
    print >> sys.stderr, '\n', 'total time for the program %.3f' % (time.time()-t0)

if (__name__ == '__main__'):
    main()
    raise SystemExit
