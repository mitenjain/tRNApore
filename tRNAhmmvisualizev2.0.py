'''
Author: Hannah Meyers

This file contains the experiment code for attempting to model
protein nanopore traces via HMMs. Please see inline comments
for an explanation of what each piece of the code is doing.
'''
from __future__ import print_function

from PyPore.parsers import *
from PyPore.DataTypes import *
from hmm import *
from yahmm import *

import math
import matplotlib.pyplot as plt
import itertools as it
import glob
import seaborn as sns
import sys
import pandas as pd
from proteinDistsv7 import *
from scipy.stats import kde

import sys



#Experiment data files. The first set before the break are all experiment files from
#the same day of data collection. Files after the break are each from different days.
filenames = [

#'trnaAbfFiles/t2_15n30003-s06.abf',
#'trnaAbfFiles/t3_15d03003-s06.abf',
#'trnaAbfFiles/t3_15d03004-s06.abf'
#'trnaAbfFiles/t3_15d08002-s06.abf'
#'trnaAbfFiles/t4_15d07002-s06.abf'

#'trnaAbfFiles/t5_16325002-s06.abf'
#'trnaAbfFiles/t5_16325003-s06.abf', #28 events
#'trnaAbfFiles/t5_16325004-s06.abf'     #36 events ~4 real

#'trnaAbfFiles/t6_16322002-s06.abf', #31
#'trnaAbfFiles/t6_16322003-s06.abf'
#'trnaAbfFiles/t6_16322004-s06.abf'

#'trnaAbfFiles/t7_16323002-s06.abf' #25 events
#'trnaAbfFiles/t7_16323003-s06.abf'
#'trnaAbfFiles/t7_16323004-s06.abf'

'trnaAbfFiles/t8_16326002-s06.abf' #29
#'trnaAbfFiles/t8_16328002-s06.abf'
#'trnaAbfFiles/t8_16404002-s06.abf'

]

#Inserts are uniform across the range of current we expect to see in an event
insert1 = UniformDistribution(15., 120.)


###MODEL 1####
#beginning of molecule to first PolyDT segment
#Profile for first model, overextends to
profile = [NormalDistribution(54, 20),
#looking for this match:
NormalDistribution(36.0, 2.),
NormalDistribution(50., 6.),
NormalDistribution(31., 2.),
NormalDistribution(98., 2.5)]


#list of board functions corresponds to model 1
boardlist = [tRNABoard5]+ [polyDTboard] + [abasictRNABoard]+([abasictRNABoard2]*2)

#build first model
model = tRNAProfileModel( boardlist, profile, "ClpXProfile-{}".format( len(profile) ), insert1)



###MODEL 2####
#polyDT to first segment of tRNA
profile2 = [
NormalDistribution(36., 2.), NormalDistribution(50., 6.), NormalDistribution(31., 2.),
#{'high': NormalDistribution(100., 2.5), 'low': NormalDistribution(90, 2.5)},
NormalDistribution(96.0, 12.0),  NormalDistribution(48., 5.), NormalDistribution(36.0, 2.)]

boardlist2 = [polyDTboard, intermediateboard, intermediateboard2, abasictRNABoard, tRNABoard, tRNABoard1]
model2 = tRNAProfileModel(boardlist2, profile2, "ClpXProfile-{}".format( len(profile2) ), insert1)


y = 0

#iteration for applying model to events in filenames list and plotting
for file in it.imap( File, filenames ):
    x = 0
    
    print(file.filename)
    #Events must drop below this threshold
    threshold = 130
    rules = [lambda event: event.min > -5,
             lambda event: event.duration > 500000,
             lambda event: event.max < threshold]
    
    file.parse( lambda_event_parser( threshold=threshold, rules = rules ) )
    
    for i, event in enumerate(file.events):
    
        event.filter(order=1, cutoff=2000)
        
        #keep track of number of events
        x+=1
        y+=1
        #false_positive_rate controls the number of segments that will be created by the segmenter
        event.parse( SpeedyStatSplit( min_width=250, max_width=5000000, window_width=2000, min_gain_per_sample = 0.25))
        
        
        #Apply first model to event
        score1, hidden_states = model.viterbi([segment.mean for segment in event.segments])
        
        
        print('Number of Events', x)
        print(event.start)
        print(event.end)
        print('Score 1: ', score1)
        
        #values to find end of first model
        offset = 0
        path = hidden_states[1:-1]
        segIndex = 0
        indexing = 0
    
    
        #loop through states and segments to find index of the segment we are looking for - the end of the first model
        for i, state in enumerate( path ):
            #prints junctions between boards, want to ignore this in finding the segmenting index
            if state[1].name[0] == 'b':
                offset += 1
            #if we found the state we are looking for, stop looking and save the position
            #note: this will return the first segment that matches, may be others but model 2 will account for this
            elif state[1].name == 'M:1':
                segIndex = i
                break
            #just in case the model ends early, we still want to save the position of the last match
            #this will only save the index if M:1 is NOT found
            elif state[1].name[0] == 'M':
                #offset+=1
                segIndex = i-offset
                #print('match end found')

        #if reach the end of the model pass
        if segIndex >= len(event.segments):
            segIndex = len(event.segments) - 1
    
        
        #change event list to only include second model
        event.segments = event.segments[segIndex:]

        
        #model just the trna
        score2, hidden_states = model2.viterbi([segment.mean for segment in event.segments])
        
        print('Score 2:', score2)
        #print('\n')
        
        if hidden_states != None:
        
            #First subplot is event + segmentation
            plt.figure( figsize=(15, 8))
            plt.subplot( 311 )
            event.plot( color='cycle' )
        
            #Second subplot is event + HMM
            plt.subplot( 312 )
            event.plot( color='hmm', hmm=model2, hidden_states=hidden_states, cmap='Set1' )

            #Final subplot is color cycle with profile means
            #this subplot is currently inaccurate as it only plots the first profile
            #furthermore, there was a bug in PyPore when I started on this that makes the color cycle
            #not match up to the HMM colors. I am unsure if the bug has been fixed since then.
            ax = plt.subplot( 313 )
            plt.imshow( [ np.arange( 0., len(profile2) ) / len(profile2) ], interpolation='nearest', cmap="Set1" )
            plt.grid( False )
            #means = [ d.parameters[0].parameters[0] for d in profile ]
            means = [35,50,31, 100]
            #print(d.parameters[0].parameters[0] for d in profile)

            for i, mean in enumerate( means ):
                plt.text( i-0.2, 0.1, str( round(mean, 1) ), fontsize=12 )
        
        #Output HMM state path to output.txt file
        outputtext = 'output' + str(y) + '.txt'
        f = open(outputtext, 'w')
        
        for i, state in enumerate(hidden_states):
            f.write(state[1].name+"\n")
            #f.write(str(state[1]))

        s =  file.filename + str(x)
        #save figure with name s + counter to prevent name duplications
        #plt.savefig(s)
        #x += 1
        
        
        #rmlist = ['b1e2', 'b2e2']
        #for item in rmlist:
        #    hidden_states.remove(item)
        
        #print ("Path: {}".format( ' '.join( state.name for i, state in hidden_states[1:-1] ) )    )
            
        f.write('{}'.format([(segment.mean, segment.std) for segment in event.segments]))
        
        f.close()
        
    
        #s = file.filename[16:] +'fp55s' + str(x)

        
        
        #these loops show all plots up till x ==
        if x ==28:
            plt.show()
        elif x < 29:
            pass
        else:
            break
    file.close()


print('number of events: ', x)

