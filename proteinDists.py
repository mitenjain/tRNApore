#Protein Distributions
from yahmm import *
from PyPore.hmm import *

'''
Board functions for nanopore modeling contained in hmm.py.

ProteinDomainBoard and ProteinDomainBoard2 have very similar transition
probabilities but ProteinDomainBoard2 allows for backslips.

Currently being used with tRNAProfileModel where a list of board
functions is passed into the models along with the list of profile means/std
of the different tRNA domains being modeled.
'''


def tRNABoard5( distribution, name, insert ): #rewrite
    """
    Used to model whole 5' marker
    """
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    backslip = State( None, name = "B:{}".format(name))


    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)
    
    #add backslip transitions
    board.add_transition( backslip, match, 0.85) #backslip to prev
    board.add_transition( backslip, board.s3, 0.15)

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.001 )
    board.add_transition( insert, insert, 0.349 )
    #board.add_transition( insert, board.e1, 0.001 )
    board.add_transition( insert, board.e2, 0.6 ) #insert to next match

    #match transitions
    board.add_transition( match, insert, 0.01 )
    #board.add_transition( match, board.e1, 0.001 ) #match to next delete
    board.add_transition( match, board.e2, 0.6 )  #match to next match
    board.add_transition( match, match, 0.39) #match loop
    #board.add_transition( match, board.s3, 0.01)    #match to backslip

    return board


def polyDTboard( distribution, name, insert ): #rewrite
    """
    Used to model abasic jump ~100 pA
    """
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    blip = State( UniformDistribution(15.0, 120.0), name = "Bl:{}".format(name))
    backslip = State( None, name = "B:{}".format(name))
    
    
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)
    
    #add backslip transitions
    board.add_transition( backslip, match, 0.95) #backslip to prev
    #board.add_transition( backslip, board.s3, 0.05)

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.001 )
    #board.add_transition( insert, blip, 0.349 )
    #board.add_transition( insert, board.e1, 0.001 )
    board.add_transition( insert, board.e2, 0.999 ) #insert to next match
    #board.add_transition( insert, board.s3, .01)

    #match transitions
    #board.add_transition( match, insert, 0.1 ) #.15
    board.add_transition( match, board.e1, 0.05 ) #match to next delete
    board.add_transition( match, board.e2, 0.45)  #match to next match #.55
    board.add_transition( match, match, 0.2 ) #match loop #.3
    board.add_transition( match, blip, 0.05)
    board.add_transition( match, board.s3, 0.25)    #match to backslip
    
    #blip transitions
    board.add_transition(blip, blip, .02)
    board.add_transition(blip, match, .98)
    
    return board

    
def intermediateboard( distribution, name, insert ): #rewrite
    """
    Used to model intermdetiate jump in polyDT section (model 2)
    """
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    blip = State( UniformDistribution(15.0, 120.0), name = "Bl:{}".format(name))
    backslip = State( None, name = "B:{}".format(name))
    
    
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)
    
    #add backslip transitions
    board.add_transition( backslip, match, 0.95) #backslip to prev
    #board.add_transition( backslip, board.s3, 0.05)

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.001 )
    #board.add_transition( insert, blip, 0.349 )
    #board.add_transition( insert, board.e1, 0.001 )
    board.add_transition( insert, board.e2, 0.999 ) #insert to next match
    #board.add_transition( insert, board.s3, .01)

    #match transitions
    board.add_transition( match, insert, 0.1 ) #.15
    board.add_transition( match, board.e1, 0.05 ) #match to next delete
    board.add_transition( match, board.e2, 0.4 )  #match to next match #.55
    board.add_transition( match, match, 0.3 ) #match loop #.3
    board.add_transition( match, blip, 0.05)
    board.add_transition( match, board.s3, 0.1)    #match to backslip
    
    #blip transitions
    board.add_transition(blip, blip, .02)
    board.add_transition(blip, match, .98)
    
    return board

def intermediateboard2( distribution, name, insert ): #rewrite
    """
    Used to model second polyDT low current before abasic jump (model 2)
    """
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']


    #use uniform insert instead of the middle of two means
    insert = UniformDistribution(40., 95.)


    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    blip = State( UniformDistribution(15.0, 120.0), name = "Bl:{}".format(name))
    backslip = State( None, name = "B:{}".format(name))
    
    
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)
    
    #add backslip transitions
    board.add_transition( backslip, match, 0.95) #backslip to prev
    #board.add_transition( backslip, board.s3, 0.05)

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.001 )
    #board.add_transition( insert, blip, 0.349 )
    #board.add_transition( insert, board.e1, 0.001 )
    board.add_transition(insert, insert, 0.499) #399
    board.add_transition( insert, board.e2, 0.5 ) #insert to next match #.6
    #board.add_transition( insert, board.s3, .01)

    #match transitions
    board.add_transition( match, insert, 0.35 )
    board.add_transition( match, board.e1, 0.05 ) #match to next delete
    board.add_transition( match, board.e2, 0.2 )  #match to next match
    board.add_transition( match, match, 0.3 ) #match loop
    board.add_transition( match, blip, 0.05)
    board.add_transition( match, board.s3, 0.05)    #match to backslip
    
    #blip transitions
    board.add_transition(blip, blip, .02)
    board.add_transition(blip, match, .98)
    
    return board




def abasictRNABoard( distribution, name, insert ): #rewrite
    """
    Used to model abasic jump ~100 pA
    """
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    blip = State( UniformDistribution(15.0, 120.0), name = "Bl:{}".format(name))
    backslip = State( None, name = "B:{}".format(name))
    
    
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)
    
    #add backslip transitions
    board.add_transition( backslip, match, 0.95) #backslip to prev
    #board.add_transition( backslip, board.s3, 0.05)

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.001 )
    #board.add_transition( insert, blip, 0.349 )
    #board.add_transition( insert, board.e1, 0.001 )
    board.add_transition( insert, board.e2, 0.999 ) #insert to next match
    #board.add_transition( insert, board.s3, .01)

    #match transitions
    board.add_transition( match, insert, 0.09 ) #.15
    board.add_transition( match, board.e1, 0.01 ) #match to next delete
    board.add_transition( match, board.e2, 0.5 )  #match to next match #.55
    board.add_transition( match, match, 0.4 ) #match loop #.3
    #board.add_transition( match, blip, 0.05)
    #board.add_transition( match, board.s3, 0.05)    #match to backslip
    
    #blip transitions
    board.add_transition(blip, blip, .02)
    board.add_transition(blip, match, .98)
    
    return board



def abasictRNABoard2( distribution, name, insert ): #rewrite
    """
    Used to model ratchetting state after abasic jump ~100 pA
    """
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']
    
    insert = UniformDistribution(40., 100.)

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    blip = State( UniformDistribution(15.0, 120.0), name = "Bl:{}".format(name))
    backslip = State( None, name = "B:{}".format(name))
    
    
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)
    
    #add backslip transitions
    board.add_transition( backslip, match, 0.95) #backslip to prev
    board.add_transition( backslip, board.s3, 0.05)

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.001 )
    #board.add_transition( insert, blip, 0.349 )
    #board.add_transition( insert, board.e1, 0.001 )
    board.add_transition( insert, board.e2, 0.999 ) #insert to next match
    board.add_transition( insert, board.s3, .01)

    #match transitions
    board.add_transition( match, insert, 0.02 ) #.15
    board.add_transition( match, board.e1, 0.02 ) #match to next delete
    board.add_transition( match, board.e2, 0.4 )  #match to next match #.55
    board.add_transition( match, match, 0.2 ) #match loop #.3
    board.add_transition( match, blip, 0.01)
    board.add_transition( match, board.s3, 0.2)    #match to backslip
    
    #blip transitions
    board.add_transition(blip, blip, .02)
    board.add_transition(blip, match, .98)
    
    return board



def tRNABoard( distribution, name, insert ):
    #def tRNABoard( distribution, name, insert=UniformDistribution( 40, 100 ) ): #rewrite
    """
    Used to model area between abasic and tRNA where it may be very noisy and varied
    """
    
    
    #insert = UniformDistribution(15, 120)
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    blip = State( UniformDistribution(15.0, 120.0), name = "Bl:{}".format(name))
    backslip = State(None, name = "B:{}".format(name))


    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 )
    
    #add backslip transitions
    board.add_transition( backslip, match, 0.85) #backslip to prev match
    board.add_transition( backslip, board.s3, 0.15)
    
    #blip transitions
    board.add_transition(blip, blip, .02)
    board.add_transition(blip, match, .98)
    
    #insert transitions
    board.add_transition( insert, match, 0.40 ) #insert to current match
    board.add_transition( insert, insert, 0.25 )
    #board.add_transition( insert, board.e1, 0.001 )
    board.add_transition( insert, board.e2, 0.35 )

    #match transitions
    board.add_transition( match, insert, 0.01 ) #.015
    board.add_transition( match, board.e1, 0.01 ) #match to next delete #0
    board.add_transition( match, board.e2, 0.4)  #match to next match .7
    board.add_transition( match, match, 0.15 ) #match loop .15
    board.add_transition( match, blip, 0.05)
    board.add_transition( match, board.s3, 0.38)  #match to backslip .1
    
    return board


def tRNABoard1( distribution, name, insert ): #rewrite
    """
    Used to model fine segments of tRNA
    """
    board = HMMBoard(n=2, name=str(name))
    board.directions = ['>', '>']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )


    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )

    #delete transisitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.4 )
    board.add_transition( insert, insert, 0.25 )
    #board.add_transition( insert, board.e1, 0.001 ) #insert to delete
    board.add_transition( insert, board.e2, 0.35 )

    #match transitions
    board.add_transition( match, insert, 0.2 ) #.2
    board.add_transition( match, board.e1, 0.01 ) #match to next delete .001
    board.add_transition( match, board.e2, 0.74 )  #match to next match .839
    board.add_transition( match, match, 0.05 ) #match loop
    
    return board



########################################################
#                                                      #
# Protein boards left for reference, not used for tRNA #
#                                                      #
########################################################



def ProteinDomainBoard( distribution, name, insert=UniformDistribution( 0, 90 ) ):
    """
    The current board being used to model each state of the protein HMM.
    
    This board is very simplistic and only models insertions, matches, and 
    deletions with modifications planned to allow for modeling backslips.
    
    The idea with this is to build a base board type which generally models
    protein data, and then make modified versions to model each unique region
    of the trace. This is intended for use with the ModularDomainProfileModel
    which allows multiple profiles and board functions to be passed to it.
    Author: jakob.houser@gmail.com
    """
    
    board = HMMBoard(n=2, name=str(name))
    board.directions = ['>', '>']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )

    #transitions between states
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )

    #deletion transitions
    board.add_transition( delete, board.e1, 0.001 ) #delete to next delete
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 ) #delete to next match
    
    #insert transitions
    board.add_transition( insert, match, 0.40 )
    board.add_transition( insert, insert, 0.25 )
    board.add_transition( insert, board.e1, 0.001 ) #insert to next delete
    board.add_transition( insert, board.e2, 0.349 ) #insert to next match

    #match transitions
    board.add_transition( match, insert, 0.01 )
    board.add_transition( match, board.e1, 0.01 ) #match to next delete
    board.add_transition( match, board.e2, 0.80 )  #match to next match
    board.add_transition( match, match, 0.18 ) #match loop
    
    return board


def ProteinDomainBoard2( distribution, name, insert=UniformDistribution( 0, 90 ) ): #rewrite
    """
    The current board being used to model each state of the protein HMM.
    
    This board is very simplistic and only models insertions, matches, and 
    deletions with modifications planned to allow for modeling backslips.
    
    The idea with this is to build a base board type which generally models
    protein data, and then make modified versions to model each unique region
    of the trace. This is intended for use with the ModularDomainProfileModel
    which allows multiple profiles and board functions to be passed to it.
    Author: jakob.houser@gmail.com edited by Hannah Meyers
    """
    
    board = HMMBoard(n=3, name=str(name))
    board.directions = ['>', '>', '<']

    delete = State( None, name="D:{}".format( name ) )
    match = State( distribution, name="M:{}".format( name ) )
    insert = State( insert, name="I:{}".format( name ) )
    backslip = State( None, name = "B:{}".format(name))


    #add transitions between these states
    board.add_transition( board.s1, delete, 1.00 )
    board.add_transition( board.s2, match, 1.00 )
    board.add_transition( board.e3, backslip, 1.00)

    #add backslip transitions
    board.add_transition( backslip, match, 0.85) #backslip to prev
    board.add_transition( backslip, board.s3, 0.15)
    
    #add deletion transitions
    board.add_transition( delete, board.e1, 0.001 )
    board.add_transition( delete, insert, 0.001 )
    board.add_transition( delete, board.e2, 0.998 )

    #insert transitions
    board.add_transition( insert, match, 0.40 )
    board.add_transition( insert, insert, 0.25 )
    board.add_transition( insert, board.e1, 0.001 ) #insert to next delete
    board.add_transition( insert, board.e2, 0.349 ) #insert to next match

    #match transitions
    board.add_transition( match, insert, 0.01 )
    board.add_transition( match, match, 0.17 )
    board.add_transition( match, board.e1, 0.01 )   #match to next delete
    board.add_transition( match, board.e2, 0.80 )   #match to next match
    board.add_transition( match, board.s3, 0.01)    #match to backslip
    
    return board



