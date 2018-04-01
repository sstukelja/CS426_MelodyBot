import numpy as np
import os
import subprocess
from subprocess import Popen, PIPE
import sys
import itertools
import string

modelPath = ""
samplePath = ""
chars = []
vocab_size = 0
seq_length = 0
hidden_size = 0
char_to_ix = { }
ix_to_char = { }
learning_rate = 1e-1

sampleDirectory = "active_samples\\"
default = True
instrument = "0"
tempo = "120"

def union(*dicts):
    return dict(itertools.chain.from_iterable(dct.items() for dct in dicts))

#parse command-line arguments
if(len(sys.argv) > 1 ):
    default = False
    model = sys.argv[1]
    instrument = str(sys.argv[2])
    tempo = str(sys.argv[3])
    seed = int(sys.argv[4])
    np.random.seed(seed)
else:
    model = ""

#clear the content of text files
open('longDatax.txt', 'w').close()
open('shortDatax.txt', 'w').close()
open('shortDataFresh.txt', 'w').close()

#model = "jimiHendrixGuitar"
#tempo = "120"
#default = False
#seed = 100
#instrument = "30"
#np.random.seed(seed)

#read from selected model
if model == "classicalViolin":
    modelPath = "master_models\\" + "classicalViolinMaster.txt"
    choice = 1
    dataFile = "shortDatas\\" + "SDclassicalViolin.txt"
    specialChar = 'g'
    #header for csv file (violin)

    #default
    if default == True:
        header = "0, 0, Header, 1, 17, 480\n1, 0, Start_track\n1, 0, Title_t, \"untitled\"\n1, 0, Copyright_t, \"Copyright © 1996 by David J. Grossman\"\n1, 0, Text_t, \"David J. Grossman\"\n1, 0, SMPTE_offset, 96, 0, 3, 0, 0\n1, 0, Time_signature, 3, 2, 24, 8\n1, 0, Key_signature, 2, \"major\"\n1, 0, Tempo, 240000\n1, 1200, Tempo, 461538\n1, 1440, Marker_t, \"A\"\n1, 47520, Marker_t, \"A'\"\n1, 93600, Marker_t, \"B\"\n1, 162720, Marker_t, \"B'\"\n1, 230400, Tempo, 923077\n1, 230400, End_track\n2, 0, Start_track\n2, 0, MIDI_port, 0\n2, 0, Title_t, \"Solo Violin\"\n2, 0, Program_c, 0, 40\n2, 0, Control_c, 0, 7, 100\n2, 0, Control_c, 0, 10, 64\n"
    else:
        header = "0, 0, Header, 1, 17, " + tempo + "\n1, 0, Start_track\n1, 0, Title_t, \"untitled\"\n1, 0, Copyright_t, \"Copyright © 1996 by David J. Grossman\"\n1, 0, Text_t, \"David J. Grossman\"\n1, 0, SMPTE_offset, 96, 0, 3, 0, 0\n1, 0, Time_signature, 3, 2, 24, 8\n1, 0, Key_signature, 2, \"major\"\n1, 0, Tempo, 240000\n1, 1200, Tempo, 461538\n1, 1440, Marker_t, \"A\"\n1, 47520, Marker_t, \"A'\"\n1, 93600, Marker_t, \"B\"\n1, 162720, Marker_t, \"B'\"\n1, 230400, Tempo, 923077\n1, 230400, End_track\n2, 0, Start_track\n2, 0, MIDI_port, 0\n2, 0, Title_t, \"Solo Violin\"\n2, 0, Program_c, 0, " + instrument +"\n2, 0, Control_c, 0, 7, 100\n2, 0, Control_c, 0, 10, 64\n"
elif model == "jazzPiano":
    modelPath = "master_models\\" + "jazzPianoMaster.txt"
    choice = 2
    specialChar = '$'
    dataFile = "shortDatas\\" + "SDjazzPiano.txt"
    #header for jazz piano

    if default == True:
        #header = "0, 0, Header, 0, 1, 120\n1, 0, Start_track\n1, 230400, End_track\n2, 0, Start_track\n2, 0, Time_signature, 4, 2, 24, 8\n2, 0, Key_signature, 0, \"major\"\n2, 0, Tempo, 681818\n2, 0, Program_c, 0, 0\n2, 0, Program_c, 1, 0\n"
        header = "0, 0, Header, 1, 2, 120\n1, 0, Start_track\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 681818\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, 0\n2, 0, Tempo, 681818\n"
    else:
        #header = "0, 0, Header, 0, 1, 120\n1, 0, Start_track\n1, 230400, End_track\n2, 0, Start_track\n2, 0, Time_signature, 4, 2, 24, 8\n2, 0, Key_signature, 0, \"major\"\n2, 0, Tempo, 681818\n2, 0, Program_c, 0, " + instrument + "\n2, 0, Program_c, 1, 0\n"
        header = "0, 0, Header, 1, 2, " + tempo + "\n1, 0, Start_track\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 681818\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, " + instrument + "\n2, 0, Tempo, 681818\n"
elif model == "mozartPiano":
    modelPath = "master_models\\" + "mozartPianoMaster.txt"
    choice = 4
    specialChar = '~'
    dataFile = "shortDatas\\" + "SDmozartPiano.txt"
    #header for mozart piano
    if default == True:
        header = "0, 0, Header, 0, 1, 120\n1, 0, Start_track\n1, 230400, End_track\n2, 0, Start_track\n2, 0, Title_t, \"A House Is Not A Home\"\n2, 0, Text_t, \"Dougmck\\012\"\n2, 0, Time_signature, 4, 2, 24, 8\n2, 0, Key_signature, 0, \"major\"\n2, 0, Tempo, 1000000\n2, 0, Program_c, 0, 0\n2, 0, Program_c, 1, 0\n"
    else:
        header = "0, 0, Header, 0, 1, " + tempo + "\n1, 0, Start_track\n1, 230400, End_track\n2, 0, Start_track\n2, 0, Title_t, \"A House Is Not A Home\"\n2, 0, Text_t, \"Dougmck\\012\"\n2, 0, Time_signature, 4, 2, 24, 8\n2, 0, Key_signature, 0, \"major\"\n2, 0, Tempo, 1000000\n2, 0, Program_c, 0, " + instrument + "\n2, 0, Program_c, 1, 0\n"
elif model == "jimiHendrixGuitar":
    modelPath = "master_models\\" + "jimiHendrixGuitarMaster.txt"
    choice = 5
    specialChar = 'Y'
    dataFile = "shortDatas\\" + "SDjimiHendrixGuitar.txt"
    #header for jimiHendrixGuitar
    if default == True:
        header = "0, 0, Header, 1, 2, 960\n1, 0, Start_track\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 722891\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, 30\n2, 0, Tempo, 722891\n"
    else:
        header = "0, 0, Header, 1, 2, " + tempo + "\n1, 0, Start_track\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 722891\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, " + instrument + "\n2, 0, Tempo, 722891\n"
else:
    modelPath = "master_models\\" + "bluesGuitarMaster.txt"
    choice = 3
    dataFile = "shortDatas\\" + "SDbluesGuitar.txt"
    specialChar = '3'
    #header for blues/spanish guitar
    if default == True:
        header = "0, 0, Header, 1, 2, 960\n1, 0, Start_track\n1, 0, Control_c, 0, 101, 0\n1, 0, Control_c, 0, 100, 0\n1, 0, Control_c, 0, 6, 12\n1, 0, Control_c, 0, 38, 0\n1, 0, Control_c, 1, 101, 0\n1, 0, Control_c, 1, 100, 0\n1, 0, Control_c, 1, 6, 12\n1, 0, Control_c, 1, 38, 0\n1, 0, Control_c, 2, 101, 0\n1, 0, Control_c, 2, 100, 0\n1, 0, Control_c, 2, 6, 12\n1, 0, Control_c, 2, 38, 0\n1, 0, Control_c, 3, 101, 0\n1, 0, Control_c, 3, 100, 0\n1, 0, Control_c, 3, 6, 12\n1, 0, Control_c, 3, 38, 0\n1, 0, Control_c, 4, 101, 0\n1, 0, Control_c, 4, 100, 0\n1, 0, Control_c, 4, 6, 12\n1, 0, Control_c, 4, 38, 0\n1, 0, Control_c, 5, 101, 0\n1, 0, Control_c, 5, 100, 0\n1, 0, Control_c, 5, 6, 12\n1, 0, Control_c, 5, 38, 0\n1, 0, Control_c, 6, 101, 0\n1, 0, Control_c, 6, 100, 0\n1, 0, Control_c, 6, 6, 12\n1, 0, Control_c, 6, 38, 0\n1, 0, Control_c, 7, 101, 0\n1, 0, Control_c, 7, 100, 0\n1, 0, Control_c, 7, 6, 12\n1, 0, Control_c, 7, 38, 0\n1, 0, Control_c, 8, 101, 0\n1, 0, Control_c, 8, 100, 0\n1, 0, Control_c, 8, 6, 12\n1, 0, Control_c, 8, 38, 0\n1, 0, Control_c, 9, 101, 0\n1, 0, Control_c, 9, 100, 0\n1, 0, Control_c, 9, 6, 12\n1, 0, Control_c, 9, 38, 0\n1, 0, Control_c, 10, 101, 0\n1, 0, Control_c, 10, 100, 0\n1, 0, Control_c, 10, 6, 12\n1, 0, Control_c, 10, 38, 0\n1, 0, Control_c, 11, 101, 0\n1, 0, Control_c, 11, 100, 0\n1, 0, Control_c, 11, 6, 12\n1, 0, Control_c, 11, 38, 0\n1, 0, Control_c, 12, 101, 0\n1, 0, Control_c, 12, 100, 0\n1, 0, Control_c, 12, 6, 12\n1, 0, Control_c, 12, 38, 0\n1, 0, Control_c, 13, 101, 0\n1, 0, Control_c, 13, 100, 0\n1, 0, Control_c, 13, 6, 12\n1, 0, Control_c, 13, 38, 0\n1, 0, Control_c, 14, 101, 0\n1, 0, Control_c, 14, 100, 0\n1, 0, Control_c, 14, 6, 12\n1, 0, Control_c, 14, 38, 0\n1, 0, Control_c, 15, 101, 0\n1, 0, Control_c, 15, 100, 0\n1, 0, Control_c, 15, 6, 12\n1, 0, Control_c, 15, 38, 0\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 480000\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, 25\n"
    else:
        header = "0, 0, Header, 1, 2, " + tempo + "\n1, 0, Start_track\n1, 0, Control_c, 0, 101, 0\n1, 0, Control_c, 0, 100, 0\n1, 0, Control_c, 0, 6, 12\n1, 0, Control_c, 0, 38, 0\n1, 0, Control_c, 1, 101, 0\n1, 0, Control_c, 1, 100, 0\n1, 0, Control_c, 1, 6, 12\n1, 0, Control_c, 1, 38, 0\n1, 0, Control_c, 2, 101, 0\n1, 0, Control_c, 2, 100, 0\n1, 0, Control_c, 2, 6, 12\n1, 0, Control_c, 2, 38, 0\n1, 0, Control_c, 3, 101, 0\n1, 0, Control_c, 3, 100, 0\n1, 0, Control_c, 3, 6, 12\n1, 0, Control_c, 3, 38, 0\n1, 0, Control_c, 4, 101, 0\n1, 0, Control_c, 4, 100, 0\n1, 0, Control_c, 4, 6, 12\n1, 0, Control_c, 4, 38, 0\n1, 0, Control_c, 5, 101, 0\n1, 0, Control_c, 5, 100, 0\n1, 0, Control_c, 5, 6, 12\n1, 0, Control_c, 5, 38, 0\n1, 0, Control_c, 6, 101, 0\n1, 0, Control_c, 6, 100, 0\n1, 0, Control_c, 6, 6, 12\n1, 0, Control_c, 6, 38, 0\n1, 0, Control_c, 7, 101, 0\n1, 0, Control_c, 7, 100, 0\n1, 0, Control_c, 7, 6, 12\n1, 0, Control_c, 7, 38, 0\n1, 0, Control_c, 8, 101, 0\n1, 0, Control_c, 8, 100, 0\n1, 0, Control_c, 8, 6, 12\n1, 0, Control_c, 8, 38, 0\n1, 0, Control_c, 9, 101, 0\n1, 0, Control_c, 9, 100, 0\n1, 0, Control_c, 9, 6, 12\n1, 0, Control_c, 9, 38, 0\n1, 0, Control_c, 10, 101, 0\n1, 0, Control_c, 10, 100, 0\n1, 0, Control_c, 10, 6, 12\n1, 0, Control_c, 10, 38, 0\n1, 0, Control_c, 11, 101, 0\n1, 0, Control_c, 11, 100, 0\n1, 0, Control_c, 11, 6, 12\n1, 0, Control_c, 11, 38, 0\n1, 0, Control_c, 12, 101, 0\n1, 0, Control_c, 12, 100, 0\n1, 0, Control_c, 12, 6, 12\n1, 0, Control_c, 12, 38, 0\n1, 0, Control_c, 13, 101, 0\n1, 0, Control_c, 13, 100, 0\n1, 0, Control_c, 13, 6, 12\n1, 0, Control_c, 13, 38, 0\n1, 0, Control_c, 14, 101, 0\n1, 0, Control_c, 14, 100, 0\n1, 0, Control_c, 14, 6, 12\n1, 0, Control_c, 14, 38, 0\n1, 0, Control_c, 15, 101, 0\n1, 0, Control_c, 15, 100, 0\n1, 0, Control_c, 15, 6, 12\n1, 0, Control_c, 15, 38, 0\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 480000\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, " + instrument + "\n"


##############################################################   
#read in saved parameters of rnn from master file
###############################################################
y = 0
with open(modelPath) as infile:
    lines = infile.readlines()
    for x in range(len(lines)):
        #read in hidden size
        if x == 0:
            hidden_size = int(lines[x])
        elif x == 1:
            seq_length = int(lines[x])
        elif x == 2:
            vocab_size = int(lines[x])
            print(vocab_size)

        #lines[3] is blank for readability
        elif x >= 4 and x < 4 + vocab_size + 1:
            chars.append(lines[x][0])
            y = y + 1
            
        #lines[4 + vocab_size + 2] is blank for readability
    #chars2 = list(set(data))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    #char_to_ix2 = { ch:i for i,ch in enumerate(chars2) }

    #char_to_ix = dict(char_to_ix2, **char_to_ix)
    #char_to_ix.update(char_to_ix2)
    #char_to_ix = union(char_to_ix, char_to_ix2)
    
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    #ix_to_char2 = { i:ch for i,ch in enumerate(chars2) }
    #ix_to_char = dict(ix_to_char2, **ix_to_char)
    #ix_to_char.update(ix_to_char2)
    #ix_to_char = union(ix_to_char, ix_to_char2)

    #initialize variables to be filled with stored values
    Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
    bh = np.zeros((hidden_size, 1)) # hidden bias
    by = np.zeros((vocab_size, 1)) # output bias

    index = 4 + vocab_size + 2

    for i in range(0, hidden_size):
        for j in range(0, vocab_size):
            #print(lines[index])
            Wxh[i][j] = float(lines[index])
            #print(str(Wxh[i][j]) + "\n")
            index = index + 1

    index = index + 1 #blank line for readability
    #print("BLANK\n")

    for i in range(0, hidden_size):
        for j in range(0, hidden_size):
            Whh[i][j] = float(lines[index])
            #print(str(Whh[i][j]) + "\n")
            index = index + 1

    index = index + 1 #blank line for readability
    #print("BLANK\n")

    for i in range(0, vocab_size):
        for j in range(0, hidden_size):
            Why[i][j] = float(lines[index])
            #print(str(Why[i][j]) + "\n")
            index = index + 1

    index = index + 1 #blank line for readability
    #print("BLANK\n")

    for i in range(0, hidden_size):
        bh[i][0] = float(lines[index])
        #print(str(bh[i][0]) + "\n")
        index = index + 1

    index = index + 1 #blank line for readability
    #print("BLANK\n")

    for i in range(0, vocab_size):
        by[i][0] = float(lines[index])
        #print(str(by[i][0]) + "\n")
        index = index + 1
    
infile.close()
#######################################################
#######################################################

#clean up data file
with open(dataFile) as infile:
    lines = infile.readlines()
    for line in lines:
        for char in line:
            if char not in chars:
                if line in lines:
                    lines.remove(line)

    with open("dataFresh.txt", 'w') as outfile:
        for line in lines:
            outfile.write(line)

# data I/O
data = open("dataFresh.txt", 'r').read() # should be simple plain text file
#chars = list(set(data))
#char_to_ix = { ch:i for i,ch in enumerate(chars) }
#ix_to_char = { i:ch for i,ch in enumerate(chars) }


z = 0
for char in chars:
    print(str(z)+": " + char)
    z = z + 1


'''
for char in chars:
    print(char)

for char in chars:
    print(char.rstrip())

'''

def updateHprev(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    #print (str(len(inputs)))
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        #print (str(inputs[t]))
        if inputs[t] >= vocab_size:
            inputs[t] = vocab_size - 1
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    return hs[len(inputs) - 1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
            
  return ixes

##############################################################
#Now produce a sample with the model that had been loaded in.
##############################################################
#n, p = 0, 0
n = 1

if default == True:
    p = np.random.randint(0, len(data) - seq_length - 5)  #random placement of data cursor so samples are different
else:
    p = seed % (len(data) - seq_length - 5)
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
hprev = np.zeros((hidden_size,1)) # reset RNN memory


for n in range(5):
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    #hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model exactly once
  if n == 4:     
    sample_ix = sample(hprev, inputs[0], 3000)  #200 originally
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)

    ##################
    txtNew = ""
    for char in str(txt):
      for y in range(len(chars)):
        if chars[y] == char and chars[y + 1] != specialChar:
          if y != len(chars) - 1:
            if model == "jimiHendrixGuitar":
                char2 = chars[y]
            else:
                char2 = chars[y + 1]
            #if char2 == ":" or char2 == "V" or char2 == ")" and model == "mozartPiano":  #to fix weird issues with Mozart Piano
            #   char2 = "+"
            #elif char2 == ")" and model == "mozartPiano":
            #    char2 = "+"
            #elif char2 == "8" or char2 == "2" or char2 == "3" or char2 == "7" or char2 == "4" or char2 == "5" and model == "mozartPiano":
            #    char2 = "+"
            txtNew = txtNew + char2
          else:
            if model =="jimiHendrixGuitar":
                char2 = chars[0]
            else:
                char2 = chars[y]
            txtNew = txtNew + char2
           
    #####################
            
    #print '----\n %s \n----' % (txt, )
    #print  ("----\n", txt, "\n----")
    print   ("----\n", txtNew, "\n----")

    #Create csv and midi file of sample by invoking midiScripts
    filestring = sampleDirectory + "shortSample" + ".txt"
    with open(filestring, 'w') as outfile:
      outfile.write(txtNew)

    filestring2 = sampleDirectory + "longSample" + ".txt"
    open(filestring2, 'w+')
    filestring3 = sampleDirectory + "sampleCSV" + ".txt"
  
    cmd = ["./midiScript2", filestring, filestring2]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()

    #footer for csv file
    footer = "3, 0, Start_track\n3, 0, MIDI_port, 0\n3, 0, Title_t, \"--------------------------------------\"\n3, 0, Program_c, 1, 40\n3, 0, Control_c, 1, 7, 100\n3, 0, Control_c, 1, 10, 74\n3, 59760, End_track\n4, 0, Start_track\n4, 0, MIDI_port, 0\n4, 0, Title_t, \"Johann Sebastian Bach  (1685-1750)\"\n4, 0, Program_c, 2, 40\n4, 0, Control_c, 2, 7, 100\n4, 0, Control_c, 2, 10, 54\n4, 59760, End_track\n5, 0, Start_track\n5, 0, MIDI_port, 0\n5, 0, Title_t, \"Six Sonatas and Partitas for Solo Violin\"\n5, 0, End_track\n6, 0, Start_track\n6, 0, MIDI_port, 0\n6, 0, Title_t, \"--------------------------------------\"\n6, 0, End_track\n7, 0, Start_track\n7, 0, MIDI_port, 0\n7, 0, Title_t, \"Partita No. 1 in B minor - BWV 1002\"\n7, 0, End_track\n8, 0, Start_track\n8, 0, MIDI_port, 0\n8, 0, Title_t, \"3rd Movement: Corrente\"\n8, 0, End_track\n9, 0, Start_track\n9, 0, MIDI_port, 0\n9, 0, Title_t, \"--------------------------------------\"\n9, 0, End_track\n10, 0, Start_track\n10, 0, MIDI_port, 0\n10, 0, Title_t, \"Sequenced with Cakewalk Pro Audio by\"\n10, 0, End_track\n11, 0, Start_track\n11, 0, MIDI_port, 0\n11, 0, Title_t, \"David J. Grossman - dave@unpronounceable.com\"\n11, 0, End_track\n12, 0, Start_track\n12, 0, MIDI_port, 0\n12, 0, Title_t, \"This and other Bach MIDI files can be found at:\"\n12, 0, End_track\n13, 0, Start_track\n13, 0, MIDI_port, 0\n13, 0, Title_t, \"Dave's J.S. Bach Page\"\n13, 0, End_track\n14, 0, Start_track\n14, 0, MIDI_port, 0\n14, 0, Title_t, \"http://www.unpronounceable.com/bach\"\n14, 0, End_track\n15, 0, Start_track\n15, 0, MIDI_port, 0\n15, 0, Title_t, \"--------------------------------------\"\n15, 0, End_track\n16, 0, Start_track\n16, 0, MIDI_port, 0\n16, 0, Title_t, \"Original Filename: vp1-3co.mid\"\n16, 0, End_track\n17, 0, Start_track\n17, 0, MIDI_port, 0\n17, 0, Title_t, \"Last Modified: February 22, 1997\"\n17, 0, End_track\n0, 0, End_of_file\n"

    #create csv file w/ header and footer
    with open(filestring2, 'r') as infile:
      content = infile.read()
      with open(filestring3, 'w+') as outfile:
        outfile.write(header + content + footer)

    #create midi file from csv
    if choice == 1:    
        cmd = ["./csvmidi", filestring3, sampleDirectory + "classicalViolinSample" + ".mid"]
    elif choice == 2:
        cmd = ["./csvmidi", filestring3, sampleDirectory + "jazzPianoSample" + ".mid"]
    elif choice == 4:
        cmd = ["./csvmidi", filestring3, sampleDirectory + "mozartPianoSample" + ".mid"]
    elif choice == 5:
        cmd = ["./csvmidi", filestring3, sampleDirectory + "jimiHendrixGuitarSample" + ".mid"]
    else:
        cmd = ["./csvmidi", filestring3, sampleDirectory + "bluesGuitarSample" + ".mid"]
        
    #cmd = ["./csvmidi", filestring3, sampleDirectory + "spanishGuitarSample" + ".mid"]
    
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()
    

  #update Hprev
  hprev = updateHprev(inputs, targets, hprev)
  

  p += seq_length # move data pointer
  n += 1 # iteration counter



        
        
        
    

