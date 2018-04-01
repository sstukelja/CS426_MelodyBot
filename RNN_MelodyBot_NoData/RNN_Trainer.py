"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Modified by Nick Harris
"""
import numpy as np
import os
import subprocess
from subprocess import Popen, PIPE

"""
This block added by Nick - invokes c++ scripts to convert
    midi/text data for learning
"""

i = 0

#clear the content of text files
open('longDatax.txt', 'w').close()
open('shortDatax.txt', 'w').close()
open('shortDataFresh.txt', 'w').close()

#create csv files from each midi in testing data

directory = "testing_music_reggae_drum\\"
#directory = "testing_music_jimi_hendrix_guitar\\"
#directory = "testing_music_mozart_piano\\"
#directory = "testing_music_solo_violin\\"
#directory = "testing_music_jazz_piano\\"
#directory = "testing_music_blues_guitar\\"
#directory = "testing_music_spanish_guitar\\"


for filename in os.listdir(directory):
  if filename.endswith(".mid"):
    filestring = directory + filename
    #cmd = ["./midicsv", filestring, directory + "jazzPianoCSV" + str(i) + ".txt"]
    #cmd = ["./midicsv", filestring, directory + "soloViolinCSV" + str(i) + ".txt"]
    #cmd = ["./midicsv", filestring, directory + "bluesGuitarCSV" + str(i) + ".txt"]
    #cmd = ["./midicsv", filestring, directory + "spanishGuitarCSV" + str(i) + ".txt"]
    #cmd = ["./midicsv", filestring, directory + "mozartPianoCSV" + str(i) + ".txt"]
    #cmd = ["./midicsv", filestring, directory + "jimiHendrixGuitarCSV" + str(i) + ".txt"]
    cmd = ["./midicsv", filestring, directory + "reggaeDrumCSV" + str(i) + ".txt"]
    i += 1
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()

#write each csv training data file into longDatax
with open("longDatax.txt", 'w') as outfile:
  for filename in os.listdir(directory):
    if filename.endswith(".txt"):
      with open(directory + filename) as infile:
        outfile.write(infile.read())
    
#convert longData into shortData for learning wih midiScript
cmd = ["./midiScript", "longDatax.txt", "shortDatax.txt"]
result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
out = result.stdout.read()

#clean up shortDatax

with open("shortDatax.txt") as infile:
  lines = infile.readlines()
  for line in lines:
    if len(line) > 100:
      lines.remove(line)

  with open("shortDataFresh.txt", 'w') as outfile:
    for line in lines:
      if len(line) < 100:
        outfile.write(line)

"""
"""

# data I/O
data = open('shortDataFresh.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
#print 'data has %d characters, %d unique.' % (data_size, vocab_size)
print ("data has",data_size,"characters, ", vocab_size, "unique.")
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 250 # size of hidden layer of neurons  was 100
seq_length = 30 # number of steps to unroll the RNN for     was 25 # then 200/30
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

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

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line + content)


#sampleDirectory = os.path.normpath("generated_music_solo_violin\\")
#sampleDirectory = "generated_music_solo_violin\\"
#sampleDirectory = "generated_music_jazz_piano\\"
#sampleDirectory = "generated_music_blues_guitar\\"
#sampleDirectory = "generated_music_spanish_guitar\\"
#sampleDirectory = "generated_music_mozart_piano\\"
#sampleDirectory = "generated_music_jimi_hendrix_guitar\\"
sampleDirectory = "generated_music_reggae_drum\\"

#model directory
modelDirectory = "trained_models\\"


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
hprev = np.zeros((hidden_size,1)) # reset RNN memory
for n in range(10000000):
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    #hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 10000 == 0:     #was 5000
    sample_ix = sample(hprev, inputs[0], 3000)  #200 originally
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    #print '----\n %s \n----' % (txt, )
    print  ("----\n", txt, "\n----")

    #Create csv and midi file of sample by invoking midiScripts
    filestring = sampleDirectory + "shortSample" + str(n) + ".txt"
    with open(filestring, 'w') as outfile:
      outfile.write(txt)

    filestring2 = sampleDirectory + "longSample" + str(n) + ".txt"
    open(filestring2, 'w+')
    filestring3 = sampleDirectory + "sampleCSV" + str(n) + ".txt"
  
    cmd = ["./midiScript2", filestring, filestring2]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()

    #header for csv file (solo violin)
    #header = "0, 0, Header, 1, 17, 480\n1, 0, Start_track\n1, 0, Title_t, \"untitled\"\n1, 0, Copyright_t, \"Copyright © 1996 by David J. Grossman\"\n1, 0, Text_t, \"David J. Grossman\"\n1, 0, SMPTE_offset, 96, 0, 3, 0, 0\n1, 0, Time_signature, 3, 2, 24, 8\n1, 0, Key_signature, 2, \"major\"\n1, 0, Tempo, 240000\n1, 1200, Tempo, 461538\n1, 1440, Marker_t, \"A\"\n1, 47520, Marker_t, \"A'\"\n1, 93600, Marker_t, \"B\"\n1, 162720, Marker_t, \"B'\"\n1, 230400, Tempo, 923077\n1, 230400, End_track\n2, 0, Start_track\n2, 0, MIDI_port, 0\n2, 0, Title_t, \"Solo Violin\"\n2, 0, Program_c, 0, 40\n2, 0, Control_c, 0, 7, 100\n2, 0, Control_c, 0, 10, 64\n"

    #header for jazz piano
    #header = "0, 0, Header, 0, 1, 120\n1, 0, Start_track\n1, 230400, End_track\n2, 0, Start_track\n2, 0, Title_t, \"A House Is Not A Home\"\n2, 0, Text_t, \"Dougmck\\012\"\n2, 0, Time_signature, 4, 2, 24, 8\n2, 0, Key_signature, 0, \"major\"\n2, 0, Tempo, 681818\n2, 0, Program_c, 0, 0\n2, 0, Program_c, 1, 0\n"

    #header for mozart piano
    #header = "0, 0, Header, 0, 1, 120\n1, 0, Start_track\n1, 230400, End_track\n2, 0, Start_track\n2, 0, Title_t, \"A House Is Not A Home\"\n2, 0, Text_t, \"Dougmck\\012\"\n2, 0, Time_signature, 4, 2, 24, 8\n2, 0, Key_signature, 0, \"major\"\n2, 0, Tempo, 803643\n2, 0, Program_c, 0, 0\n2, 0, Program_c, 1, 0\n"

    #header for blues/spanish guitar
    #header = "0, 0, Header, 1, 2, 960\n1, 0, Start_track\n1, 0, Control_c, 0, 101, 0\n1, 0, Control_c, 0, 100, 0\n1, 0, Control_c, 0, 6, 12\n1, 0, Control_c, 0, 38, 0\n1, 0, Control_c, 1, 101, 0\n1, 0, Control_c, 1, 100, 0\n1, 0, Control_c, 1, 6, 12\n1, 0, Control_c, 1, 38, 0\n1, 0, Control_c, 2, 101, 0\n1, 0, Control_c, 2, 100, 0\n1, 0, Control_c, 2, 6, 12\n1, 0, Control_c, 2, 38, 0\n1, 0, Control_c, 3, 101, 0\n1, 0, Control_c, 3, 100, 0\n1, 0, Control_c, 3, 6, 12\n1, 0, Control_c, 3, 38, 0\n1, 0, Control_c, 4, 101, 0\n1, 0, Control_c, 4, 100, 0\n1, 0, Control_c, 4, 6, 12\n1, 0, Control_c, 4, 38, 0\n1, 0, Control_c, 5, 101, 0\n1, 0, Control_c, 5, 100, 0\n1, 0, Control_c, 5, 6, 12\n1, 0, Control_c, 5, 38, 0\n1, 0, Control_c, 6, 101, 0\n1, 0, Control_c, 6, 100, 0\n1, 0, Control_c, 6, 6, 12\n1, 0, Control_c, 6, 38, 0\n1, 0, Control_c, 7, 101, 0\n1, 0, Control_c, 7, 100, 0\n1, 0, Control_c, 7, 6, 12\n1, 0, Control_c, 7, 38, 0\n1, 0, Control_c, 8, 101, 0\n1, 0, Control_c, 8, 100, 0\n1, 0, Control_c, 8, 6, 12\n1, 0, Control_c, 8, 38, 0\n1, 0, Control_c, 9, 101, 0\n1, 0, Control_c, 9, 100, 0\n1, 0, Control_c, 9, 6, 12\n1, 0, Control_c, 9, 38, 0\n1, 0, Control_c, 10, 101, 0\n1, 0, Control_c, 10, 100, 0\n1, 0, Control_c, 10, 6, 12\n1, 0, Control_c, 10, 38, 0\n1, 0, Control_c, 11, 101, 0\n1, 0, Control_c, 11, 100, 0\n1, 0, Control_c, 11, 6, 12\n1, 0, Control_c, 11, 38, 0\n1, 0, Control_c, 12, 101, 0\n1, 0, Control_c, 12, 100, 0\n1, 0, Control_c, 12, 6, 12\n1, 0, Control_c, 12, 38, 0\n1, 0, Control_c, 13, 101, 0\n1, 0, Control_c, 13, 100, 0\n1, 0, Control_c, 13, 6, 12\n1, 0, Control_c, 13, 38, 0\n1, 0, Control_c, 14, 101, 0\n1, 0, Control_c, 14, 100, 0\n1, 0, Control_c, 14, 6, 12\n1, 0, Control_c, 14, 38, 0\n1, 0, Control_c, 15, 101, 0\n1, 0, Control_c, 15, 100, 0\n1, 0, Control_c, 15, 6, 12\n1, 0, Control_c, 15, 38, 0\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 480000\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, 25\n"

    #header for Jimi Hendrix guitar
    header = "0, 0, Header, 1, 2, 960\n1, 0, Start_track\n1, 0, Control_c, 0, 101, 0\n1, 0, Control_c, 0, 100, 0\n1, 0, Control_c, 0, 6, 12\n1, 0, Control_c, 0, 38, 0\n1, 0, Control_c, 1, 101, 0\n1, 0, Control_c, 1, 100, 0\n1, 0, Control_c, 1, 6, 12\n1, 0, Control_c, 1, 38, 0\n1, 0, Control_c, 2, 101, 0\n1, 0, Control_c, 2, 100, 0\n1, 0, Control_c, 2, 6, 12\n1, 0, Control_c, 2, 38, 0\n1, 0, Control_c, 3, 101, 0\n1, 0, Control_c, 3, 100, 0\n1, 0, Control_c, 3, 6, 12\n1, 0, Control_c, 3, 38, 0\n1, 0, Control_c, 4, 101, 0\n1, 0, Control_c, 4, 100, 0\n1, 0, Control_c, 4, 6, 12\n1, 0, Control_c, 4, 38, 0\n1, 0, Control_c, 5, 101, 0\n1, 0, Control_c, 5, 100, 0\n1, 0, Control_c, 5, 6, 12\n1, 0, Control_c, 5, 38, 0\n1, 0, Control_c, 6, 101, 0\n1, 0, Control_c, 6, 100, 0\n1, 0, Control_c, 6, 6, 12\n1, 0, Control_c, 6, 38, 0\n1, 0, Control_c, 7, 101, 0\n1, 0, Control_c, 7, 100, 0\n1, 0, Control_c, 7, 6, 12\n1, 0, Control_c, 7, 38, 0\n1, 0, Control_c, 8, 101, 0\n1, 0, Control_c, 8, 100, 0\n1, 0, Control_c, 8, 6, 12\n1, 0, Control_c, 8, 38, 0\n1, 0, Control_c, 9, 101, 0\n1, 0, Control_c, 9, 100, 0\n1, 0, Control_c, 9, 6, 12\n1, 0, Control_c, 9, 38, 0\n1, 0, Control_c, 10, 101, 0\n1, 0, Control_c, 10, 100, 0\n1, 0, Control_c, 10, 6, 12\n1, 0, Control_c, 10, 38, 0\n1, 0, Control_c, 11, 101, 0\n1, 0, Control_c, 11, 100, 0\n1, 0, Control_c, 11, 6, 12\n1, 0, Control_c, 11, 38, 0\n1, 0, Control_c, 12, 101, 0\n1, 0, Control_c, 12, 100, 0\n1, 0, Control_c, 12, 6, 12\n1, 0, Control_c, 12, 38, 0\n1, 0, Control_c, 13, 101, 0\n1, 0, Control_c, 13, 100, 0\n1, 0, Control_c, 13, 6, 12\n1, 0, Control_c, 13, 38, 0\n1, 0, Control_c, 14, 101, 0\n1, 0, Control_c, 14, 100, 0\n1, 0, Control_c, 14, 6, 12\n1, 0, Control_c, 14, 38, 0\n1, 0, Control_c, 15, 101, 0\n1, 0, Control_c, 15, 100, 0\n1, 0, Control_c, 15, 6, 12\n1, 0, Control_c, 15, 38, 0\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 800000\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, 28\n"

    #header for reggae drum
    #header = "0, 0, Header, 1, 2, 120\n1, 0, Start_track\n1, 0, Time_signature, 4, 2, 24, 8\n1, 0, Tempo, 500000\n1, 0, End_track\n2, 0, Start_track\n2, 0, Program_c, 0, 114\n2, 0, Tempo, 500000\n"    

    #footer for csv file
    footer = "3, 0, Start_track\n3, 0, MIDI_port, 0\n3, 0, Title_t, \"--------------------------------------\"\n3, 0, Program_c, 1, 40\n3, 0, Control_c, 1, 7, 100\n3, 0, Control_c, 1, 10, 74\n3, 59760, End_track\n4, 0, Start_track\n4, 0, MIDI_port, 0\n4, 0, Title_t, \"Johann Sebastian Bach  (1685-1750)\"\n4, 0, Program_c, 2, 40\n4, 0, Control_c, 2, 7, 100\n4, 0, Control_c, 2, 10, 54\n4, 59760, End_track\n5, 0, Start_track\n5, 0, MIDI_port, 0\n5, 0, Title_t, \"Six Sonatas and Partitas for Solo Violin\"\n5, 0, End_track\n6, 0, Start_track\n6, 0, MIDI_port, 0\n6, 0, Title_t, \"--------------------------------------\"\n6, 0, End_track\n7, 0, Start_track\n7, 0, MIDI_port, 0\n7, 0, Title_t, \"Partita No. 1 in B minor - BWV 1002\"\n7, 0, End_track\n8, 0, Start_track\n8, 0, MIDI_port, 0\n8, 0, Title_t, \"3rd Movement: Corrente\"\n8, 0, End_track\n9, 0, Start_track\n9, 0, MIDI_port, 0\n9, 0, Title_t, \"--------------------------------------\"\n9, 0, End_track\n10, 0, Start_track\n10, 0, MIDI_port, 0\n10, 0, Title_t, \"Sequenced with Cakewalk Pro Audio by\"\n10, 0, End_track\n11, 0, Start_track\n11, 0, MIDI_port, 0\n11, 0, Title_t, \"David J. Grossman - dave@unpronounceable.com\"\n11, 0, End_track\n12, 0, Start_track\n12, 0, MIDI_port, 0\n12, 0, Title_t, \"This and other Bach MIDI files can be found at:\"\n12, 0, End_track\n13, 0, Start_track\n13, 0, MIDI_port, 0\n13, 0, Title_t, \"Dave's J.S. Bach Page\"\n13, 0, End_track\n14, 0, Start_track\n14, 0, MIDI_port, 0\n14, 0, Title_t, \"http://www.unpronounceable.com/bach\"\n14, 0, End_track\n15, 0, Start_track\n15, 0, MIDI_port, 0\n15, 0, Title_t, \"--------------------------------------\"\n15, 0, End_track\n16, 0, Start_track\n16, 0, MIDI_port, 0\n16, 0, Title_t, \"Original Filename: vp1-3co.mid\"\n16, 0, End_track\n17, 0, Start_track\n17, 0, MIDI_port, 0\n17, 0, Title_t, \"Last Modified: February 22, 1997\"\n17, 0, End_track\n0, 0, End_of_file\n"

    #create csv file w/ header and footer
    with open(filestring2, 'r') as infile:
      content = infile.read()
      with open(filestring3, 'w+') as outfile:
        outfile.write(header + content + footer)

    #create midi file from csv
    #cmd = ["./csvmidi", filestring3, sampleDirectory + "mozartPianoSample" + str(n) + ".mid"]
    #cmd = ["./csvmidi", filestring3, sampleDirectory + "classicalViolinSample" + str(n) + ".mid"]
    #cmd = ["./csvmidi", filestring3, sampleDirectory + "jazzPianoSample" + str(n) + ".mid"]
    #cmd = ["./csvmidi", filestring3, sampleDirectory + "bluesGuitarSample" + str(n) + ".mid"]
    #cmd = ["./csvmidi", filestring3, sampleDirectory + "spanishGuitarSample" + str(n) + ".mid"]
    #cmd = ["./csvmidi", filestring3, sampleDirectory + "jimiHendrixGuitarSample" + str(n) + ".mid"]
    cmd = ["./csvmidi", filestring3, sampleDirectory + "reggaeDrumSample" + str(n) + ".mid"]
    
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()

    #save trained model parameters
    
    #filestring4 = modelDirectory + "jazzPiano" + str(n) + ".txt"
    #filestring4 = modelDirectory + "classicalViolin" + str(n) + ".txt"
    #filestring4 = modelDirectory + "bluesGuitar" + str(n) + ".txt"
    #filestring4 = modelDirectory + "spanishGuitar" + str(n) + ".txt"
    #filestring4 = modelDirectory + "mozartPiano" + str(n) + ".txt"
    #filestring4 = modelDirectory + "jimiHendrixGuitar" + str(n) + ".txt"
    filestring4 = modelDirectory + "reggaeDrum" + str(n) + ".txt"
    
    with open(filestring4, 'w') as outfile:
      outfile.write(str(hidden_size) + '\n' + str(seq_length) + '\n' + str(vocab_size) + "\n")
      outfile.write("\n")
      for i in range(0, len(chars)):
        outfile.write(chars[i] + "\n")
      outfile.write("\n")
      for i in range (0, hidden_size):
        for j in range(0, vocab_size):
          outfile.write(str(Wxh[i][j]) + '\n')
      outfile.write("\n")
      for i in range(0, hidden_size):
        for j in range(0, hidden_size):
          outfile.write(str(Whh[i][j]) + '\n')
      outfile.write("\n")
      for i in range (0, vocab_size):
        for j in range (0, hidden_size):
          outfile.write(str(Why[i][j]) + '\n')
      outfile.write("\n")
      for i in range (0, hidden_size):
        outfile.write(str(bh[i][0]) + '\n')
      outfile.write("\n")
      for i in range (0, vocab_size):
        outfile.write(str(by[i][0]) + '\n')
      outfile.write("\n")
      for i in range (0, hidden_size):
        outfile.write(str(hprev[i][0]) + '\n')
    

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print ("iter " ,n, " loss: ",smooth_loss, " ") # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter
