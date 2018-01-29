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

directory = "testing_music_solo_violin\\"
for filename in os.listdir(directory):
  if filename.endswith(".mid"):
    filestring = directory + filename
    cmd = ["./midicsv", filestring, directory + "violinCSV" + str(i) + ".txt"]
    i += 1
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()

with open("longDatax.txt", 'w') as outfile:
  for filename in os.listdir(directory):
    if filename.endswith(".txt"):
      with open(directory + filename) as infile:
        outfile.write(infile.read())
    

cmd = ["./midiScript", "longDatax.txt", "shortDatax.txt"]
result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
out = result.stdout.read()

#clean up shortDatax

with open("shortDatax.txt") as infile:
  lines = infile.readlines()
  for line in lines:
    if len(line) > 30:
      lines.remove(line)

  with open("shortDataFresh.txt", 'w') as outfile:
    for line in lines:
      if len(line) < 30:
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
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
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
sampleDirectory = "generated_music_solo_violin\\"


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
hprev = np.zeros((hidden_size,1)) # reset RNN memory
for n in range(50000):
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    #hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 1000 == 0:
    sample_ix = sample(hprev, inputs[0], 1500)  #200 originally
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    #print '----\n %s \n----' % (txt, )
    print  ("----\n", txt, "\n----")
    filestring = sampleDirectory + "shortSample" + str(n) + ".txt"
    with open(filestring, 'w') as outfile:
      outfile.write(txt)

    filestring2 = sampleDirectory + "longSample" + str(n) + ".txt"
    open(filestring2, 'w+')
    filestring3 = sampleDirectory + "sampleCSV" + str(n) + ".txt"
  
    cmd = ["./midiScript2", filestring, filestring2]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()

    #header for csv file
    header = "0, 0, Header, 1, 17, 480\n1, 0, Start_track\n1, 0, Title_t, \"untitled\"\n1, 0, Copyright_t, \"Copyright © 1996 by David J. Grossman\"\n1, 0, Text_t, \"David J. Grossman\"\n1, 0, SMPTE_offset, 96, 0, 3, 0, 0\n1, 0, Time_signature, 3, 2, 24, 8\n1, 0, Key_signature, 2, \"major\"\n1, 0, Tempo, 240000\n1, 1200, Tempo, 461538\n1, 1440, Marker_t, \"A\"\n1, 47520, Marker_t, \"A'\"\n1, 93600, Marker_t, \"B\"\n1, 162720, Marker_t, \"B'\"\n1, 230400, Tempo, 923077\n1, 230400, End_track\n2, 0, Start_track\n2, 0, MIDI_port, 0\n2, 0, Title_t, \"Solo Violin\"\n2, 0, Program_c, 0, 40\n2, 0, Control_c, 0, 7, 100\n2, 0, Control_c, 0, 10, 64\n"
    #line_prepender(filestring2, header)
    footer = "3, 0, Start_track\n3, 0, MIDI_port, 0\n3, 0, Title_t, \"--------------------------------------\"\n3, 0, Program_c, 1, 40\n3, 0, Control_c, 1, 7, 100\n3, 0, Control_c, 1, 10, 74\n3, 13440, Note_on_c, 1, 66, 100\n3, 13680, Note_on_c, 1, 66, 0\n3, 59520, Note_on_c, 1, 66, 100\n3, 59760, Note_on_c, 1, 66, 0\n3, 59760, End_track\n4, 0, Start_track\n4, 0, MIDI_port, 0\n4, 0, Title_t, \"Johann Sebastian Bach  (1685-1750)\"\n4, 0, Program_c, 2, 40\n4, 0, Control_c, 2, 7, 100\n4, 0, Control_c, 2, 10, 54\n4, 13440, Note_on_c, 2, 59, 100\n4, 13680, Note_on_c, 2, 59, 0\n4, 59520, Note_on_c, 2, 59, 100\n4, 59760, Note_on_c, 2, 59, 0\n4, 59760, End_track\n5, 0, Start_track\n5, 0, MIDI_port, 0\n5, 0, Title_t, \"Six Sonatas and Partitas for Solo Violin\"\n5, 0, End_track\n6, 0, Start_track\n6, 0, MIDI_port, 0\n6, 0, Title_t, \"--------------------------------------\"\n6, 0, End_track\n7, 0, Start_track\n7, 0, MIDI_port, 0\n7, 0, Title_t, \"Partita No. 1 in B minor - BWV 1002\"\n7, 0, End_track\n8, 0, Start_track\n8, 0, MIDI_port, 0\n8, 0, Title_t, \"3rd Movement: Corrente\"\n8, 0, End_track\n9, 0, Start_track\n9, 0, MIDI_port, 0\n9, 0, Title_t, \"--------------------------------------\"\n9, 0, End_track\n10, 0, Start_track\n10, 0, MIDI_port, 0\n10, 0, Title_t, \"Sequenced with Cakewalk Pro Audio by\"\n10, 0, End_track\n11, 0, Start_track\n11, 0, MIDI_port, 0\n11, 0, Title_t, \"David J. Grossman - dave@unpronounceable.com\"\n11, 0, End_track\n12, 0, Start_track\n12, 0, MIDI_port, 0\n12, 0, Title_t, \"This and other Bach MIDI files can be found at:\"\n12, 0, End_track\n13, 0, Start_track\n13, 0, MIDI_port, 0\n13, 0, Title_t, \"Dave's J.S. Bach Page\"\n13, 0, End_track\n14, 0, Start_track\n14, 0, MIDI_port, 0\n14, 0, Title_t, \"http://www.unpronounceable.com/bach\"\n14, 0, End_track\n15, 0, Start_track\n15, 0, MIDI_port, 0\n15, 0, Title_t, \"--------------------------------------\"\n15, 0, End_track\n16, 0, Start_track\n16, 0, MIDI_port, 0\n16, 0, Title_t, \"Original Filename: vp1-3co.mid\"\n16, 0, End_track\n17, 0, Start_track\n17, 0, MIDI_port, 0\n17, 0, Title_t, \"Last Modified: February 22, 1997\"\n17, 0, End_track\n0, 0, End_of_file\n"

    with open(filestring2, 'r') as infile:
      content = infile.read()
      with open(filestring3, 'w+') as outfile:
        outfile.write(header + content + footer)

    cmd = ["./csvmidi", filestring3, sampleDirectory + "violinSample" + str(n) + ".mid"]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = result.stdout.read()

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
