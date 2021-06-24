'''
Adaptat dupa codul din https://github.com/radu-dogaru/ELM-super-fast
@ R. Dogaru, UPB ETTI - aprilie 2019 
Suport laborator ICI - anul IVA (program INF)
Include optiunea "neuroni ascunsi = 0" care implementeaza un 
sistem Adaline cu antrenare ELM 
'''

import os
import numpy as np
import scipy.linalg
import time as ti 
from skimage.io import imread
from skimage.filters import threshold_otsu

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

#training_directory este directorul in care se afla baza de date cu imagini
#number_of_files reprezintă numărul de fisiere pe care le folosim pentru antrenare.
def read_training_data(training_directory, number_of_files):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(number_of_files):
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            # citesc imaginea fiecarui caracter
            img_details = imread(image_path, as_gray=True)
            #convertesc fiecare imagine intr-o imagine binara
            binary_image = img_details < threshold_otsu(img_details)
            binary_image = binary_image.astype(int)
			# fiecare imagine are dimensiunea 20x20 pixeli.
			# voi transforma aceasta matrice, intr-un vector de tipul 1 x 400
			# deoarece pe acest tip de structura functioneaza clasificatorul.
			# fiecare pixel reprezinta o trasatura, astfel vom avea 400 de trasaturi
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def hidden_nonlin(hid_in, tip):
# implementation of the hidden layer 
# additional nonlinearitys may be added 
    if tip==0: 
        # sigmoid 
        H=np.tanh(hid_in)        
    elif tip==1:
        # linsat 
        H=abs(1+hid_in)-abs(1-hid_in)
    elif tip==2:
        # ReLU
        H=abs(hid_in)+hid_in
    elif tip==3:
        # see [1] - very well suited for emmbeded systems 
        H=abs(hid_in)
    elif tip==4:
        H=np.sqrt(hid_in*hid_in+1)
        # multiquadric 
    elif tip==5:
        H=np.sign(hid_in)
        # multiquadric
    return H
        

def elmTrain_optim(X, Y, h_Neurons, C , tip):
# Training phase - floating point precision (no quantization)
# X - Samples (feature vectors) Y - Labels
      Ntr = np.size(X,1)
      in_Neurons = np.size(X,0)
      classes = np.max(Y)
      # transforms label into binary columns  
      targets = np.zeros( (classes, Ntr), dtype='int8' )
      for i in range(0,Ntr):
          targets[Y[i]-1, i ] = 1
      targets = targets * 2 - 1
      
      #   Generate inW layer  
      if h_Neurons>0:
          rnd = np.random.RandomState()
          inW=-1+2*rnd.rand(h_Neurons, in_Neurons).astype('float32')
          #inW=rnd.randn(nHiddenNeurons, nInputNeurons).astype('float32')
      
          #  Compute hidden layer 
          hid_inp = np.dot(inW, X)
          H=hidden_nonlin(hid_inp,tip)
      elif h_Neurons==0:
          inW=[]
          h_Neurons=in_Neurons
          H = X
      
      # Moore - Penrose computation of output weights (outW) layer 
      if h_Neurons<Ntr:
          print('LLL - Less neurons than training samples')
          outW = scipy.linalg.solve(np.eye(h_Neurons)/C+np.dot(H,H.T), np.dot(H,targets.T))     
      else:
          print('MMM - More neurons than training samples')
          outW = np.dot(H,scipy.linalg.solve(np.eye(Ntr)/C+np.dot(H.T,H), targets.T))
      return inW, outW 
 

# implements the ELM training procedure with weight quantization       
def elmTrain_fix( X, Y, h_Neurons, C , tip, ni):
# Training phase - emulated fixed point precision (ni bit quantization)
# X - Samples (feature vectors) Y - Labels
# ni - number of bits to quantize the inW weights 
      Ntr = np.size(X,1)
      in_Neurons = np.size(X,0)
      classes = np.max(Y)
      # transforms label into binary columns  
      targets = np.zeros( (classes, Ntr), dtype='int8' )
      for i in range(0,Ntr):
          targets[Y[i]-1, i ] = 1
      targets = targets * 2 - 1
      
      #   Generare inW 
      #   Generate inW layer  
      rnd = np.random.RandomState()
      inW=-1+2*rnd.rand(h_Neurons, in_Neurons).astype('float32')
      #inW=rnd.randn(nHiddenNeurons, nInputNeurons).astype('float32')
      Qi=-1+pow(2,ni-1) 
      inW=np.round(inW*Qi)
      
      #  Compute hidden layer 
      hid_inp = np.dot(inW, X)
      H=hidden_nonlin(hid_inp,tip)
     
      # Moore - Penrose computation of output weights (outW) layer 
      if h_Neurons<Ntr:
          print('LLL - Less neurons than training samples')
          outW = scipy.linalg.solve(np.eye(h_Neurons)/C+np.dot(H,H.T), np.dot(H,targets.T))     
      else:
          print('MMM - More neurons than training samples')
          outW = np.dot(H,scipy.linalg.solve(np.eye(Ntr)/C+np.dot(H.T,H), targets.T))
     
      return inW, outW 
      

def elmPredict_optim( X, inW, outW, tip):
# implements the ELM predictor given the model as arguments 
# model is simply given by inW, outW and tip 
# returns a score matrix (winner class has the maximal score)
      if inW==[]: 
        H=X
        
      else: 
        hid_in=np.dot(inW, X)
        H=hidden_nonlin(hid_in,tip)
      print('aspect:',np.shape(H))
      score = np.transpose(np.dot(np.transpose(H),outW))
      return score 

# ======================================================
#  IMPLEMENTATION 
#================================================================================
# parameters 
nr_neuroni=800 # Daca se aleg 0 - implementeaza Adaline  (pe cuantizor)
C=0.0001 # Coeficient de regularizare -  C
tip=2 # Neliniaritate strat ascuns (0-tanh, 1-linsat, 2-ReLu, 3- abs(), 4-multiquad, 5 sign() )  
nb_in=2;  # 0 = float; >0 = nr. biti cuantizare ponderi strat intrare 
nb_out=8; # 0 = float; >0 = nr. biti cuantizare ponderi strat iesire
      
# current_dir = os.path.dirname(os.path.realpath(__file__))
#
# training_dataset_dir = os.path.join(current_dir, 'train')

#===============  TRAIN DATASET LOADING ==========================================
# converts into 'float32' for faster execution 
print('reading data')
t1 = ti.time()
training_dataset_dir = './train20X20'
number_of_files = 7
image_data, target_data = read_training_data(training_dataset_dir,number_of_files)
print('reading data completed')
Samples=image_data.astype('float32')
size_of_sample = len(Samples)
Labels=target_data
Labels2 = np.zeros((size_of_sample,1), dtype=int)
counter = number_of_files-1;
value_of_label = 1;
for value in range(size_of_sample):
    Labels2[value] = value_of_label
    if counter == 0:
        counter = 6
        value_of_label = value_of_label+1
    else:
        counter=counter-1
Labels2=np.transpose(Labels2)
Samples=np.transpose(Samples)
clase=np.max(Labels2)+1
trun = ti.time()-t1
print(" load train data time: %f seconds" %trun)

#================= TRAIN ELM =====================================================
t1 = ti.time()
if nr_neuroni==0:
    nb_in=0
if nb_in>0:
    inW, outW = elmTrain_fix(Samples, np.transpose(Labels2), nr_neuroni, C, tip, nb_in)
else:
    inW, outW = elmTrain_optim(Samples, np.transpose(Labels2), nr_neuroni, C, tip)
trun = ti.time()-t1
print(" training time: %f seconds" %trun)

print("model trained.saving model..")
np.save('./saved_data/no_neurons/outW_data2', outW)
np.save('./saved_data/no_neurons/inW_data2', inW)
print("model saved")

# ==============  Quantify the output layer ======================================
Qout=-1+pow(2,nb_out-1)
if nb_out>0:
     O=np.max(np.abs(outW))
     outW=np.round(outW*(1/O)*Qout)
     
#================= TEST (VALIDATION) DATASET LOADING
print('reading test data')
t1 = ti.time()
test_dataset_dir = './test20X20'
number_of_files = 3
image_test_data, target_test_data = read_training_data(test_dataset_dir,number_of_files)
print('reading test completed')
test_samples=image_test_data.astype('float32')
size_of_test_sample = len(test_samples)
test_labels=target_test_data
test_formatted_labels = np.zeros((size_of_test_sample,1), dtype=int)
counter_test = number_of_files-1;
value_of_label = 1;
for value in range(size_of_test_sample):
    test_formatted_labels[value] = value_of_label
    if counter_test == 0:
        counter_test = 2
        value_of_label = value_of_label+1
    else:
        counter_test=counter_test-1

test_formatted_labels=np.transpose(test_formatted_labels)
test_samples=np.transpose(test_samples)
clase_test=np.max(test_formatted_labels)+1
n=test_samples.shape[0]
N=test_samples.shape[1]
trun = ti.time()-t1
print(" load test data time: %f seconds" %trun)

#====================== VALIDATION PHASE (+ Accuracy evaluation) =================
t1 = ti.time()
scores = elmPredict_optim(test_samples, inW, outW, tip)
trun = ti.time()-t1
print( " prediction time: %f seconds" %trun)

# CONFUSION MATRIX computation ==================================
Conf=np.zeros((clase,clase),dtype='int16')
for i in range(N):
    # gasire pozitie clasa prezisa 
    ix=np.nonzero(scores[:,i]==np.max(scores[:,i]))
    ix_list = [x[0] for x in ix]
    pred=int(ix_list[0])
    actual=test_formatted_labels[0,i]-1
    Conf[actual,pred]+=1
accuracy=100.0*np.sum(np.diag(Conf))/np.sum(np.sum(Conf))
print("Confusion matrix is: ")
print(Conf)
print("Accuracy is: %f" %accuracy)
print( "Number of hidden neurons: %d" %nr_neuroni)
print( "Hidden nonlinearity (0=sigmoid; 1=linsat; 2=Relu; 3=ABS; 4=multiquadric 5=sign): %d" %tip)