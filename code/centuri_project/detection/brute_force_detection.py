from centuri_project.utils import processed_directory, evaluate_labels
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from numpy import linalg as LA
import operator


# inputs: 
#    sequence: time-series data
#    winSize: window size
#    step: iteration step
# output: windows
def slidingWindow(sequence,winSize,step):
    # Verify if the input is iterable.
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
    
    # Pre-compute number of chunks to emit
    numOfChunks = int(math.ceil((len(sequence)-winSize)/step))+1
#     print(numOfChunks)
    # Do the work
    windows=[]
    for i in range(0,numOfChunks*step,step):
        windows.append(sequence[i:i+winSize])
    return windows

def normalise_data(data):
    # scaling to unit length
    normalised_data=preprocessing.normalize(data.reshape(1,-1))
    return normalised_data[0]

# compute labels
# inputs: 
#    sequence: time-series data
#    winSize: window size
#    threshold: for an event
# output: labels
def comp_label(sequence,win_size,threshold):
    #slice the sequence to windows
    windows=slidingWindow(sequence,win_size,step=1)
    # print('There are {} windows'.format(len(windows)))
    #compute the mean of each window
    win_mean=[]
    for win in windows:
        # scaling to unit length
        processed_win=normalise_data(win)
        win_mean.append(np.mean(processed_win))

        # x_win=np.arange(0, len(processed_win), 1)
        # plt.plot(x_win,processed_win)
        # plt.show()
    # print('finish computing the means for each window')

    # labels=[]
    # for index in range(0,len(sequence)):
    #     win_label=max(0,(index-win_size))
    #     norm_seq=sequence[index]/LA.norm(windows[win_label])
    #     is_event=bool(win_mean[win_label]-norm_seq>threshold)
    #     labels.append(int(is_event))

    labels=[int(bool((win_mean[max(0,(index-win_size))]-sequence[index]/LA.norm(windows[max(0,(index-win_size))]))>threshold)) for index in range(0,len(sequence))]
    return labels

# for a sequence of successive 1s, find the peak and only label that time point as 1 and the rest as 0
def optimise_label(sequence,label):
    #indices of the elements with value 1
    indices = [i for i, x in enumerate(label) if x == 1]
    opt_labels=label
    if len(indices)>0:
        # print('indices: {}'.format(indices))
        len_=len(indices)-1

        slice_indices=[i+1 for i,x in enumerate(indices) if ((indices[min(i+1,len_)]-x)>1)]
        if len(slice_indices)==0 or slice_indices[0]!=0:
            slice_indices.insert(0, 0)
        slice_indices.append(len_+1)
        sliced_list=[]
        for j in range(len(slice_indices)-1):
            sliced_list.append(indices[slice_indices[j]:slice_indices[j+1]])
        # print('indices: {}'.format(indices))
        # print('sliced indices: {}'.format(sliced_list))
        
        for sublist in sliced_list:
            min_index, min_value = min(enumerate(sequence[sublist]), key=operator.itemgetter(1))
            # print('min_index: {} - min_value: {}'.format(min_index,min_value))
            min_index=min_index+sublist[0]
            zero_indices=[i for i in sublist if i!=min_index]
            for zero_ind in zero_indices:
                opt_labels[zero_ind]=0
    return opt_labels


#generate a grid for a range of window sizes and thresholds
#para_x=[min_x,max_x,num_x]
def generate_grid(para_thr,pata_win):
    x = np.linspace(para_thr[0], para_thr[1], para_thr[2])
    y = np.linspace(para_win[0], para_win[1], para_win[2],dtype=np.int32)
    x_thr, y_win = np.meshgrid(x, y)
    return x_thr, y_win


if __name__ == "__main__":

    #load the data
    data_file = processed_directory / "train.npz"
    data = np.load(data_file, allow_pickle=True)
    data_sweepX = data["signal_times"]  # for all trace the time for each value record (n x d matrix)
    data_signal_values = data["signals"]  # for all trace the value records (n x d matrix): the actual traces we need to deal with
    data_sampling_rates = data["sampling_rates"]  # for all trace the sampling rate in Hz (n sized vector)
    data_labels = data["labels"]  # for all trace, the event presence (as 1/0 vector), the event amplitude and the baseline (n x 3 x d cube of data)

    # the number of traces
    num_trace=data_sweepX.shape[0]
    # iterate the traces
    for ind_ in range(1): #change 1 to num_trace later
        ind_=2
        x=data_sweepX[ind_][4000:15000]
        signal=data_signal_values[ind_][4000:15000].astype(float)
        # sampling_rate=data_sampling_rates[ind_]
        bench=data_labels[ind_][0][4000:15000] #benchmark labels
        

        # #set the parameters(x: threshold; y:win_size)
        # #para_x=[min_x,max_x,num_x]
        para_thr=[0.001,0.001,1]
        para_win=[1250,1250,1]
        x_thr,y_win=generate_grid(para_thr,para_win)
        # plt.plot(x_thr, y_win, marker='.', color='k', linestyle='none')
        # plt.show()
        print('threshold: \n{}'.format(x_thr))
        print('window size: \n{}'.format(y_win))

        # for each pair (win_size, threshold), compute the false negative and false positive
        FNs=np.empty([para_win[2],para_thr[2]])
        FPs=np.empty([para_win[2],para_thr[2]])

        count_test=0
        for i in range(para_win[2]):
            for j in range(para_thr[2]):
                count_test=count_test+1
                # print('\ntest {}'.format(count_test))
                thr=x_thr[i][j]
                win_size=int(y_win[i][j])             
                # print('win_size: {} - threshold: {}'.format(win_size,thr))

                # mark the currents         
                labels=comp_label(signal,win_size,thr)
                # print('finish computing labels')
                # print('labels: {}'.format(labels))               

                # #optimise the labels. for  continuous '111', find the peak
                opt_labels=optimise_label(signal,labels)
                # print('finish optimising the computed labels')
                # print('optLab: {}'.format(opt_labels))

                comp_optLabels= np.asarray(opt_labels).astype(np.bool)
                f, ax = plt.subplots()
                ax.plot(x, signal, color="grey", zorder=-2, label="raw data")
                ax.scatter(x[comp_optLabels], signal[comp_optLabels],edgecolors="red", facecolors="none", s=80, zorder=1, label="computed labels")
                bench_labels=bench.astype(np.bool)
                ax.scatter(x[bench_labels], signal[bench_labels], marker="x", color="blue", s=80, zorder=1, label="manual labels")
                plt.title('Evaluation of the brute-force algorithm')
                plt.legend()
                plt.show()

                #Evaluate the computed labels
                FN,FP = evaluate_labels(opt_labels, bench, win_size=10)
                FNs[i][j]=FN
                FPs[i][j]=FP
                # print('false negative: {} - false positive: {}'.format(FN, FP))

        # print(num_diffs)   
        # flat_FNs= FNs.flatten()
        # flat_FPs= FPs.flatten()
        # xf=np.arange(0, len(flat_FNs), 1)
        # plt.plot(xf,flat_FNs,'or',label="false negative")
        # plt.plot(xf,flat_FPs, 'g^',label="false positive")
        # plt.legend()
        # plt.show()
        print(FNs)
        print(FPs)    

