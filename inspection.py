#author : Abhishek Basrithaya
#author andrew ID : abasrith

import sys
import numpy as np

args = sys.argv
# 2 inputs, 3 outputs and file names
assert(len(args) == 3)

# # # Parse every argument
train_input_path = args[1]
inspect_out_path = args[2]
# train_input_path = './small_train.tsv'
# inspect_out_path = './small_inspect.txt'

train_input = np.loadtxt(train_input_path, dtype='str')

# Training section
train_label_arr = train_input[1:, -1].astype(int)
unique_values, count = np.unique(train_label_arr, return_counts=True)
# print("unique", unique_values, "count", count)

#Majority vote classifier
majorityVoteIndex = 0
max_value = 0
if (len(count) > 0):
    for index in range(len(count)):
        if index == 0:
            max_value = count[index]
            majorityVoteIndex = index
        else:
            if count[index] > max_value:
                max_value = count[index]
                majorityVoteIndex = index
            elif count[index] == max_value:
                if unique_values[index] > unique_values[majorityVoteIndex]:
                    max_value = count[index]
                    majorityVoteIndex = index

majority_classifer_train_label = unique_values[majorityVoteIndex]
# print("majority_classifer_train_label", majority_classifer_train_label)

wrong_prediction_count = 0
for index in range(len(count)):
    if  index == majorityVoteIndex:
        pass
    else:
        wrong_prediction_count += count[index]
# print("wrong_prediction_count", wrong_prediction_count)
train_error = wrong_prediction_count/train_label_arr.size
train_error = "{:.6f}".format(train_error)
# print("train_error", train_error)

# Calculate entropy
entropy = 0
for x in count:
    probx = x/train_label_arr.size
    entropy += -(probx*np.log2(probx))
entropy = "{:.6f}".format(entropy)

# print("entropy", entropy)
# print entropy and train error metrics
with open(inspect_out_path, "w") as metrics_output:
    metrics_output.write("entropy: "+str(entropy)+"\n"+"error: "+str(train_error))