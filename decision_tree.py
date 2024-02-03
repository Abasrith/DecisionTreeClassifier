#author : Abhishek Basrithaya
#author andrew ID : abasrith

import argparse
import numpy as np

class Node:
    def __init__(self, featureSpace, labelSpace, Nodedepth):
        self.left = None
        self.right = None
        self.attr = None
        self.featureSpace = featureSpace
        self.labelSpace = labelSpace
        self.label = None
        self.node_depth = Nodedepth
        
class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def DecisionTreeTrain(self, featureSpace, labelSpace):
        self.root = self.growTree(featureSpace,  labelSpace, 0)

    def growTree(self, featureSpace, labelSpace, Depth):
      node = Node(featureSpace, labelSpace, Depth)
      #base case
      if((self.max_depth is None) or (Depth>=self.max_depth) or (len(np.unique(labelSpace[1:])) == 1) or (featureSpace.size == 0)):
        node.label = self.majorityVote(labelSpace)

      #else case
      else:
          mutual_info_arr = self.mutual_information(featureSpace, labelSpace)
          max_mutual_info = np.max(mutual_info_arr)
          if max_mutual_info > 0:
              max_mutual_info_indices = [index for index, value in enumerate(mutual_info_arr) if value == max_mutual_info]
              max_mutual_info_index = np.min(max_mutual_info_indices)

              node.attr = featureSpace[0,max_mutual_info_index]
              splitFeature = featureSpace[:,max_mutual_info_index]

              updatedLeftLabelSpace = labelSpace[splitFeature == '0']
              if updatedLeftLabelSpace.size == 0:
                pass
              else:
                updatedLeftLabelSpace = np.append(labelSpace[0], updatedLeftLabelSpace)

              updatedLeftFeatureSpace = featureSpace[splitFeature == '0']
              if(updatedLeftFeatureSpace.size == 0):
                  print("Error! Feature not viable!")
              else:
                updatedLeftFeatureSpace = np.vstack((featureSpace[0,:], updatedLeftFeatureSpace))
                updatedLeftFeatureSpace = np.delete(updatedLeftFeatureSpace, max_mutual_info_index, axis=1)

              updatedRightLabelSpace = labelSpace[splitFeature == '1']
              if updatedRightLabelSpace.size == 0:
                pass
              else:
                updatedRightLabelSpace = np.append(labelSpace[0], updatedRightLabelSpace)

              updatedRightFeatureSpace = featureSpace[splitFeature == '1']
              if (updatedRightFeatureSpace.size == 0):
                pass
              else:
                updatedRightFeatureSpace = np.vstack((featureSpace[0,:], updatedRightFeatureSpace))
                updatedRightFeatureSpace = np.delete(updatedRightFeatureSpace, max_mutual_info_index, axis=1)

              node.left = self.growTree(updatedLeftFeatureSpace, updatedLeftLabelSpace, node.node_depth+1)
              node.right = self.growTree(updatedRightFeatureSpace, updatedRightLabelSpace, node.node_depth+1)
          else:
              #majority vote
              node.label = self.majorityVote(labelSpace)
      return node

    def majorityVote(self, labelSpace):
      unique_label_vals, label_counts = np.unique(labelSpace[1:], return_counts = True)
      majorityVoteIndex = 0
      max_value = 0
      for index in range(len(label_counts)):
          if index == 0:
              max_value = label_counts[index]
              majorityVoteIndex = index
          else:
              if label_counts[index] > max_value:
                  max_value = label_counts[index]
                  majorityVoteIndex = index
              elif label_counts[index] == max_value:
                  if int(unique_label_vals[index]) > int(unique_label_vals[majorityVoteIndex]):
                      max_value = label_counts[index]
                      majorityVoteIndex = index

      return unique_label_vals[majorityVoteIndex]


    def entropy(self, DataSpace):
      unique_data_vals, data_counts = np.unique(DataSpace, return_counts = True)
      entropyVal = 0
      for x in data_counts:
          probx = x/len(DataSpace)
          if probx > 0:
              entropyVal += -(probx*np.log2(probx))

      return entropyVal

    def conditional_entropy(self, featureSpace, labelSpace):
      unique_feature_vals, feature_counts = np.unique(featureSpace, return_counts = True)
      p_x = feature_counts/len(featureSpace)

      conditionalEntropyPerFeatureVal = np.zeros(len(unique_feature_vals))
      for index, value in enumerate(unique_feature_vals):
          conditionalEntropyPerFeatureVal[index] = self.entropy(labelSpace[featureSpace == value])

      conditionalEntropy = 0
      for i in range(len(unique_feature_vals)):
          conditionalEntropy += (p_x[i]*conditionalEntropyPerFeatureVal[i])

      return conditionalEntropy


    def mutual_information(self, featureSpace, labelSpace):
      #Entropy of overall label space
      label_entropy = self.entropy(labelSpace[1:])

      num_features = featureSpace.shape[1]
      mutual_info_arr = np.zeros(num_features)

      for i in range(num_features):
          feature = featureSpace[1:, i]
          mutual_info_arr[i] = label_entropy - self.conditional_entropy(feature, labelSpace[1:])

      return mutual_info_arr

    def DecisionTreePredict(self, testSpace):
      featureSet = testSpace[0,:]
      predictions = []
      for x in testSpace[1:]:
        predictions.append(self.traceTree(featureSet, x, self.root))
      return predictions

    def traceTree(self, featureSet, x, node):
      if node.label == None:
        feature_index = np.where(featureSet == node.attr)
        if len(feature_index[0]) > 0:
            if x[feature_index[0]] == '0':
                return self.traceTree(featureSet, x, node.left)
            elif x[feature_index[0]] == '1':
                return self.traceTree(featureSet, x, node.right)
            else:
                print("Error, value of feature not binary!")
        else:
            print("Error, feature not found!")
      else:
          return node.label
        
def printTree(parent_attr, node, branchVal):
  if node is not None:
    #printing action
    if node.node_depth == 0:
      unique_node_label_vals, node_label_counts = np.unique(node.labelSpace[1:], return_counts = True)
      label_0_count = 0
      label_1_count = 0
      for i in range(len(unique_node_label_vals)):
        if unique_node_label_vals[i] == '0':
          label_0_count = node_label_counts[i]
        if unique_node_label_vals[i] == '1':
          label_1_count = node_label_counts[i]
      print(f"[{label_0_count} 0/{label_1_count} 1]")
      printTree(node.attr, node.left,0)
      printTree(node.attr, node.right,1)

    else:
      if (node.attr is not None) or (node.label is not None):
        #non leaf
        unique_node_label_vals, node_label_counts = np.unique(node.labelSpace[1:], return_counts = True)
        label_0_val_count = 0
        label_1_val_count = 0
        for i in range(len(unique_node_label_vals)):
          if unique_node_label_vals[i] == '0':
            label_0_val_count = node_label_counts[i]
          if unique_node_label_vals[i] == '1':
            label_1_val_count = node_label_counts[i]

        print('| '*node.node_depth, f"{parent_attr} = {branchVal}: [{label_0_val_count} 0/{label_1_val_count} 1]")
      printTree(node.attr, node.left,0)
      printTree(node.attr, node.right,1)              
        
if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    # parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    # parser.add_argument("max_depth", type=int, 
    #                     help='maximum depth to which the tree should be built')
    # parser.add_argument("train_out", type=str, 
    #                     help='path to output .tsv file to which the feature extractions on the training data should be written')
    # parser.add_argument("test_out", type=str, 
    #                     help='path to output .tsv file to which the feature extractions on the test data should be written')
    # parser.add_argument("metrics_out", type=str, 
    #                     help='path of the output .txt file to which metrics such as train and test error should be written')
    # args = parser.parse_args()
    
    # trainingSetInput = np.loadtxt(args.train_input, dtype='str')
    # testSetInput = np.loadtxt(args.test_input, dtype='str')
    # max_depth = int(args.max_depth)
    # trainingLabelOut = args.train_out
    # testingLabelOut = args.test_out
    # metricsOut  =   args.metrics_out
    
    trainingInputFile = './heart_train.tsv'
    testingInputFile = './heart_test.tsv'
    max_depth_input = 3
    trainingLabelOutputFile = './heart_train_label.txt'
    testingLabelOutputFile = './heart_test_label.txt'
    metricOutputFile = './heart_metrics.txt'
    
    trainingSetInput = np.loadtxt(trainingInputFile, dtype='str')
    testSetInput = np.loadtxt(testingInputFile, dtype='str')
    max_depth = max_depth_input
    trainingLabelOut = trainingLabelOutputFile
    testingLabelOut = testingLabelOutputFile
    metricsOut  =   metricOutputFile
    
    
    classifier = DecisionTree(max_depth)
    trainingFeatureSpace = trainingSetInput[:,:-1]
    trainingLabelSpace = trainingSetInput[:,-1]
    classifier.DecisionTreeTrain(trainingFeatureSpace, trainingLabelSpace)
    printTree(0, classifier.root, 0)
    # Retrieving labels for train and test
    trainingPredictions = classifier.DecisionTreePredict(trainingFeatureSpace)
    testingSpace = testSetInput[:,:-1]
    testingLabelSpace = testSetInput[:,-1]
    testingPredictions = classifier.DecisionTreePredict(testingSpace)
    
    # Calculating training and testing error
    trainingCorrectCount = 0
    for i in range(len(trainingPredictions)):
        if trainingPredictions[i] == trainingLabelSpace[i+1]:
            trainingCorrectCount += 1

    train_error = (len(trainingPredictions)-trainingCorrectCount)/len(trainingPredictions)
    train_error = "{:.4f}".format(train_error)
    print(f"train error with depth {classifier.max_depth}:", train_error)

    testingCorrectCount = 0
    for i in range(len(testingPredictions)):
        if testingPredictions[i] == testingLabelSpace[i+1]:
            testingCorrectCount += 1
    test_error = (len(testingPredictions)-testingCorrectCount)/len(testingPredictions)
    test_error = "{:.4f}".format(test_error)
    print(f"test error with depth {classifier.max_depth}:", test_error)

        
    # print train labels
    with open(trainingLabelOut, "w") as train_output:
        for i in range(len(trainingPredictions)):
            train_output.write(str(trainingPredictions[i])+"\n")

    # print test labels
    with open(testingLabelOut, "w") as test_output:
        for i in range(len(testingPredictions)):
            test_output.write(str(testingPredictions[i])+"\n")

    # print train and test metrics
    with open(metricsOut, "w") as metrics_output:
        metrics_output.write("error(train): "+str(train_error)+"\n"+"error(test): "+str(test_error))
    