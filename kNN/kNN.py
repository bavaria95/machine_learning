import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import re

def file_to_matrix(filename):
    fr = open(filename)
    number_of_lines = len(fr.readlines())
    return_matrix = zeros((number_of_lines, 3))
    class_label_vector = []

    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = re.split('\s+', line)
        return_matrix[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_matrix, class_label_vector

def create_data_set():
    groups = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels

def plot_data(x, y, z):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, 15*array(z), 15*array(z))
    plt.show()

def normalization(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    normed_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    normed_data_set = data_set - tile(min_vals, (m, 1))
    normed_data_set = normed_data_set/tile(ranges, (m, 1))
    return normed_data_set, ranges, min_vals

def euclidean_distance(a, b):
    return sum([(x - y)**2 for x, y in zip(a, b)])**0.5

def classify(inX, data_set, labels, k):
    distances = []
    for point in data_set:
        distances.append(euclidean_distance(inX, point))
    distances_indexes_sorted = sorted(range(len(distances)), key=lambda p: distances[p])[:k]
    class_count = {}
    for number_of_point in distances_indexes_sorted:
        votes_for_label = labels[number_of_point]
        class_count[votes_for_label] = class_count.get(votes_for_label, 0) + 1
    return max(class_count.iteritems(), key = operator.itemgetter(1))[0]

def dating_class_test():
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file_to_matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = normalization(dating_data_mat)
    m = norm_mat.shape[0]
    number_of_tests = int(m * ho_ratio)
    error_count = 0.0

    for i in range(number_of_tests):
        classifier_result = classify(norm_mat[i, :], norm_mat[number_of_tests:m, :], dating_labels[number_of_tests:m], 5)
        # print "the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1
    print 'total error rate ', error_count/float(number_of_tests)

# groups, labels = create_data_set()
# print classify([0, 0], groups, labels, 3)

dating_class_test()
# plot_data(dating_data_mat[:,1], dating_data_mat[:,2], dating_labels)
# print dating_data_mat
# print classify([12, 14, 200], dating_data_mat, dating_labels, 3)

