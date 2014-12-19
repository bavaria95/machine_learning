from numpy import *
import operator

def create_data_set():
    groups = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels

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

def file_to_matrix(filename):
    fr = open(filename)
    number_of_lines = len(fr.readlines())
    return_matrix = zeros((number_of_lines, 3))
    class_label_vector = []

    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index, :] = list_from_line[0 : 3]
        class_label_vector.append(list_from_line[-1])
        index += 1
    return return_matrix, class_label_vector

# groups, labels = create_data_set()
# print classify([0, 0], groups, labels, 3)

dating_data_mat, dating_labels = file_to_matrix('datingTestSet.txt')