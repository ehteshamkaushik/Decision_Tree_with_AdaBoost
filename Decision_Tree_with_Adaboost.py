import csv
from collections import OrderedDict, Counter
from operator import itemgetter
import math
import numpy
from numpy.random import choice
from sklearn.model_selection import train_test_split

filename = 'data1.csv'
attributes = {}
attribute_sequence = {}
continuous_attributes = []
unnecessary_attributes = []
missing_data_in_attributes = {}
classification_vals = {"Yes": +1, "No": -1}
class_column = 0
train = []
test = []
classification = []


def read_data_set(fname):
    with open(fname, newline='') as csvfile:
        examples = []
        attributes = {}
        attribute_sequence = {}
        reader = csv.DictReader(csvfile)
        counters = OrderedDict((attr, Counter()) for attr in reader.fieldnames)
        cnt = 0
        for row in reader:
            r = list(row.values())
            for i in unnecessary_attributes:
                index = reader.fieldnames.index(i)
                del r[index]
            examples.append(r)
            cnt += 1
            for attr, value in row.items():
                counters[attr][value] += 1
        for attr, counts in counters.items():
            d = dict(counts)
            l = list(d.keys())
            attributes[attr] = l
        for i in unnecessary_attributes:
            attributes.pop(i)
        for i in range(len(list(attributes.keys()))):
            attribute_sequence[list(attributes.keys())[i]] = i
    return examples, attributes, attribute_sequence


def clean_missing_classes(examples, m):
    index = len(examples[0]) - 1
    for e in examples:
        if e[index] == m:
            examples.remove(e)


def read_dataset_2():
    training_data = []
    cnt = 0
    with open("adult.data", newline='') as f:
        for line in f:
            cnt += 1
            training_data.append(line.replace(",", "").split(" "))

    print(cnt)
    print(len(training_data))

    c1 = 0
    c2 = 0

    for e in training_data:
        if e[len(e) - 1][0] == "<":
            e[len(e) - 1] = "No"
            c1 += 1
        else:
            e[len(e) - 1] = "Yes"
            c2 += 1
    print(len(training_data))

    print("No : ", c1, "Yes: ", c2)

    test_data = []
    with open("adult.test", newline='') as f:
        for line in f:
            test_data.append(line.replace(",", "").split(" "))

    c1 = 0
    c2 = 0
    for e in test_data:
        if e[len(e) - 1][0] == "<":
            e[len(e) - 1] = "No"
            c1 += 1
        else:
            e[len(e) - 1] = "Yes"
            c2 += 1
    print("No : ", c1, "Yes: ", c2)

    return training_data, test_data


def check_missing_data(examples, m):
    for i in attribute_sequence:
        index = attribute_sequence[i]
        cnt = 0
        missing_index = []
        for e in examples:
            if e[index] == m:
                missing_index.append(cnt)
            cnt += 1
        if len(missing_index) != 0:
            missing_data_in_attributes[i] = missing_index


def fill_continuous_attributes(examples, attribute, missing_index, m):
    index = list(attributes.keys()).index(attribute)
    cnt = 0
    mean = 0
    for i in range(len(examples)):
        if examples[i][index] != m:
            mean += float(examples[i][index])
            cnt += 1
    mean = mean / cnt
    for i in missing_index:
        examples[i][index] = str(mean)
        # print("filled Value at ", i, " ", parent_examples[i][index])


def fill_discrete_attributes(examples, attribute, missing_index, m):
    print(attribute)
    index = attribute_sequence[attribute]
    possible_values = attributes[attribute]
    possible_value_count = {}
    for p in possible_values:
        possible_value_count[p] = 0
    for i in range(len(examples)):
        if examples[i][index] != m:
            possible_value_count[examples[i][index]] += 1
    max_count = -1
    plurality_val = ""
    for p in possible_value_count:
        if possible_value_count[p] > max_count:
            max_count = possible_value_count[p]
            plurality_val = p
    for i in missing_index:
        examples[i][index] = plurality_val
        # print("filled Value at ", i, " ", parent_examples[i][index])


def fill_missing_data(examples, m):
    missing_attributes = list(missing_data_in_attributes.keys())
    print(missing_attributes)
    for i in missing_attributes:
        try:
            index = continuous_attributes.index(i)
            fill_continuous_attributes(examples, i, missing_data_in_attributes[i], m)
        except ValueError:
            fill_discrete_attributes(examples, i, missing_data_in_attributes[i], m)


def info_gain(less_than, greater_than):
    total_less = 0
    total_greater = 0
    for c in classification:
        total_less += less_than[c]
        total_greater += greater_than[c]

    total = total_greater + total_less
    less_entropy = 0
    greater_entropy = 0
    for c in classification:
        if less_than[c] != 0:
            less_entropy -= ((less_than[c] / total_less) * math.log(less_than[c] / total_less, 2))
        if greater_than[c] != 0:
            greater_entropy -= ((greater_than[c] / total_greater) * math.log(greater_than[c] / total_greater, 2))

    weighted_entropy = ((total_less / total) * less_entropy) + ((total_greater / total) * greater_entropy)
    return weighted_entropy


def binarize_data(examples):
    print(continuous_attributes)
    for attribute in continuous_attributes:
        index = attribute_sequence[attribute]
        # print(index)
        current_columns = []
        for e in examples:
            current_columns.append([float(e[index]), e[class_column]])
        sorted_current_columns = sorted(current_columns, key=itemgetter(0))
        count_vals = {}
        total_vals = {y: 0 for y in classification}
        for s in sorted_current_columns:
            count_vals[s[0]] = {y: 0 for y in classification}
        for s in sorted_current_columns:
            count_vals[s[0]][s[1]] += 1
            total_vals[s[1]] += 1
        split_vals = []
        for i in list(count_vals.keys()):
            split_vals.append(i)
        total_count = len(examples)
        min_entropy = 0
        for i in total_vals:
            if total_vals[i] != 0:
                min_entropy -= (total_vals[i] / total_count) * math.log((total_vals[i] / total_count), 2)
        split = split_vals[0]
        less_than_split_val = {y: 0 for y in classification}
        greater_than_split_val = total_vals
        for i in range(1, len(split_vals)):
            for c in classification:
                less_than_split_val[c] += count_vals[split_vals[i - 1]][c]
                greater_than_split_val[c] -= count_vals[split_vals[i - 1]][c]
            current_entropy = info_gain(less_than_split_val, greater_than_split_val)
            if current_entropy <= min_entropy:
                min_entropy = current_entropy
                split = split_vals[i]

        print(attribute)
        print(split)
        for e in examples:
            if float(e[index]) < split:
                e[index] = "No"
            else:
                e[index] = "Yes"
        attributes[attribute] = ["No", "Yes"]


def prepare_data_set_1():
    global class_column, classification, train, test, filename, attribute_sequence, attributes, \
        continuous_attributes, unnecessary_attributes
    attrs = []
    with open("attr.txt", newline='') as f:
        for line in f:
            attrs.append(list)
    continuous_attributes = ["tenure", "MonthlyCharges", "TotalCharges"]
    unnecessary_attributes = ["customerID"]
    filename = "data1.csv"
    examples, attributes, attribute_sequence = read_data_set(filename)
    clean_missing_classes(examples, " ")
    class_column = len(attributes.keys()) - 1
    classification = list(attributes.values())[class_column]
    attributes.pop("Churn")
    check_missing_data(examples, " ")
    fill_missing_data(examples, " ")
    binarize_data(examples)
    train, test = train_test_split(examples, test_size=0.2)
    print("Training data ", len(train))
    print("Test data ", len(test))


def prepare_data_set_2():
    global class_column, classification, train, test, filename, attribute_sequence, attributes, continuous_attributes, unnecessary_attributes
    continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    unnecessary_attributes = []
    train, test = read_dataset_2()

    attrs = []
    with open("attr.txt") as f:
        for line in f:
            attrs.append(line.replace("\n", ""))
    vals = []
    with open("pos_val.txt") as f:
        for line in f:
            vals.append(line.replace("\n", "").replace(",", "").split(" "))
    for i in range(len(attrs)):
        attributes[attrs[i]] = vals[i]
    attrs = []
    with open("total_attr.txt") as f:
        for line in f:
            attrs.append(line.replace("\n", ""))

    for i in range(len(attrs)):
        attribute_sequence[attrs[i]] = i

    arr = ["Yes", "No"]
    for c in continuous_attributes:
        attributes[c] = arr
    train.pop(32561)
    test.pop(16281)
    clean_missing_classes(train, "?")
    clean_missing_classes(test, "?")

    class_column = len(attributes.keys())
    classification = arr
    check_missing_data(train, "?")
    print(missing_data_in_attributes)
    fill_missing_data(train, "?")
    check_missing_data(test, "?")
    fill_missing_data(test, "?")

    binarize_data(train)
    binarize_data(test)
    print("Training data ", len(train))
    print("Test data ", len(test))


def prepare_data_set_3():
    global class_column, classification, train, test, filename, attribute_sequence, attributes, \
        continuous_attributes, unnecessary_attributes
    continuous_attributes = []
    unnecessary_attributes = ["Time"]
    filename = "creditcard.csv"
    examples, attributes, attribute_sequence = read_data_set(filename)

    continuous_attributes = list(attribute_sequence.keys())
    continuous_attributes.remove("Class")
    class_column = len(attributes.keys()) - 1
    reduced_example = []
    reduced_example_N = []
    reduced_example_Y = []
    count = 0
    for i in range(len(examples)):
        if examples[i][class_column] == "0":
            if count < 20000:
                examples[i][class_column] = "No"
                count += 1
                reduced_example.append(examples[i])
        else:
            examples[i][class_column] = "Yes"
            reduced_example.append(examples[i])
    attributes["Class"] = ["Yes", "No"]
    classification = list(attributes.values())[class_column]
    print(classification)
    attributes.pop("Class")
    binarize_data(reduced_example)
    for i in range(len(reduced_example)):
        if reduced_example[i][class_column] == "No":
            reduced_example_N.append(reduced_example[i])
        else:
            reduced_example_Y.append(reduced_example[i])
    train, test = train_test_split(reduced_example_N, test_size=0.2)
    count1 = int(len(reduced_example_Y) * 0.8)
    for i in range(0, count1):
        train.append(reduced_example_Y[i])
    for i in range(count1, len(reduced_example_Y)):
        test.append(reduced_example_Y[i])
    print(attributes)
    print("Training data ", len(train))
    print("Test data ", len(test))


class Node:
    def __init__(self, name, depth):
        self.name = name
        self.depth = depth
        self.children = {}
        self.classification = ""
        self.isLeaf = True


def plurality_value(examples):
    max_val = -1
    plural_class = ""
    pv = {}
    for c in classification:
        pv[c] = 0
    for e in examples:
        pv[e[class_column]] += 1
    for c in classification:
        if pv[c] > max_val:
            plural_class = c
            max_val = pv[c]
    return plural_class


def determine_classes(examples):
    data = {}
    for c in classification:
        data[c] = 0
    for example in examples:
        for c in classification:
            if example[class_column] == c:
                data[c] += 1
    return data


def is_same_class(data):
    keys = list(data.keys())
    cnt = []
    for k in keys:
        if data[k] > 0:
            cnt.append(k)
    return cnt


def calculate_initial_entropy(classes, total_count):
    entropy = 0
    classes_list = list(classes.values())
    for c in classes_list:
        if c > 0:
            x = (c / total_count)
            x = -x * math.log(x, 2)
            entropy += x
    return entropy


def calculate_info_gain(examples, attributes, parent_entropy):
    global classification
    best_attribute = ""
    possible_value_counts = {}
    possible_value_entropy = {}
    max_info_gain = -10000
    #examples_with_attributes = {}
    # print(classification)
    cnt = 0
    for a in attributes:
        index = attribute_sequence[a]
        # print(index_list)
        possible_values = list(attributes[a])
        # print(possible_values)
        #examples_with_attributes[a] = {}
        count = {}
        total = {}
        for p in possible_values:
            # print(p)
            count[p] = {}
            total[p] = 0
            #examples_with_attributes[a][p] = []
            for l in classification:
                count[p][l] = 0

        for e in examples:
            val = e[index]
            #examples_with_attributes[a][val].append(examples.index(e))
            cls = e[class_column]
            count[val][cls] += 1
            total[val] += 1

        # print(count)
        # print(total)
        entropy = {}
        expected_entropy = 0
        for p in possible_values:
            entropy[p] = calculate_initial_entropy(count[p], total[p])
            expected_entropy += (total[p] / len(examples) * entropy[p])
        # print(entropy)
        # print(expected_entropy)
        info_gain = parent_entropy - expected_entropy
        # print(info_gain)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            possible_value_counts = count
            possible_value_entropy = entropy
            best_attribute = a

    #best_attribute_index = examples_with_attributes[best_attribute]
    best_attribute_index = {}
    possible_values = list(attributes[best_attribute])
    for p in possible_values:
        best_attribute_index[p] = []
    index = attribute_sequence[best_attribute]

    for i in range(len(examples)):
        val = examples[i][index]
        best_attribute_index[val].append(i)

    return best_attribute, possible_value_entropy, possible_value_counts, best_attribute_index


def decision_tree_learning(examples, attributes, parent_examples, depth, entropy, max_depth):
    # print("1")
    root = Node("root", depth)
    curr_entropy = entropy
    if len(examples) == 0:
        root.classification = plurality_value(parent_examples)
        return root
    # print("2")
    classes = determine_classes(examples)
    # print("3")
    non_zero_classes = is_same_class(classes)
    # print("4")
    if len(non_zero_classes) == 1:
        root.classification = non_zero_classes[0]
        return root
    # print("5")
    if len(attributes) == 0 or depth == max_depth:
        root.classification = plurality_value(examples)
        return root
    else:
        # print("6")
        if depth == 0:
            curr_entropy = calculate_initial_entropy(classes, len(examples))
        # print("7")
        best_attribute, possible_value_entropy, possible_value_counts, best_attribute_index = calculate_info_gain(
            examples, attributes, curr_entropy)
        # print("8")
        root.isLeaf = False
        root.name = best_attribute
        for a in list(attributes[best_attribute]):
            new_parent_examples = examples
            new_examples = []
            for index in best_attribute_index[a]:
                new_examples.append(examples[index])
            new_keys = list(attributes.keys())
            new_keys.pop(new_keys.index(best_attribute))
            new_attributes = {}
            for k in new_keys:
                new_attributes[k] = attributes[k]
            new_depth = depth + 1
            new_entropy = possible_value_entropy[a]
            child = decision_tree_learning(new_examples, new_attributes, new_parent_examples, new_depth, new_entropy,
                                           max_depth)
            root.children[a] = child
        return root


def get_decision(tree, sample):
    if tree.isLeaf:
        decision = tree.classification
        return decision
    node = tree.name
    index = attribute_sequence[node]
    val = sample[index]
    new_tree = tree.children[val]
    decision = get_decision(new_tree, sample)
    return decision


def re_sample(examples, w):
    data = []
    n = len(examples)
    id = [i for i in range(n)]
    index_array = choice(id, n, p=w)
    for i in range(len(index_array)):
        j = index_array[i]
        data.append(examples[j])
    return data


def ada_boost(examples, K):
    n = len(examples)
    w = [(1 / n) for _ in range(n)]
    h = []
    z = []
    id = []

    for k in range(K):
        current_decision = []
        data = re_sample(examples, w)
        # print(data)
        # print(k)
        # print(len(h))
        temp_tree = decision_tree_learning(data, attributes, data, 0, 0, 1)
        error = 0.0
        for j in range(n):
            sample = examples[j]
            decision = get_decision(temp_tree, sample)
            current_decision.append(decision)
            if decision != sample[class_column]:
                error = error + w[j]
        if error > 0.5:
            # k -= 1
            continue
        h.append(temp_tree)
        for j in range(n):
            sample = examples[j]
            if current_decision[j] == sample[class_column]:
                w[j] = w[j] * (error / (1 - error))
        norm = numpy.linalg.norm(w, ord=1)
        w = w / norm
        z.append(math.log((1 - error) / error, 10))

    return h, z


def print_tree(tree):
    if tree.isLeaf:
        print(tree.classification)
        if tree.classification == "Yes":
            print("--------------------------------------------------------------------------------------")
        return
    print(tree.name)
    for c in list(tree.children):
        print("<" + c)
        print_tree(tree.children[c])
        print(c + "/>")


def ada_boost_decision(h, z, sample):
    decision = 0
    n = len(h)
    for i in range(n):
        d = get_decision(h[i], sample)
        decision += (z[i] * classification_vals[d])

    if decision >= 0:
        decision = 1
    else:
        decision = -1
    for c in classification_vals:
        if classification_vals[c] == decision:
            return c


def get_conditioned_value(examples):
    count_P = 0
    count_N = 0
    for e in examples:
        if e[class_column] == "Yes":
            count_P += 1
        else:
            count_N += 1
    return count_P, count_N


prepare_data_set_3()
p, n = get_conditioned_value(train)
print(p, n)
print("Original Tree :")
tree = decision_tree_learning(train, attributes, train, 0, 0, len(list(attributes.keys())))


# print_tree(tree)

def print_stat_training():
    print("Data set 2")
    print("Train")
    cp, cn = get_conditioned_value(train)
    print("Conditioned : ", cp, cn)
    true_p = 0
    true_n = 0
    predicted_p = 0
    false_p = 0
    for e in train:
        dec = get_decision(tree, e)
        if dec == "Yes":
            predicted_p += 1
        if dec == e[class_column]:
            if dec == "Yes":
                true_p += 1
            else:
                true_n += 1
        else:
            if dec == "Yes":
                false_p += 1
    accuracy = (true_p + true_n) / len(train) * 100
    print("Accuracy", accuracy)
    tpr = (true_p / cp) * 100
    print("True Positive Rate : ", tpr)
    tnr = (true_n / cn) * 100
    print("True Negative Rate : ", tnr)

    ppv = (true_p / predicted_p) * 100
    print("Positive Predictive Value : ", ppv)
    fdr = (false_p / predicted_p) * 100
    print("False Discovery Rate : ", fdr)
    f1 = 2 / ((1 / tpr) + (1 / ppv))
    print("F1 : ", f1)



def print_stat_test():
    print("Data set 2")
    print("Test")
    cp, cn = get_conditioned_value(test)
    print("Conditioned : ", cp, cn)
    true_p = 0
    true_n = 0
    false_p = 0
    predicted_p = 0
    for e in test:
        dec = get_decision(tree, e)
        if dec == "Yes":
            predicted_p += 1
        if dec == e[class_column]:
            if dec == "Yes":
                true_p += 1
            else:
                true_n += 1
        else:
            if dec == "Yes":
                false_p += 1
    accuracy = (true_p + true_n) / len(test) * 100
    print("Accuracy", accuracy)
    tpr = (true_p / cp) * 100
    print("True Positive Rate : ", tpr)
    tnr = (true_n / cn) * 100
    print("True Negative Rate : ", tnr)

    ppv = (true_p / predicted_p) * 100
    print("Positive Predictive Value : ", ppv)
    fdr = (false_p / predicted_p) * 100
    print("False Discovery Rate : ", fdr)
    f1 = 2 / ((1 / tpr) + (1 / ppv))
    print("F1 : ", f1)



def ada_boost_stat():
    for i in [5, 10, 15, 20]:
        print("adaboot train", i)
        h, z = ada_boost(train, i)
        c = 0
        cp, cn = get_conditioned_value(train)
        true_p = 0
        true_n = 0
        predicted_p = 0
        for e in train:
            dec = ada_boost_decision(h, z, e)
            if dec == "Yes":
                predicted_p += 1
            if dec == e[class_column]:
                if dec == "Yes":
                    true_p += 1
                else:
                    true_n += 1
        accuracy = (true_p + true_n) / len(train) * 100
        print("Accuracy : ", accuracy)

        print("adaboot test", i)
        cp, cn = get_conditioned_value(test)
        true_p = 0
        true_n = 0
        predicted_p = 0
        for e in test:
            dec = ada_boost_decision(h, z, e)
            if dec == "Yes":
                predicted_p += 1
            if dec == e[class_column]:
                if dec == "Yes":
                    true_p += 1
                else:
                    true_n += 1
        accuracy = (true_p + true_n) / len(test) * 100
        print("Accuracy : ", accuracy)


print_stat_training()
print_stat_test()
ada_boost_stat()
