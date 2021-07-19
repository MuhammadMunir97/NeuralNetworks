import sys
import math
import copy

def gradient_descent_preceptron_learning(iterations_to_run, data, headers):
    total_attributes = len(data[0]) - 1
    weights = [0] * total_attributes
    for iteration in range(iterations_to_run):
        (x_vals, y_val) = get_vals_from_data(iteration, data)
        weight_dot_x = dot_product(weights, x_vals)
        sigmoid_weight_dot_x = sigmoid (weight_dot_x)
        sigmoid_derivative = sigmoid_weight_dot_x * (1 - sigmoid_weight_dot_x)
        error = y_val - sigmoid_weight_dot_x
        weights = weight_update_rule(weights, x_vals, error, sigmoid_derivative)
        output = get_output (weights, x_vals)
        print_to_console (iteration, weights, headers, output)

    return weights

def predict_output(weights, data, threshold):
    data_len = len(data)
    y_val_idx = len(data[0]) - 1
    total_correct = 0
    for row in data:
        x_vals = row [0:y_val_idx]
        actual_y_val = row [y_val_idx]
        prediction = 0
        dot = dot_product (weights, x_vals)
        sigmoid_dot = sigmoid (dot)
        if sigmoid_dot >= threshold:
            prediction = 1
        
        if prediction == actual_y_val:
            total_correct += 1

    return round((total_correct / data_len) * 100, 1) 


def print_to_console(iteration, weights, headers, output):
    string_to_print = "After iteration %s"%(iteration + 1) + ": "
    for idx, header in enumerate (headers):
        string_to_print += " w(" + header + ") ="
        string_to_print += " %s"% round (weights[idx], 4) + ", "

    string_to_print += " output = %s"% round (output, 4)
    print (string_to_print)

def get_vals_from_data(idx, data):
    data_len = len(data)
    y_val_idx = len(data[0]) - 1
    idx += 1
    if idx >= data_len:
        idx = idx % data_len

    idx -= 1
    return (data[idx][0:y_val_idx], data[idx][y_val_idx])


def get_output(vector_one, vector_two):
    dot = dot_product(vector_one, vector_two)
    return sigmoid(dot)

def dot_product(vector_one, vector_two):
    return sum([i*j for (i, j) in zip(vector_one, vector_two)])

def weight_update_rule(weights, x_vals, error, sigmoid_derivative):
    constant_to_multiply = error * sigmoid_derivative * learning_rate
    x_vals_copy = copy.deepcopy(x_vals)
    for idx, val in enumerate (x_vals_copy):
        x_vals_copy[idx] = val * constant_to_multiply

    return [x + y for x, y in zip(weights, x_vals_copy)]

def sigmoid (val):
    return 1 / (1 + math.exp(-val))

def convert_data_to_int(data):
    for row in data:
        for idx, field in enumerate (row):
            row [idx] = int (field)

if len (sys.argv) != 5:
    raise ValueError("Please provide four arguments")

training_set_file_path =  sys.argv [1]
test_set_file_path =  sys.argv [2]
learning_rate = float (sys.argv [3])
iterations_to_run = int (sys.argv [4])


training_set = [i.strip().split() for i in open(training_set_file_path).readlines()]
test_set = [i.strip().split() for i in open(test_set_file_path).readlines()]

# index for the class value, last index of the training set
class_val_idx = len (training_set[0]) - 1
headers = training_set [0][0:class_val_idx]

data = training_set [1:]
test_data = test_set [1:]

convert_data_to_int(data)
convert_data_to_int(test_data)

learned_weights = gradient_descent_preceptron_learning (iterations_to_run, data, headers)

training_accuracy = predict_output(learned_weights, data, 0.5)
test_accuracy = predict_output(learned_weights, test_data, 0.5)

print ("Accuracy on training set (%s"%len(data) + "): %s"%training_accuracy + "%")
print ("Accuracy on test set (%s"%len(test_data) + "): %s"%test_accuracy + "%")