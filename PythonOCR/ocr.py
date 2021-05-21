#!/usr/bin/env python

DATA = 'ocrdata/' #filepath declarations
TEST_DATA = DATA + 't10k-images-idx3-ubyte'
TEST_LABELS = DATA + 't10k-labels-idx1-ubyte'
TRAIN_DATA = DATA + 'train-images-idx3-ubyte'
TRAIN_LABELS = DATA + 'train-labels-idx1-ubyte'

#frequently used function to convert byte_data type into integers
def parseToInt(b_data):
    return int.from_bytes(b_data, 'big')

#read and list every pixel in each image
def read_images(filename, max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4) #first 4 bytes of images can be ignored
        n_images = parseToInt(f.read(4))
        if max_images:
            n_images = max_images
        n_rows = parseToInt(f.read(4))
        n_columns = parseToInt(f.read(4))
        for image_index in range(n_images):
            image = []
            for row_index in range(n_rows):
                row = []
                for columns_index in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

#read values of the labels (known values of images)
def read_labels(filename, max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4) #first 4 bytes to be ignored
        n_labels = parseToInt(f.read(4))
        if max_labels:
            n_labels = max_labels
        for label_index in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels

def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist] #flatten imaages to one pixel

def extract_features(X):
    return [flatten_list(sample) for sample in X] #extract similarities on each flattened image

def dist(X, Y):
    return sum(
        [
            (parseToInt(X_i) - parseToInt(Y_i)) ** 2
            for X_i, Y_i in zip(X,Y)
        ]
    ) ** (0.5)


def get_distances_for_sample(X_train, test_sample): #get euclidian distances from sample images
    return [dist(train_sample, test_sample) for train_sample in X_train]

def knn(X_train, Y_train, X_test, Y_test, k=3): #k nearest neighbours algorithm to find similarities between neighbouring images
    Y_pred=[]
    for test_sample_index,  test_sample in enumerate(X_test):
        training_distances = get_distances_for_sample(X_train, test_sample)
        sorted_distance_indices = [ 
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda X: X[1]
            )
        ]
        candidates = [ #list of predicted characters
            parseToInt(Y_train[index])
            for index in sorted_distance_indices[:k]
        ]
        print(f'actual character was: {parseToInt(Y_test[test_sample_index])} predicted character was: {candidates}') #print actual character and character guess by the system
        Y_sample = 5
        Y_pred.append(Y_sample)
    return Y_pred #return predicted value



def main():
    X_train = read_images(TRAIN_DATA, 1000) #train by reading 10000 images
    Y_train = read_labels(TRAIN_LABELS)
    X_test = read_images(TRAIN_DATA,10) #attempt to recognise 10 characters
    Y_test = read_labels(TEST_LABELS)
    
    #print(len(X_train[0]))
    #print(len(Y_train))
    #print(len(X_test[0]))
    #print(len(Y_test))

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    knn(X_train,Y_train,X_test,Y_test,3)

    #print(len(X_train[0]))
    #print(len(Y_train))
    #print(len(X_test[0]))
    #print(len(Y_test))

if __name__=='__main__':
    main()
