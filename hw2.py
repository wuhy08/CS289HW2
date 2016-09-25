from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import time

NUM_CLASSES = 10 #:=K
SIGMA = 0.05 * np.pi
D = 500
#N = 60000
#P = 784

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    print X_train.shape #X is D*N
    xxt = X_train.dot(X_train.T) #D*D
    print("Finished X*X^T")
    A = xxt + reg*np.eye(xxt.shape[0]) #D*D
    print("Finished X*X^T + lambda*I")
    B = X_train.dot(y_train) #D*K
    print("Finished X*y")
    return scipy.linalg.inv(A).dot(B)#D*K

def train_gd(X, y, alpha=0.1, reg=0, num_iter=10000, ini = np.array([])):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    if not ini.any():
        newModel = np.zeros((X.shape[0], NUM_CLASSES))
        print 1
    else:
        newModel = ini
        print 2
    xxt = X.dot(X.T)
    xy = X.dot(y)
    for iteration in range(num_iter):
        print iteration
        oldModel = newModel
        gradient = xxt.dot(oldModel) - xy + reg * oldModel
        newModel = oldModel - alpha * gradient
    return newModel

    return np.zeros((X_train.shape[1], NUM_CLASSES))

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    return np.zeros((X_train.shape[1], NUM_CLASSES))

def one_hot(labels_train):
    '''Convert categorical labels 
    0,1,2,....9 to standard basis vectors in R^{10} '''
    y_train = np.zeros((X_train.shape[0], NUM_CLASSES))
    for line_index, label_train in enumerate(labels_train):
        y_train[line_index, label_train] = 1
    return y_train
    # y_train will return a nxk matrix, 
    #where n is the number of data points

def predict(model, X): #model is dxk, X is d*n
    ''' From model and data points, output prediction vectors '''
    Y = X.T.dot(model) #n*k matrix
    data_length = Y.shape[0]
    label_train = np.zeros(data_length)
    for data_index in range(data_length):
        label_train[data_index] = np.argmax(Y[data_index,:])
    return label_train

def phi(X, G, B):
    ''' Featurize the inputs using random Fourier features '''
    return np.cos(np.dot(G.T, X.T) + np.dot(B, np.ones((1, X.shape[0])))) #D*N

def genG(row, col):
    return np.random.normal(loc = 0.0, scale = SIGMA, size = (row, col)) #row * col

def genB(row):
    return np.random.uniform(low = 0.0, high = np.pi, size = (row, 1)) #row * 1



if __name__ == "__main__":
    print("Loading and lifting data")
    t = time.time()
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    P = X_train.shape[1]
    G = genG(P, D)
    b = genB(D)
    X_train, X_test = phi(X_train, G, b), phi(X_test, G, b)
    elapsed = time.time() - t
    print("Finished Loading and lifting data")
    print "Time cost: %f" % elapsed

    # print("Start Training, Closed Form")
    # t = time.time()
    # model_CF = train(X_train, y_train, reg=10)
    # elapsed = time.time() - t
    # print("Finished Training, Closed Form")
    # print "Time cost: %f" % elapsed
    # print("Start Predicting, Closed Form")
    # t = time.time()
    # pred_labels_train = predict(model_CF, X_train)
    # pred_labels_test = predict(model_CF, X_test)
    # elapsed = time.time() - t
    # print("Finished Prediction, Closed Form")
    # print("Closed form solution")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    print("Start Training, Gradient Descent")
    t = time.time()
    model = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=1000)
    print("Finished Training, Gradient Descent")
    print "Time cost: %f" % elapsed
    print("Start Predicting, Gradient Descent")
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    # model = train_sgd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=100000)
    # pred_labels_train = predict(model, X_train)
    # pred_labels_test = predict(model, X_test)
    # print("Stochastic gradient descent")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
