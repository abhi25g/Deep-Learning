#part1
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

#part2    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#part3
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print("train_set_x_orig: " + str(train_set_x_orig.shape[0]))
print("train_set_x_orig: " + str(train_set_x_orig.shape[1]))
print("train_set_x_orig: " + str(train_set_x_orig.shape[2]))
print("train_set_x_orig: " + str(train_set_x_orig.shape[3]))


#part4
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#part5
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
#print ("train_set_y shape: ", train_set_y.shape)
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("classes: " + str(classes))
print ("(train_set_y[0,26]): " + str(train_set_y[0][26]))

#part6
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1 ).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1 ).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#part7
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
print(train_set_x.shape)

#part8
def sigmoid(z):
    s = 1.0/(1.0+np.exp(-z))
    return s

#part9
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

#part10
def initialize_with_zeros(dim):
   
    w = np.zeros((dim, 1))
    b = 0.0
   
    assert(w.shape == (dim, 1))
    print("w.shape: ", w.shape)
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

#part11
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))

#part12
a = np.array([[1, 2, 3], [4, 5, 6]])
print("axis = 1", np.sum(a, axis = 1))
print("axis = 0", np.sum(a, axis = 0))
print("a.shape: ", a.shape)
b = np.array([[2,3]]).reshape(2,1)
print("b.shape: ", a.shape)
print("sigmoid a "+ str(sigmoid(a)))
c = np.array([1,2,3])
print("c:shape ", c.shape, c.reshape(1,3).shape, c.reshape(3,1).shape)

#part13
def propagate(w, b, X, Y):

    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1.0/m)*np.sum((Y*np.log(A)+ (1-Y)*np.log(1-A)), axis = 1)
    print("shape of cost: " + str(cost.shape))
    print("cost = ", cost)
    print("shape of X: " + str(X.shape)) #extra code
    print("shape of Y: " + str(Y.shape)) #extra code
    print("shape of w: " + str(w.shape)) #extra code
    dw = (1.0/m)*np.dot(X, (A-Y).T)
    print("shape of dz = (A-Y): " + str((A-Y).shape)) #extra code
    print("shape of dz[0] = (A-Y)[0]: " + str(((A-Y)[0]).shape)) #extra code
    print("Note that dz[0] selects the entire first row, hence a row vector of size m")
    print("shape of dw: " + str(dw.shape)) #extra code
    db = (1.0/m)*np.sum(A-Y, axis = 1)
    print("shape of db: " + str(db.shape))
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

#part14
X = np.array([[1.,2.,-1.],[3.,4.,-3.2]])
dz = np.array([1,2,3]).reshape(3,1)
print(X, "\n", dz, "\n", np.dot(X,dz))

#part15
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

#part16
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y) 
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

#part17
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

#part18
def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    print("A = ", A)
    p = np.zeros(m).reshape(1,m)
    for i in range(A.shape[1]):
        
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

#part19
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))

#part20
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    w, b = np.zeros(X_train.shape[0]).reshape(X_train.shape[0],1), 0.0

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

#part21
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


#part26
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index])+ ", you predicted that it is a \""+ classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")


#part27
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

#part28
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

#part29
my_image = "my_image2.jpg"

fname = my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")