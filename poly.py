"""
Simple Polynomial Regressor with TensorFlow
Also solves for exact solutions

by Steven O'Riley
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Number of times to attempt to fit the regressed polynomial to the data given
num_regression_steps = 20000

# Log progress (and redraw graph) after this number of iterations
log_every = 1000

# Gradient descent delta, if regression doesn't work, freezes, or the blue line isn't shown, push this value closer to 0
step = 0.001


# #####################################
# #####################################
# #####################################

# Just some important stuff
nptime = np.arange(0, 3, .05)
global_x = []
global_y = []

def main():
    """
    Initialization for polynomial regression each time the program is launched
    Input a file of points in the format #, # separated by lines
    """
    path = os.path.dirname(os.path.realpath(__file__)) + "/"
    basefilename = raw_input("Input a file for regression: %s" % path)
    filename = path + basefilename
    
    if not os.path.exists(filename):
        print("Error: No file discovered\n")
        main()
    else:
        print("File discovered! Attempting parse...")
        file = open(filename, "r")
        text = file.read()
        file.close()
        
        degree = 0
        points = []
        for line in text.split("\n"):
            if line.replace(" ", "") != "":
                points.append([ float(x) for x in line.split(",") ])
        
        terms = len(points)
        degree = terms
        
        if degree == 0 or terms == 0:
            print("Presented file is invalid!")
            main()
            return
        
        x  = np.zeros([terms, degree])
        y_ = np.zeros([terms])
        
        for i in range(terms):
            point = points[i]
            y_[i] = point[1]
            
            global_x.append(point[0])
            global_y.append(point[1])
            
            for n in range(degree):
                x[i, n] = pow(point[0], n)
        
        print(x, y_)
        
        if degree == terms:
            yes_no = raw_input("An exact solution for this polynomial is available, would you like to use that instead of regressing a solution? [Y/n] ")
            
            if "y" in yes_no.lower():
                solution = np.matmul(np.linalg.inv(x), y_)
                display_solution(x, y_, solution, degree, np.zeros([degree]))
                return
        
        regress(x, y_, terms, degree)

def regress(x, y_, rows, cols):
    """
    Regress a coefficient matrix from an input matrix x and an outupt matrix y'
    """
    
    # Pretty self-explanatory
    
    # Maybe I'll switch to a regular session eventually
    sess = tf.InteractiveSession()
    
    t_x  = tf.placeholder(tf.float32, shape=[rows, cols])
    t_y_ = tf.placeholder(tf.float32, shape=[rows])
    
    C = tf.Variable(tf.zeros([cols, 1]))
    b = tf.Variable(tf.zeros([cols]))
    
    tf.global_variables_initializer().run()
    
    t_y = tf.matmul(t_x, C) + b
    loss = tf.reduce_sum(tf.square(t_y - t_y_)) / cols
    train_step = tf.train.GradientDescentOptimizer(step).minimize(loss)
    
    plt.axis([0, 10, 0, 30])
    plt.ion()
    
    print("Regressing...")
    
    for i in range(num_regression_steps):
        if (i + 1) % log_every == 0:
            real_solution = np.matmul(np.linalg.inv(x), y_ - b.eval())
            
            plt.clf()
            plt.figure(1)
            plt.plot(nptime, f(nptime, C.eval(), cols), "b-")
            plt.plot(nptime, f(nptime, real_solution, cols), "r-")
            plt.plot(global_x, global_y - b.eval(), "ro")
            plt.pause(0.001)
            
            print( "%f" % ( float(i + 1) / num_regression_steps * 100 ) + "% " ),
            print( "loss: %f" % loss.eval(feed_dict={ t_x: x, t_y_: y_ }) )
        
        train_step.run(feed_dict={ t_x: x, t_y_: y_ })
    
    plt.close(1)
    plt.ioff();
    
    print("Finished.")
    
    display_solution( x, y_, C.eval(), cols, b.eval() )

def pearson(a, b):
    """
    Calculate the pearson correlation coefficient between two input lists/matrices/arrays a and b
    """
    x = np.array(a).flatten()
    y = np.array(b).flatten()
    
    xhat = np.mean(x)
    yhat = np.mean(y)
    
    return np.sum( (x - xhat) * (y - yhat) ) / np.sqrt( np.sum( (x - xhat) ** 2 ) ) / np.sqrt( np.sum( (y - yhat) ** 2 ) )

def f(time, C, degree):
    """
    Calculate f(x) - b for an input coefficient matrix C and a degree 'degree'
    """
    sum = np.zeros_like(time)
    coeff = np.array(C).flatten()
    
    for n in range(degree):
        sum += (time ** (n)) * coeff[n]
    return sum

def display_solution(x, y_, C, degree, b):
    """
    Displays a coefficient solution to the original point list
    """
    print(C)
    print(b)
    
    C = np.mat(C).reshape([-1, 1])
    predicted = np.array(np.matmul(x, C)).flatten()
    
    path = os.path.dirname(os.path.realpath(__file__)) + "/"
    filename = path + "output"
    
    np.save(filename + "_W", C)
    np.save(filename + "_b", np.mat(b))
    
    print("Results written to %s\nCorrelation of regression targets with original output: %f\nAverage squared error between regressed result and actual result: %f" % ( filename, pearson( predicted, y_ - b ), np.mean( np.square( y_ - b - predicted ) ) ))
    
    real_solution = np.matmul(np.linalg.inv(x), y_ - b)
    
    # Another way to grab C
    #raw = np.array(predicted).flatten()
    #regressed_solution = np.matmul( np.linalg.inv(np.mat([raw ** (n + 1) for n in range(degree)])), predicted - b )
    
    plt.figure(1)
    plt.clf()
    plt.plot(nptime, f(nptime, C, degree), "b-")
    plt.plot(nptime, f(nptime, real_solution, degree), "r-")
    
    plt.plot(global_x, global_y - b, "ro")
    plt.plot(global_x, np.array(predicted).flatten(), "bo")
    
    plt.show()

# Let's go
main()
