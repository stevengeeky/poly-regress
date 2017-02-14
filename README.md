# poly-regress

A lightweight polynomial regressor created with TensorFlow.

## About

Visualize how deep learning works with a very easy-to-use piece of polynomial regression software. Most to all of modern deep learning techniques utilize the basic idea of tweaking coefficients in a polynomial in order to fit an approximated dataset to an actual one. This project takes that idea literally, and applies deep learning to fit a polynomial to a set of points.

## Usage

* Download this repository to a local directory or `git clone https://github.com/stevengeeky/poly-regress`
* Install tensorflow (`pip install tensorflow`)
* Install matplotlib (`pip install matplotlib`)
* Run poly.py
* As a test file use `test.txt` (contains an example set of points to regress a polynomial for)
* Answer 'n' to whether or not you want an exact solution
* Finished! You can watch the regression play out, and quantitatively see just how well the regressed result approximates the actual one in the final terminal output.

## Notice

* When inputting your own points, try to not make them too spread out, this can cause gradient descent to diverge.
* Having a learning rate which is too high can also cause divergence; you can change this in poly.py at the top of the file.
* A linear transformation is applied to the original set of points in order to make it possible to discover a solution for all set of points. You can visualize this with the transformation of the red line as regression takes place.

# Requirements
* Python 2 or 3, either should work just fine
* TensorFlow, latest release
* Matplotlib, latest release