# Breaking it Down: Gradient Descent
Exploring and visualizing the mathematical fundamentals of gradient descent with [Grad-Descent-Visualizer][gdv].

https://gfycat.com/pitifulleftenglishpointer

**Outline**
1. [What is Gradient Descent?](#1-what-is-gradient-descent)
2. [Breaking Down Gradient Descent](#2-breaking-down-gradient-descent)
    1. [Computing the gradient](#21-computing-the-gradient)
    2. [Descending the gradient](#22-descending-the-gradient)
3. [Visualizing Multivariate Descents with Grad-Descent-Visualizer](#3-visualizing-multivariate-descents-with-grad-descent-visualizer)
    3.1 [Descent Montage](#31-descent-montage)
4. [Conclusion: Contextualizing Gradient Descent](#4-conclusion-contextualizing-gradient-descent)
5. [Resources](#5-resources)

### 1. What is Gradient Descent?
Gradient descent is an optimization algorithm that is used to improve the performance of deep/machine learning models. Over a repeated series of training steps, gradient descent identifies optimal parameter values that minimize model the cost function outputs.

In the next two sections of this post, we'll step down from this satellite-view and break down gradient descent into something that's a bit easier to understand.

https://gfycat.com/menacingdismalcowbird

### 2. Breaking Down Gradient Descent
To gain an intuitive understanding of gradient descent, let's first ignore machine and deep learning. Let's instead start with a simple function:

![Simple Function](/gradient_descent/latex/simple_function@2x.png)

The goal in gradient descent is to find the *minima* of a function, or the lowest possible output value of that function. This means that given our function ***f(x)***, the goal of gradient descent will be to find the value of ***x*** that leads the output of ***f(x)*** to approach ***0***. Below we visualize this function, it's quite obvious to see that ***x = 0*** produces our minima of ***f(x)***.

https://gfycat.com/achingmerryhoneybadger

The important part of this problem is: if we initialize ***x*** to some random number, say ***x = 1.8***, is there some way to automatically update ***x*** so that it eventually produces the minimal output of the function? This is essentially the goal in machine/deep learning gradient descent where we want to *automatically* find parameter weights in a model that will produce a minimal output from its cost function.

We can automatically find (or come close to) these minima with gradient descent in a two step process. 

1. First, we need to find the *slope* (i.e., *gradient*, hence gradient descent) of the function at the point where our input parameter ***x*** sits. 
2. Then, we need to *update* our input parameter ***x*** by telling it to take a step *down* the gradient.

This two step process repeated over and over until the output of our function stabilizes at a minima or it reaches a defined tolerance level.

##### 2.1. Computing the gradient
To find the slop (or *gradient*) of the function ***f(x)*** at any value of ***x***, we can differentiate* the function. Differentiating the simple example function is simple with the power rule (below), providing us with: ***f'(x) = 2x***.

![](/gradient_descent/latex/power_rule@2x.png)

Using our starting point ***x = 1.8***, we find our starting gradient to be ***dx = 3.6***.

Below let's write a simple function in python to automatically differentiate this function for us.

###### *I'd strongly recommend checking out [3Blue1Brown's video][3b1b] to intuitively understand differentiation. The differentiation of this sample function from first principals can be seen [here][socratic].

[3b1b]: https://www.youtube.com/watch?v=9vKqVkMQHKk&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr&index=2&t=2s
[socratic]: https://socratic.org/questions/how-you-you-find-the-derivative-f-x-x-2-using-first-principles


```python
def compute_gradient(x: float) -> float:
    """Compute the gradient of an input to the function f(x) = x**2.

    Args:
        x (float)

    Returns:
        float: dx
    """
    dx = 2 * x
    return dx

x = 1.8
dx = compute_gradient(x)
print(f"Gradient at x = 1.8: dx = {dx}")
```

    Gradient at x = 1.8: dx = 3.6


##### 2.2. Descending the gradient
Once we find the gradient of the starting point, we want to update our input parameter in such a way that will move it *down* this gradient so that the output of our function will be minimized.

To do this, we can simply subtract the gradient from the input parameter. But if you've looked closely, you may have noticed that subtracting the entire gradient from the input parameter ***x*** would cause it to infinitely bounce back and forth between ***1.8*** and ***-1.8***, never coming close to ***0***. 

Instead, we can define a ***Learning Rate = 0.1***. We'll use this learning rate to scale the gradient prior to subtracting it from our input parameter. Large learning rates produce large jumps along the function, and small learning rates lead to small steps along the function.

Lastly, we'll eventually have to stop the gradient descent, otherwise it would continue endlessly as it approaches 0. For this example, we'll simply stop the descent once the gradient of ***x***, ***dx***, is less than ***0.01***. In your own IDE, you can alter the `learning_rate` and `tolerance` parameters to see how the iterations and the output of ***x*** vary.


```python
def descend_gradient(x: float, learning_rate: float = 0.1) -> float:
    """Descends gradient of a point on the input function f(x) = x**2.

    Args:
        x (float)
        learning_rate (float): The rate by which the input parameter is updated.
        Defaults to 0.1.

    Returns:
        float
    """
    dx = compute_gradient(x)
    x -= dx * learning_rate  # step the input parameter 'down' the gradient
    return x

x = 1.8
iterations = 0
tolerance = 0.01
learning_rate = 0.1
while compute_gradient(x) > tolerance:
    x = descend_gradient(x, learning_rate=learning_rate)
    iterations += 1

print(f"Function minimum found in {iterations} iterations. X = {x:0.2f}")
```

    Function minimum found in 27 iterations. X = 0.00


As seen in the video above, our starting value of ***x = 1.8*** was able to automatically be updated to ***x = 0.0*** through the iterative process of gradient descent.

### 3. Visualizing Multivariate Descents with Grad-Descent-Visualizer
Hopefully this univariate example provided some foundational insight into what gradient descent actually does. Now let's expand to the context of multivariate functions. 

We'll first visualize a gradient descent of [Himmelblau's function][himmelblau].

[himmelblau]:https://en.wikipedia.org/wiki/Himmelblau%27s_function

![](/gradient_descent/latex/himmelblau@2x.png)

To visualize the descent of this landscape, we're going to initialize our starting parameters as ***x = -0.4*** and ***y = -0.65***. We can then watch the descent of each parameter in it's own dimension, sliced by the position of the opposite parameter.

https://gfycat.com/ifr/CourageousPotableLark

There are a few key differences for the descent of multivariate functions.

First, we need to compute *partial* derivatives in order to update each parameter. In Himmelblau's function, the gradient of ***x*** depends on ***y*** (their sums are squared, requiring the [chain rule][chain]). This means that the formula used to differentiate ***x*** will contain ***y***, and vice versa.

[chain]: https://g.co/kgs/8bwVeF

Second, you may have noticed that there was only one minima in the Section 2 simple function. In reality, there may be many unknown local minima in our models. This means that the local minima that our parameters find will depend on their starting point (initialization values) and the behavior of the gradient descent algorithm.

Now let's visualize the descent of the same point in 3D using my [grad-descent-visualizer][gdv] package created with the help of [PyVista][pyvista].

[gdv]: https://github.com/JacobBumgarner/grad-descent-visualizer
[pyvista]: https://github.com/pyvista/pyvista

https://gfycat.com/grandiosepowerlessfiddlercrab

### 3.1 Descent Montage
Now let's visualize the descent of some more functions! We'll place a grid of points across each of these functions and watch how the points move as they descend whatever gradient they are sitting on.

The [Sphere Function][sphere].

https://gfycat.com/dimpledfrayedhorsechestnutleafminer

The [Griewank Function][griewank].

https://gfycat.com/messypositivebluebreastedkookaburra

The [Six-Hump Camel Function][six-hump-camel]. Notice the many local minima of the function.

https://gfycat.com/pitifulleftenglishpointer

Let's re-visualize a gridded descent of the [Himmelblau Function][himmelblau]. Notice how different parameter initializations lead to different minima.

https://gfycat.com/cavernouscoarseflatfish

And lastly, the [Easom Function][easom]. Notice how many points sit still because they are initialized on a flat gradient.

https://gfycat.com/ablewigglyhellbender

[sphere]: https://www.sfu.ca/~ssurjano/spheref.html
[griewank]: https://www.sfu.ca/~ssurjano/griewank.html
[six-hump-camel]: https://www.sfu.ca/~ssurjano/camel6.html
[himmelblau]:https://en.wikipedia.org/wiki/Himmelblau%27s_function
[easom]: https://www.sfu.ca/~ssurjano/easom.html


### 4. Conclusion: Contextualizing Gradient Descent
So far we've worked through gradient descent with a univariate function and have visualized the descent several multivariate functions. In reality, modern deep learning models have ***vastly*** more parameters than what we've worked three here. For example, Hugging Face's newest natural language processing model, Bloom, has *175 billion* parameters.

*175 billion* parameters is *quite* a few more than what we've just looked at in this post. The chained functions used in this model are also certainly more complicated than our test functions.

However, it's important to realize that the *foundations* of what we've learned still apply. During each iteration of training of any deep learning model, the gradient of each parameter is calculated. This gradient will then be subtracted from the parameters so that they 'step down' their gradients, pushing them to produce a minimal output from the model's cost function.

Thanks for reading!

### 5. Resources
- [3Blue1Brown](https://www.youtube.com/c/3blue1brown)  
    - [Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)
    - [Derivatives](https://www.youtube.com/watch?v=9vKqVkMQHKk&t=10s)
- [Simon Fraser University: Test Functions for Optimization](https://www.sfu.ca/~ssurjano/optimization.html)
- [PyVista](https://docs.pyvista.org)
- [Michael Nielsen's Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)
