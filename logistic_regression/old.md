
In the context of machine and deep learning, univariate functions are nonexistent. In fact, some of the largest deep learning models can have ***billions*** of parameters, like the Bloom Language Model from Hugging Face, which has 176 billion parameters.

So what does gradient descent have to 

To answer these questions, let's imagine the development of a very simple machine learning model. We're going to start far away from gradient descent, but we'll end up there!

This simple model has one job: to predict if someone is tired. To make this prediction, the model can only use two pieces of data: a person's height, and the number of hours they've been awake. 

To generate a prediction, the model will multiply each feature by a parameter (commonly called a weight). It will then sum the two values and will will create a probability by compressing the sum between 0 and 1. If the output result ≥ 0.5, the model will guess that the person is tired. If the output < 0.5, the model will guess that the person is not tired.

In this example, we'll start the model off with random weights, say $\text{Weight}_{height} = -1.75$, and $\text{Weight}_{awake} = -1.75$. With these random parameters, the model likely won't be able to predict tiredness with any greater accuracy than just a 50:50 chance.

In order to accurately predict if a person is tired, the model must learn how important each input variable is to its prediction. In this example, **we** know the best parameter (the hours awake), but in reality we usually don't. How can we automatically 'teach' the model which parameter should have the greatest weight?

To tell the model what parameters weights will be *better* for it's prediction, we will use the model's guess to help train itself. If the model makes a *good* guess, we'll tell it's parameters to stay put - they are doing something write. However, if the model makes a *bad* guess, we'll tell ti. 
