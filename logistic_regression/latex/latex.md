
Logit: 
$$
p(Y|X) = \sigma (W \cdot X + b)
$$

Linear:
$$
p(Y|X) = \sigma (\underbrace{W \cdot X + b}_\text{Linear})
$$

Sigmoid:
$$
p(Y|X) = \underbrace{\sigma}_\text{Sigmoid}(W \cdot X + b) \\
\sigma = \frac{1}{1+e^{-X}} \\
$$

CE: 
$$
L_{CE}(Y, A) = \frac{1}{m}\sum^{n}_{i=0}-Y^{(i)}\log(A^{(i)}) - (1-Y^{(i)})(\log(1-A^{(i)})
$$

CE Loss Switch:
$$
loss = \begin{cases}
        Y^{(i)} = 0 & - (1 - Y^{(i)})\log(1-A^{(i)}) \\
        Y^{(i)} = 1 & - Y^{(i)} \log(A^{(i)})
    \end{cases}
$$

Chained:
$$
\begin{align*}
    \frac{dC}{dW} & = \frac{dC}{dA} \frac{dA}{dZ} \frac{dZ}{dW} &
    & & 
    \frac{dC}{db} & = \frac{dC}{dA} \frac{dA}{dZ} \frac{dZ}{db} \\
\\
    \frac{dC}{dA} & = - \frac{Y}{A} + \frac{1-Y}{1-A} &
     & & 
    \frac{dA}{dZ} & =  \sigma(Z)(1-\sigma(Z)) = A(1-A) \\ \\

    \frac{dZ}{dW} & = X &
    & & 
    \frac{dZ}{db} & = 1  \\ 
\end{align*}

$$

Simplified:

$$
\begin{align*}
    \frac{dC}{dA} \frac{dA}{dZ} \frac{dZ}{dW} & = (- \frac{Y}{A} + \frac{1-Y}{1-A})A(1-A)X \\
    & = (-\frac{A(Y-YA)}{A} + \frac{(1-A)(A-YA)}{(1-A)})X \\
    & = (-Y + YA + A - YA)X \\
    \frac{dC}{dW} &= (A - Y)X \\ 
    \\
    \therefore \frac{dC}{db} & = (A-Y)
\end{align*}
$$
