# Learning for ML
I start this for practicing training models of machine learning and as
a note of what I've learnt in the courses([Stanford CS229: Machine Learning (Autumn 2018)](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)) on youtube.

Content
-----
The files follow the order of the courses with slight changes. Variables' name are chosen as
what they look like in theory as possible, which means most of them consist of Greek alphabeta and
some subscripts encoded in unicode. In the julia REPL, just type the LaTeX-like abbreviations and enter
Tab to let julia print the characters, which looks like:

```julia-repl
julia> \pi
```
and with Tab pressed
```julia-repl
julia> π
π = 3.1415926535897...

julia>
```

Some IDE for julia and extensions also support this. 

Data Format
----- 
Dataset here contains all samples of row vector in column, fitting the form:  

![](http://chart.googleapis.com/chart?cht=tx&chl=X%3d%5cbegin{bmatrix}%5ccdots%20x_1^T%5ccdots%20%5c%5c%5ccdots%20x_2^T%5ccdots%20%5c%5c%5cvdots%5c%5c%5ccdots%20x_m^T%5ccdots%20%5c%5c%5cend{bmatrix})

So do the labels. But to make matrix calculus easier, we just set samples in row:  

<center><img src=http://chart.googleapis.com/chart?cht=tx&chl=X=[x^1,x^2,%5ccdots,x^m]></img></center>

In this form, the calculation in loss function of NN would be written as:  

![](http://chart.googleapis.com/chart?cht=tx&chl=a^{[n]}=%5csigma%28w^{[n]}%28%5ccdots%5csigma%28w^{[2]}%5csigma%28w^{[1]}X%2bb^{[1]}%29%2bb^{[2]}%29%5ccdots%29%2bb^{[n]}%29)

where ![](http://chart.googleapis.com/chart?cht=tx&chl=a^{[n]}) is activation of layer,
![](http://chart.googleapis.com/chart?cht=tx&chl=w^{[n]}) is the weight matrix,
![](http://chart.googleapis.com/chart?cht=tx&chl=b^{[n]}) is bias.

The shape of ![](http://chart.googleapis.com/chart?cht=tx&chl=a^{[n]}) as well as
![](http://chart.googleapis.com/chart?cht=tx&chl=b^{[n]}) for one single sample here is
![](http://chart.googleapis.com/chart?cht=tx&chl=N_n), the number of neurons at layer n, by 1.
And the shape of ![](http://chart.googleapis.com/chart?cht=tx&chl=w^{[n]}) is
![](http://chart.googleapis.com/chart?cht=tx&chl=N_{n}) by ![](http://chart.googleapis.com/chart?cht=tx&chl=N_{n-1})

