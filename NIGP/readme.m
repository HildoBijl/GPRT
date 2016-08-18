% Code to run NIGP training
% 
% DEMO
x = linspace(-10,10,100)'; y = cos(x) + 0.05*randn(size(x));
x = x + 0.2*randn(size(x)); % Training set
xs = linspace(-10,10,1000)'; ys = cos(xs); xs = xs + 0.2*randn(size(xs)); % Test set
testNIGP(x,y,xs,ys);
% 
% This code runs in the latest version of Matlab. Users with older versions
% of Matlab may need to replace the tildes ('~'), representing ignored 
% output arguments, with dummy variables.
% 
% This code is only setup to run with the squared exponential kernel with
% automatic relevence determination, although the theory can be applied to
% any differentiable covariance function. In the same vein, this code learns
% a diagonal input noise matrix, i.e. the input dimensions are corrupted by
% independent Gaussian noise, although the theory can be applied to full
% covariance matrices.
%
% Two versions are provided, the version from the paper just considers
% derivatives of the posterior mean. The second version, suffixed with a
% 'u', also considers the uncertainty in the derivatives when calculating
% the referred variance. This is helpful if you have lots of test points in
% areas where you have few training points but has shown to be slightly
% more unstable (for currently unknown reasons). It does often have
% improved estimation of the input noise variance, however.
%
% Two prediction modes are also provided. The first uses the same method as
% in training - referring the input noise to the output via the slope of
% the posterior. The second, used by calling gprNIGP with the final argument
% set to 'gpm', uses the exact moment formulation as cited in the paper.
% Unfortunately neither method strictly outperforms the other so I have
% provided both for you to test on your own problem.
%
% To train the model call the 'trainNIGP' function, which will return a
% struct of hyperparameters. Optimisation uses the BFGS gradient descent 
% method. Predictions can then be made by directly calling gprNIGP with the
% test input locations. The prediction outputs are the mean and variance of
% the posterior at the test point. Training sets with more than 1000 points
% and 10 dimensions will take a while to train. Multiple output dimensions
% are automatically handled by simultaneously training one GP per output.
% The 'testNIGP' function is a handy, quick way to test NIGP on your
% dataset.
%
% The code calls 3 mex functions. tprod should automatically compile but
% the other two don't. It is worth compiling these functions as it will
% give a decent speed up. You will need to mex with 'largeArrayDims', 
% and BLAS and LAPACK paths.
%
% Any bugs, questions or feedback, please do send me an email. I would be
% interested to hear where NIGP works and where it fails.
%
% Andrew McHutchon, Dec 2011