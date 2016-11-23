% This file prints the plot on the cover of the thesis. It does not use the regular Matlab plot functions but generates all the graphics pixel by pixel. It's more work, but it does make it a lot
% prettier. So for fancy plots, feel free to adjust this code to your own needs.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.

% We set up some important settings.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
filename = 'GPPlot.png'; % What should the file name be when saving?
screenDPI = 300; % Which resolution do we use when not exporting figures? This is in pixels per inch.
printDPI = 300; % Which resolution do we use when exporting figures?
imageWidth = 10; % How many centimeters should the image be wide? Everything else is calculated from this.

% We define plot dimensions.
xMin = -10; % This is the minimum value for x.
xMax = 10; % This is the maximum value for x.
yMin = xMin/3;
yMax = xMax/3;

% We set up the Gaussian process properties.
lf = 0.8; % This is the output length scale.
lx = 0.6; % This is the input length scale.

% We define the function that we will approximate with the GP.
sinPeriod = 6;
func = @(x)(sin(2*pi*x/sinPeriod) - ((x-3).^2/10^2 - 0.1));

% We define properties of the measurements.
nm = 40; % This is the number of measurements we will do.
sfh = 0.1; % This is the output noise scale.
rng(1, 'twister'); % We fix the random number generator so we always get the same results. Vary the seed to get different results.

% We define a few settings for how to plot the graphics.
alphaGPCoefficient = .6; % This says something about how light the GP is made and how quickly the blue fades out. Higher means faster fade out and less blue.
innerRadius = 0.01; % The radius of the part of the circles which is fully red.
outerRadius = 0.06; % The radius of the part of the circles which fades out.
decreaseExponent = 1.3; % The exponent at which the red fades out. Higher means more red.

% It's time to start calculating things! First of all, dimensions and such.
if exportFigs == 0
	dpi = screenDPI;
else
	dpi = printDPI;
end
cmpi = 2.54; % Number of centimeters per inch.
imageWidthInch = imageWidth/cmpi;
pixelsWidth = imageWidthInch*dpi;
pixelsHeight = ceil((yMax - yMin)/(xMax - xMin)*pixelsWidth);
pixelsWidth = ceil(pixelsWidth);

% We define colors for the eventual plot.
red = cat(3, ones(pixelsHeight, pixelsWidth), zeros(pixelsHeight, pixelsWidth), zeros(pixelsHeight, pixelsWidth));
green = cat(3, zeros(pixelsHeight, pixelsWidth), ones(pixelsHeight, pixelsWidth), zeros(pixelsHeight, pixelsWidth));
blue = cat(3, zeros(pixelsHeight, pixelsWidth), zeros(pixelsHeight, pixelsWidth), ones(pixelsHeight, pixelsWidth));
black = cat(3, zeros(pixelsHeight, pixelsWidth), zeros(pixelsHeight, pixelsWidth), zeros(pixelsHeight, pixelsWidth));

% We set up input points and measurement data.
Xs = xMin:(xMax - xMin)/(pixelsWidth - 1):xMax; % These are the trial points.
Xm = xMin + rand(1,nm)*(xMax - xMin); % These are the measurement points.
Xm = [-9.63, -9.45, -9.22, -8.15, -8.03, -7.19, -6.60, -6.27, -6.04, -4.65, -3.95, -3.73, -3.09, -2.06, -1.66, -1.16, 1.17, 1.98, 3.06, 3.41, 3.70, 3.84, 4.41, 5.00, 6.01, 6.69, 7.48, 7.89, 9.16, 9.37]; % We override the measurement points, to get pretty ones for the plot.
fmh = func(Xm)' + sfh*randn(size(Xm))'; % These are the measurement values, corrupted by noise.
f = linspace(yMin,yMax,pixelsHeight)'; % These are the y-coordinates we plot in. We don't use them for the GP but we do use them for the graphics.

% We now set up the (squared exponential) covariance matrix and related terms.
nm = size(Xm,2); % This is the number of measurement points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
Sfh = sfh^2*eye(nm); % This is the noise covariance matrix.
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression.
mPost = ms + Ksm/(Kmm + Sfh)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfh)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We use the GP result to calculate alpha values for every pixel.
gaussianDensity = @(y,m,v)(bsxfun(@rdivide,exp(-0.5*bsxfun(@rdivide,bsxfun(@minus,y,m').^2,v'))./sqrt(2*pi),sqrt(v')));
alphaGaussianDensity = gaussianDensity(f,mPost,diag(SPost));
alphaNormalized = alphaGaussianDensity/max(max(alphaGaussianDensity));
alphaGP = alphaNormalized.^alphaGPCoefficient;

% We calculate the image data for the measurement markers. So once more, we calculate the alpha values for every pixel.
alphaDots = zeros(pixelsHeight, pixelsWidth);
for i = 1:nm
	distance = bsxfun(@hypot, Xm(i) - Xs, fmh(i) - f); % Calculating the distances of each pixel with respect to the current measurement point.
	alphaCoefficient = bsxfun(@max, ((distance - innerRadius)/(outerRadius - innerRadius)), 0);
	alphaAddition = 1 - alphaCoefficient.^decreaseExponent; % Calculating the reduction in alpha.
	alphaDots = alphaDots + bsxfun(@max, alphaAddition, 0); % We add the non-negative entries.
end

% We set up the figure, layer by layer.
figure(1);
clf(1);
h = imshow(black); % Showing a black background.
set(h, 'AlphaData', ones(pixelsHeight, pixelsWidth)); % The background is fully opaque.
hold on;
h = imshow(blue); % Showing a blue foreground.
set(h, 'AlphaData', alphaGP); % Making the foreground transparent according to the GP alpha data.
h = imshow(red);  % Showing a red foreground.
set(h, 'AlphaData', alphaDots); % Making the foreground transparent according to the measurement dot data.

% If necessary, we save the figure. For this, we generate it again, but only mathematically. We do not plot it.
if exportFigs ~= 0
	% We set up the image to export.
	image = zeros(pixelsHeight, pixelsWidth, 4);
% 	image(:,:,4) = 1; % Set the alpha of the (black) background to zero to have a black background.
	
	% We add the GP layer over what we already have.
	newAlpha = image(:,:,4).*(1 - alphaGP) + alphaGP;
	image(:,:,1:3) = bsxfun(@rdivide, (bsxfun(@times, image(:,:,1:3), image(:,:,4).*(1 - alphaGP)) + bsxfun(@times, blue, alphaGP)), newAlpha);
	image(:,:,4) = newAlpha;
	
	% We add the dots layer over what we already have.
	newAlpha = image(:,:,4).*(1 - alphaDots) + alphaDots;
	image(:,:,1:3) = bsxfun(@rdivide, (bsxfun(@times, image(:,:,1:3), image(:,:,4).*(1 - alphaDots)) + bsxfun(@times, red, alphaDots)), newAlpha);
	image(:,:,4) = newAlpha;
	imwrite(image(:,:,1:3), filename, 'Alpha', image(:,:,4), 'ResolutionUnit', 'meter', 'XResolution', dpi/cmpi*100, 'YResolution', dpi/cmpi*100);
end
