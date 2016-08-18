function [handle1,handle2] = makeGPPlot(fig, xp, mup, stdp)
%makeGPPlot Will generate a plot in the given figure frame for the given GP output.
% The makeGPPlot function makes a plot of the results of a GP regression algorithm. Required parameters are the following.
%	fig: The figure number which the plot should be in.
%	xp: The input points for the plot.
%	mup: The mean values at the plot points.
%	stdp: The standard deviation at the plot points.
% The xp, mup and stdp should all be vectors of the same size.
% As output, there will be two plot handles.
%	handle1: The handle of the mean line which is plotted.
%	handle2: The handle of the grey area representing the 95% uncertainty region.
% Either of these handles can be used for making a proper figure legend later on, if necessary.

% We want all given vectors to be row vectors. If they're not, we flip (transpose) them.
if size(xp,2) == 1
	xp = xp';
end
if size(mup,2) == 1
	mup = mup';
end
if size(stdp,2) == 1
	stdp = stdp';
end

% We make a plot of the result.
figure(fig);
clf(fig);
hold on;
grid on;
handle2 = patch([xp, fliplr(xp)],[mup-2*stdp, fliplr(mup+2*stdp)], 1, 'FaceColor', [1,1,1]*0.85, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
handle1 = plot(xp, mup, 'k-', 'LineWidth', 1); % We plot the mean line.
axis([min(xp), max(xp), floor(min(mup-2*stdp)), ceil(max(mup+2*stdp))]); % We set the plot bounds to the correct value.

end

