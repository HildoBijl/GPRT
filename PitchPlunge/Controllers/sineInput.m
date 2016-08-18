function [out] = sineInput(h, alpha, hDot, alphaDot, U, t, reward)
%sineInput This function applies a sine input to the Pitch Plunge model.
%   This Matlab function applies a sine input to the Pitch Plunge model. The frequency of this sine is set by the global sineFrequency parameter and the amplitude by sineAmplitude.

global sineFrequency sineAmplitude;

out = sineAmplitude*sin(2*pi*sineFrequency*t);

end

