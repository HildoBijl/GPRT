function [out] = stateController(h, alpha, hDot, alphaDot, U, t, reward)
%stateController This function uses the state data to come up with a control signal. It's just a matter of tuning the gains and you're done.

% We start by obtaining all the settings defined in the defineControllerParameters file.
global hRef alphaRef hDotRef alphaDotRef hGain alphaGain hDotGain alphaDotGain;

% We now calculate the errors in the signals.
alphaError = alpha - alphaRef;
hError = h - hRef;
alphaDotError = alphaDot - alphaDotRef;
hDotError = hDot - hDotRef;

% Using the errors, we calculate the control signal.
beta = 0;
beta = beta - hGain*hError;
beta = beta - alphaGain*alphaError;
beta = beta - hDotGain*hDotError;
beta = beta - alphaDotGain*alphaDotError;

% We should tell the controller that beta is actually the function output.
out = beta;

% We save some stuff to the log file.
global cLogs cCounter;
cLogs(cCounter,:) = [out,hGain*hError,alphaGain*alphaError,hDotGain*hDotError,alphaDotGain*alphaDotError];

end