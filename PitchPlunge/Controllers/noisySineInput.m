function [out] = noisySineInput(h, alpha, hDot, alphaDot, U, t, reward)
%noisySineInput This controller provides a sine input with some added randomness, switched after every set interval of time. It basically combines the sineInput and the randomController.

out = sineInput(h, alpha, hDot, alphaDot, U, t, reward) + randomController(h, alpha, hDot, alphaDot, U, t, reward);

end

