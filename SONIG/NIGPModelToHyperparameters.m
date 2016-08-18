function [hyp] = NIGPModelToHyperparameters(model)
%NIGPModelToHyperparameters will convert an NIGP model to a hyperparameter object.
% This function converts an NIGP model, with the seard and lsipn parameters, to a hyperparameter object, with lx, sx, ly, sy parameters.

hyp.lx = exp(model.seard(1:end-2,:));
hyp.sx = exp(model.lsipn);
hyp.ly = exp(model.seard(end-1,:)');
hyp.sy = exp(model.seard(end,:)');

end