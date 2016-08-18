function [model] = hyperparametersToNIGPModel(hyp)
%hyperparametersToNIGPModel will convert a hyperparameter object to an NIGP model.
% This function converts a hyperparameter object, with lx, sx, ly, sy parameters to an NIGP model, with the seard and lsipn parameters.

model.seard = [log(hyp.lx);log(hyp.ly');log(hyp.sy')];
model.lsipn = log(hyp.sx);

end

