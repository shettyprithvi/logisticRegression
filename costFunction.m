function [J, grad] = costFunction(theta, X, y)

m = length(y);


J = 0;
grad = zeros(size(theta));


predictions=X*theta;

J=-(1/(m))* sum(y'*log(sigmoid(predictions))+(1-y)'*log(1-sigmoid(predictions)));
grad=(1/(m))*(X'*(sigmoid(predictions)-y));


end
