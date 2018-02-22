function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);
 
J = 0;
grad = zeros(size(theta));

predictions=X*theta;
thetaz=[0;theta(2:length(theta));];
J=(-(1/(m))* sum(y'*log(sigmoid(predictions))+(1-y)'*log(1-sigmoid(predictions))))+ (lambda/(2*m))*sum(thetaz.^2);
grad=(1/(m))*(X'*(sigmoid(predictions)-y))+(lambda/m)*thetaz;

end
