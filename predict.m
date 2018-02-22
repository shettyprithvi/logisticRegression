function p = predict(theta, X)

m = size(X, 1); 

p = zeros(m, 1);

predictions=X*theta;
p=sigmoid(predictions)>=0.5;

end
