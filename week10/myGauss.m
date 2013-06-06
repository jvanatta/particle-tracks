function y = myGauss(x,sigma) % myGauss(x,sigma) 
%	returns a Gaussian centered on zero with %	a width of sigma. 
% 
%	sigma is a single number %	x is an array of ordinate values 
%	the return values are an array evaluated at x.
% finally the function 
y = exp(-x.^2/sigma^2);
end