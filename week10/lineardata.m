clear all;
close all;
%% 
% Enter the measured data points by hand
x=[-10 -9 -8 -7 -6 -5 -4 -3 0];
y = [2.65 2.10 1.90 1.40 1.00 0.80 0.60 0.30 0.00];
ey = [0.1 0.1 0.1 0.1 0.05 0.05 0.05 0.05 0.2];
% Plot the data with error bars figure(1
errorbar(x,y,ey,'b.')
% Don?t forget the labels
xlabel('x (mm)')
ylabel('y (mm)')
axis equal

%
%
hold on
[m,b,merr,berr] = WeightedLSQFit(x, y, ey);
m
merr
b
berr
hold off
%%
%Do something in a second figure window.
%f igure(5)
%plot(x,x.^2)