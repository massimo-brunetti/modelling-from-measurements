%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     This code developed by M.Brunetti, in partial fulfilment of the     % 
%     requirements of course "Modelling from Measurements" - #055461      % 
%                                                                         %
%                                               Politecnico di Milano     %
%                                                           June 2020     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% GOVERNING EQUATIONS EXTRAPOLATION (SYNDY) FOR THE FOLLOWING LIBRARY %%%

% x y x^2 x*y y^2

%% DATA INITIALIZATION

clear all; close all; clc;

% time base
t1 = 1845;
t2 = 1903;
dt = 2;
t = (t1:dt:t2); % historical time base 
te = (t1:dt:t2+dt*length(t)); % extended time base 

% hare status
xa = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70];
xb = [100 92 70 10 11 137 137 18 22 52 83 18 10 9 65]; 
x = [xa xb];

% lynx status
ya = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20];
yb = [34 45 40 15 15 60 80 26 18 37 50 35 12 12 25]; 
y = [ya yb];

% system status
X = [x; y];
[n,m] = size(X); % state dimension (n), number of snapshots (m)  

%% LINEAR SYSTEM COMPUTATION

% Library matrix computation
A = [x; y; x.^2; x.*y; y.^2]'; 
[ma,na] = size(A);
A = A(2:ma-1,:);
    
% B1 vector (center difference scheme)
xdot = zeros(1, m-2);
for i = 2:(m-1)
    xdot(i-1) = (x(i+1)-x(i-1))/(2*dt);
end
B1 = xdot';

% B2 vector (center difference scheme)
ydot = zeros(1, m-2);
for i = 2:(m-1)
    ydot(i-1) = (y(i+1)-y(i-1))/(2*dt);
end
B2 = ydot';

%% WEIGHTS FITTING & RECONSTRUCTION

xi1_L2 = pinv(A)*B1;
xi2_L2 = pinv(A)*B2;

Xi_L2 = [xi1_L2; xi2_L2];
Xi_L2s = Xi_L2;
Xi_L2s(abs(Xi_L2)<0.01) = 0;

%xi1_L2 = A\B1;
%xi2_L2 = A\B2;

xi1_L1 = lasso(A,B1,'Lambda',0.5);
xi2_L1 = lasso(A,B2,'Lambda',0.5);

Xi_L1 = [xi1_L1; xi2_L1];
Xi_L1s = Xi_L1;
Xi_L1s(abs(Xi_L1)<0.01) = 0;

LotkaVolterra = @(t,x)([Xi_L1s(2)*x(2); Xi_L1s(6)*x(1)]);
[~,Xn] = ode23(LotkaVolterra,t,X(:,1));

%% RESULTS PLOTTING

subplot(2,2,1)
plot(t,X(1,:),'r',t,X(2,:),'b');
title('Hares and Lynxes: historical records');
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,2)
plot(t,Xn(:,1),'r',t,Xn(:,2),'b');
title('Hares and Lynxes: SINDY sparse promoted extrapolation');
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,3);
bar(Xi_L1s(1:end/2));
title('Hares functional extrapolation by SINDY, using lasso (L1)');
xlabel('Functions (x, y, x^2, x*y, y^2)');
ylabel('Loadings')
grid ON;

subplot(2,2,4);
bar(Xi_L1s(end/2+1:end));
title('Lynxes functional extrapolation by SINDY, using lasso (L1)');
xlabel('Functions (x, y, x^2, x*y, y^2)');
ylabel('Loadings')
grid ON;