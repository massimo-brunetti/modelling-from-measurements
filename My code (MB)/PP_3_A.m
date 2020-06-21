%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This code developed by M.Brunetti - Politecnico di Milano in partial   % 
% fulfilment of "Modelling from Measurements" course #055461.            %
%                                                                        %
%                                                    Milano, June 2020   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Parameter fitting for LOTKA-VOLTERRA NON LINEAR DYNAMIC MODEL %%%%%%  

% xdot = (b-p*y)*x;
% ydot = (r*x-d)*y;

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

%% PARAMETERS FITTING & PREDICTION

% Initial guess, based on visual data analysis
T = 12; % Hare & Lynx population single complete cycle (1871-1883)
Hf1 = 36; % Hare free-grow while Lynx population at minimum (1861)
Hf2 = 150; % Hare free-grow while Lynx population still at minimum (1863)
%He = sum(x(14:19))/T; % Hare equilibrium over a single cycle (1871-1883)
%Le = sum(y(14:19))/T; % Lynx equilibrium over a single cycle (1871-1883)
He = sum(x(1:30))/60; % Hare equilibrium over the full record (1845-1903)
Le = sum(y(1:30))/60; % Lynx equilibrium over the full record (1845-1903)

b = log(Hf2/Hf1)/dt;
d = sqrt(2*pi/T)/b;
p = b/Le;
r = d/He;

% Paramater optimization based on objective function minimization
P0 = [b p r d]';
LotkaVolterra = @(t,x)([P0(1)*x(1)-P0(2)*x(1)*x(2); ...
                        P0(3)*x(1)*x(2)-P0(4)*x(2)]);
[~,Xn] = ode23(LotkaVolterra,t,X(:,1));

P = fminsearch(@PP_3_sub,P0);
LotkaVolterra = @(t,x)([P(1)*x(1)-P(2)*x(1)*x(2); ...
                        P(3)*x(1)*x(2)-P(4)*x(2)]);
[~,Xnp] = ode23(LotkaVolterra,t,X(:,1));
   
%% RESULTS PLOTTING

subplot(2,2,1)
plot(t,X(1,:),'r',t,X(2,:),'b');
title('Hares and Lynxes: historical records');
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,2)
plot(t,Xnp(:,1),'r',t,Xnp(:,2),'b');
title('Hares and Lynxes: Lotka-Volterra fitting (optimized)');
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,3);
bar(P0);
title('Lotka-Volterra parameters estimation (guess)');
xlabel('Parameters (b, p, r, d)');
ylabel('Values')
grid ON;

subplot(2,2,4);
bar(P);
title('Lotka-Volterra parameters estimation (optimized)');
xlabel('Parameters (b, p, r, d)');
ylabel('Values')
grid ON;
