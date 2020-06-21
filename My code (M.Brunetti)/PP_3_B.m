%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     This code developed by M.Brunetti, in partial fulfilment of the     % 
%     requirements of course "Modelling from Measurements" - #055461      % 
%                                                                         %
%                                               Politecnico di Milano     %
%                                                           June 2020     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Parameter fitting for LOTKA-VOLTERRA + discrepancy model by DMD %%%%%%  

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
   
%% DISCREPANCY DMD

% Discrepancy
DSC = X-Xnp';

% Hankel matrix
d = 15; 
H=[];
for i = 1:(d+1)
    H = [H; DSC(:,i:m-(d+1)+i)]; 
end  
    
% DMD algorithm
H1 = H(:,1:end-1); % Hankel matrix (Koopman approximation)
H2 = H(:,2:end); % time shifted Hankel matrix
[U,Sigma,V] = svd(H1, 'econ'); 

r = 6; % rank-truncation
U_r = U(:,1:r); % reduced left singular vector set
Sigma_r = Sigma(1:r,1:r); % reduced singular value set
V_r = V(:,1:r); % reduced right singular vector set

Atilde = U_r'*H2*V_r/Sigma_r; % DMD linear operator
[W,D] = eig(Atilde); % DMD eigenvalues & vectors
Phi = H2*V_r/Sigma_r*W; % DMD modes (unloaded)
lambda = diag(D); % DMD eigenvalues
omega = log(lambda)/dt; % DMD continuos eigenvalues 
y0 = Phi\H(:,1); % DMD loadings (initial conditions)
frequencies = imag(omega)/(2*pi); % DMD frequencies (real)

% DMD reconstruction 
X_modes = zeros(r,length(t)); 
for i = 1:length(t)
    X_modes(:,i) = (y0.*exp(omega*(i-1)));
end
X_dmd = Phi*X_modes;
X2_dmd = X_dmd (1:2,:); 
X3 = Xnp' + X2_dmd;

%% RESULTS PLOTTING

subplot(2,2,1)
plot(t,X(1,:),'r',t,X(2,:),'b');
title('Hares and Lynxes: historical records');
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,2)
plot(t,real(X3(1,:)),'r',t,real(X3(2,:)),'b');
title(['Hares and Lynxes: Lotka-Volterra + discrepancy DMD,'...
       ' based on ' num2str(r) ' modes']);
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,3);
aux = plot(diag(Sigma)/sum(diag(Sigma)),'mo');
set(aux, 'markerfacecolor', get(aux, 'color'));
title(['Singular values of time-delay discrepancy DMD, up to '...
        num2str(d) ' steps']);
xlabel('Identifier')
ylabel('Value (normalized)')
grid ON;

subplot(2,2,4);
plot(real(omega),imag(omega),'m*');
title(['Continuous eigenvalues of time-delay discrepancy DMD, up to '...
        num2str(d) ' steps']);
xlabel('Real part');
ylabel('Imaginary part');
grid ON;

