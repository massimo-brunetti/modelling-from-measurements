%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     This code developed by M.Brunetti, in partial fulfilment of the     % 
%     requirements of course "Modelling from Measurements" - #055461      % 
%                                                                         %
%                                               Politecnico di Milano     %
%                                                           June 2020     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% DMD reconstruction and forecast from assigned data set %%%%%%%%%%

%% DATA INITIALIZATION

clear all; close all; clc;

% time base
t1 = 1845;
t2 = 1903;
dt = 2;
t = (t1:dt:t2); % historical time base 
te = (t1:dt:t2+dt*length(t)); % extended time base 
ths = length(t); % reconstruction to forecast threshold

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

%% DMD ALGORITHM

X1 = X(:,1:end-1); % data matrix
X2 = X(:,2:end); % time shifted data matrix
[U,Sigma,V] = svd(X1, 'econ'); % SVD decomposition

r = length(Sigma); % rank-truncation
U_r = U(:,1:r); % reduced left singular vector set
Sigma_r = Sigma(1:r,1:r); % reduced singular value set
V_r = V(:,1:r); % reduced right singular vector set

Atilde = U_r'*X2*V_r/Sigma_r; % DMD linear operator
[W,D] = eig(Atilde); % DMD eigenvalues & vectors
Phi = X2*V_r/Sigma_r*W; % DMD modes (unloaded)
lambda = diag(D); % DMD eigenvalues
omega = log(lambda)/dt; % DMD continuos eigenvalues 
y0 = Phi\X(:,1);  % DMD loadings (initial conditions)

%% DMD PREDICTION

% DMD reconstruction & forecast
X_modes = zeros(r,length(te)); 
for i = 1:length(te)
    X_modes(:,i) = (y0.*exp(omega*(i-1)));
end
X_dmd = Phi*X_modes;

% DMD error L2
error = zeros(1,length(t)); 
for i = 1:dt:length(t)
    error (i) = norm(X_dmd(:,i)-X(:,i)); 
end

%% RESULTS PLOTTING

subplot(2,2,1);
plot(t,X(1,:),'r',t,X(2,:),'b');
title('Hares and Lynxes: historical records');
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,2);
plot(te(1:ths),X_dmd(1,1:ths),'r',...
     te(1:ths),X_dmd(2,1:ths),'b',...
     te(ths:end),X_dmd(1,ths:end),'r-.',...
     te(ths:end),X_dmd(2,ths:end),'b-.');
title(['DMD reconstruction and forecast, based on ' num2str(r) ' modes']);
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,3);
aux = plot(diag(Sigma)/sum(diag(Sigma)),'mo');
set(aux, 'markerfacecolor', get(aux, 'color'));
title('Singular values of raw data');
xlabel('Identifier')
ylabel('Variance (normalized)')
grid ON;

subplot(2,2,4);
plot(real(omega),imag(omega),'m*');
title('Continuous eigenvalues of raw-data');
xlabel('Real part');
ylabel('Imaginary part');
grid ON;












