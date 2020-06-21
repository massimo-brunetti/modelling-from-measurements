%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This code developed by M.Brunetti - Politecnico di Milano in partial   % 
% fulfilment of "Modelling from Measurements" course #055461.            %
%                                                                        %
%                                                    Milano, June 2020   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% DMD reconstruction and forecast in cascade to Time Delay Embedding %%%

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

% Hankel matrix
d = 15; % max delay [0..m-1]

H=[];
for i = 1:(d+1)
    H = [H; X(:,i:m-(d+1)+i)]; 
end  
    
%% TDE-DMD ALGORITHM

H1 = H(:,1:end-1); % Hankel matrix (Koopman approximation)
H2 = H(:,2:end); % time shifted Hankel matrix
[U,Sigma,V] = svd(H1, 'econ'); % SVD decomposition

r = 11; % rank-truncation
U_r = U(:,1:r); % reduced left singular vector set
Sigma_r = Sigma(1:r,1:r); % reduced singular value set
V_r = V(:,1:r); % reduced right singular vector set

Atilde = U_r'*H2*V_r/Sigma_r; % DMD linear operator
[W,D] = eig(Atilde); % DMD eigenvalues & vectors
Phi = H2*V_r/Sigma_r*W; % DMD modes (unloaded)
lambda = diag(D); % DMD eigenvalues
omega = log(lambda)/dt; % DMD continuos eigenvalues 
y0 = Phi\H(:,1);  % DMD loadings (initial conditions)
frequencies = imag(omega)/(2*pi); % DMD frequencies (real)

%% TDE-DMD PREDICTION 

% Time-delay DMD reconstruction & forecast
X_modes = zeros(r,length(te)); 
for i = 1:length(te)
    X_modes(:,i) = (y0.*exp(omega*(i-1)));
end
X_dmd = Phi*X_modes;
X2_dmd = X_dmd (1:2,:); 

% Time-delay DMD error L2
error = zeros(1,length(t)); 
for i = 1:dt:length(t)
    error (i) = norm(X2_dmd(:,i)-X(:,i)); 
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
plot(te(1:ths),real(X2_dmd(1,1:ths)),'r',...
     te(1:ths),real(X2_dmd(2,1:ths)),'b',...
     te(ths:end),real(X2_dmd(1,ths:end)),'r-.',...
     te(ths:end),real(X2_dmd(2,ths:end)),'b-.');
title(['Time-delay DMD reconstruction and forecast, based on '...
       num2str(r) ' modes']);
xlabel('Time (year)');
ylabel('Population (x1000)');
legend({'Hares','Lynxes'});
grid ON;

subplot(2,2,3);
aux = plot(diag(Sigma)/sum(diag(Sigma)),'mo');
set(aux, 'markerfacecolor', get(aux, 'color'));
title(['Singular values of time-delay data up to '...
        num2str(d) ' steps']);
xlabel('Identifier')
ylabel('Value (normalized)')
grid ON;

subplot(2,2,4);
plot(real(omega),imag(omega),'m*');
title(['Continuous eigenvalues of time-delay data up to '...
        num2str(d) ' steps']);
xlabel('Real part');
ylabel('Imaginary part');
grid ON;

%plot(t,error,'m');
%title('Time-delay DMD fitting error');
%xlabel('Time [year]');
%ylabel('Deviation (per snapshot, L2)');
%grid ON;
%grid('minor');












