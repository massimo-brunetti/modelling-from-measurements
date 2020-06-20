%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This code developed by M.Brunetti - Politecnico di Milano in partial   % 
% fulfilment of "Modelling from Measurements" course #055461.            %
%                                                                        %
%                                                    Milano, June 2020   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% DMD on projected data (via SVD), then reconstruction and forecast %%%%

% [m,n,k] = size(BZ_tensor); % x vs y vs time data
% for j = 1:k
%     A = BZ_tensor(:,:,j);
%     pcolor(A), shading interp, pause(0.2)
% end

%%%%%%%%%%%%%%%%%%% file "BZ.mat" includes: BZ_tensor %%%%%%%%%%%%%%%%%%%%

%% DATA INITIALIZATION

clear all; close all; clc

load('BZ.mat');
[m,n,k] = size(BZ_tensor); % state (unA x unB), snapshots (um)

figure(11) % video
for j=1:40:k
    pcolor(BZ_tensor(:,:,j)), shading interp, pause(0.2)
end

rsc = 3; % BZ tensor scaling factor, for computational handling
u = BZ_tensor(1:floor(m/rsc),1:floor(n/rsc),1:floor(n/rsc));
[unA,unB,um] = size(u);

dt = 1; t = 0:dt:um-dt; 
dx = 1; x = 0:dx:unA-dx;
dy = 1; y = 0:dy:unB-dy;

val = 20; % data percentage, for forecast validation
ths = floor(length(t)*(1-val/100));

figure(1); 

subplot(2,2,1);
pcolor(u(:,:,1)), shading interp;
title(['Belousov-Zhabotinsky sol. at time = ' num2str(0)]);
xlabel('X');
ylabel('Y')

%% DATA RESHAPING AND SVD

u2 = reshape(u,[unA*unB um]); % data are sliced and cascaded
[U,Sigma,V] = svd(u2,'econ'); % singular values decomposition 

figure(1); 

subplot(2,2,3);
aux = plot(diag(Sigma)/sum(diag(Sigma)),'mo');
set(aux, 'markerfacecolor', get(aux, 'color'));
title('Singular values of reshaped solution');
xlabel('Identifier')
ylabel('Value (normalized)')
grid on;

r = 10; % truncation to first r singular values
Ur = U(:,1:r);
Sr = Sigma(1:r,1:r);
Vr = V(:,1:r);
VT = Vr';
pu2 = Ur'*u2; % let's apply DMD on this 

subplot(2,2,2);
plot(t(1:ths),pu2(:,1:ths),t(ths:end),pu2(:,ths:end),'k:');
title(['Reshaped, projected and truncated data up to ' num2str(r)]);
xlabel('Time')
ylabel('Amplitude')
grid on

%% DMD ALGORITHM  

X1 = pu2(:,1:ths-1); % data matrix
X2 = pu2(:,2:ths); % shifted data matrix

[U2,Sigma2,V2] = svd(X1, 'econ'); % SVD decomposition on projected data
Atilde = U2'*X2*V2/Sigma2; % DMD linear operator
[W,D] = eig(Atilde); % DMD eigenvalues & vectors

Phi = X2*V2/Sigma2*W; % DMD modes (unloaded)
lambda = diag(D); % DMD eigenvalues
omega = log(lambda)/dt; % DMD continuous eigenvalues
y0 = Phi\pu2(:,1);  % DMD loadings (initial conditions)
frequencies = imag(omega)/(2*pi); % DMD frequencies (real)

%% DMD PREDICTION

u_modes = zeros(r,length(t)); % original time base
for i = 1:length(t)
    u_modes(:,i) = (y0.*exp(omega*(i-1)));
end

pu2_dmd = Phi*u_modes; % DMD reconstruction 
u2_dmd = Ur*pu2_dmd; % de-projection
u_dmd = reshape(u2_dmd,[unA unB um]); % de-shaping

figure(1); 

subplot(2,2,4);
plot(t(1:ths),real(pu2_dmd(:,1:ths)),t(ths:end),real(pu2_dmd(:,ths:end)),'-.');
title('DMD reconstruction and forecast');
xlabel('Time')
ylabel('Amplitude')
grid on

%% RESULTS PLOTTING

figure(2); subplot(2,3,1)
pcolor(u(:,:,1)), shading interp;
title(['Belousov-Zhabotinsky sol. at time = ' num2str(0)]);
xlabel('X');
ylabel('Y');

figure(2); subplot(2,3,2)
pcolor(u(:,:,floor(um/2))), shading interp;
title(['Belousov-Zhabotinsky sol. at time = ' num2str(floor(um/2)*dt)]);
xlabel('X');
ylabel('Y');

figure(2); subplot(2,3,3)
pcolor(u(:,:,um)), shading interp;
title(['Belousov-Zhabotinsky sol. at time = ' num2str(um*dt)]);
xlabel('X');
ylabel('Y');

figure(2); subplot(2,3,4)
pcolor(real(u_dmd(:,:,1))), shading interp;
title(['NN reconstruction at time = ' num2str(0) ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y');

figure(2); subplot(2,3,5)
pcolor(real(u_dmd(:,:,floor(um/2)))), shading interp;
title(['NN reconstruction at time = ' num2str(floor(um/2)*dt) ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y');

figure(2); subplot(2,3,6)
pcolor(real(u_dmd(:,:,um))), shading interp;
title(['NN forecast at time = ' num2str(um*dt) ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y')