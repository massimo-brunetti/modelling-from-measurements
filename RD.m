%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This code developed by M.Brunetti - Politecnico di Milano in partial   % 
% fulfilment of "Modelling from Measurements" course #055461.            %
%                                                                        %
%                                                    Milano, June 2020   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NN trainining on pre-calculated data, then prediction for testing %%%%

% lambda-omega reaction-diffusion system
%  u_t = lam(A) u - ome(A) v + d1*(u_xx + u_yy) = 0
%  v_t = ome(A) u + lam(A) v + d2*(v_xx + v_yy) = 0
%
%  A^2 = u^2 + v^2 and
%  lam(A) = 1 - A^2
%  ome(A) = -beta*A^2

%%%%%%%% file "reaction_diffusion_big.mat" includes: t, u, v, x, y %%%%%%%

%% DATA INITIALIZATION

clear all; close all; clc

load ('reaction_diffusion_big.mat')
dt = t(2)-t(1);
[unA,unB,um] = size(u); % state (unA x unB), snapshots (um)
val = 20; % raw data percentage, reserved for forecast validation
ths = floor(um*(1-val/100)); % training threshold 

figure(11) % video
for k = 1:10:um
    subplot(1,2,1)
    pcolor(x,y,u(:,:,k)); shading interp; colormap(hot)
    title(['RD sol: u component at time = ' num2str(k*dt-dt)]);
    xlabel('X');
    ylabel('Y');
    
    subplot(1,2,2)
    pcolor(x,y,v(:,:,k)); shading interp; colormap(hot)
    title(['RD sol: v component at time = ' num2str(k*dt-dt)]);
    xlabel('X');
    ylabel('Y');
    
    pause(0.02);
end

figure(1);
subplot(2,2,1);
pcolor(x,y,u(:,:,1)); shading interp; colormap(hot)
title(['RD sol: u component at time = ' num2str(0)]);
xlabel('X');
ylabel('Y');

%% SVD AND ROM

u2 = reshape(u,[unA*unB um]); % u sol. vectors ("slices") in cascade 
[U,Sigma,V] = svd(u2,'econ'); % singular values decomposition 

figure(1); 

subplot(2,2,3);
aux = plot(diag(Sigma)/sum(diag(Sigma)),'mo');
set(aux, 'markerfacecolor', get(aux, 'color'));
title('Singular values of reshaped u sol.');
xlabel('Identifier')
ylabel('Value (normalized)')
grid on;

r = 4; % truncation to first r singular values
Ur = U(:,1:r);
Sr = Sigma(1:r,1:r);
Vr = V(:,1:r);
VT = Vr';
pu2 = Ur'*u2;

subplot(2,2,2);
plot(t(1:ths),VT(:,1:ths),t(ths:end),VT(:,ths:end),'k:');
title(['Reshaped u sol. temporal modes up to ' num2str(r)]);
xlabel('Time')
ylabel('Amplitude')
grid on

%% NN TRAINING

input = VT(:,1:ths-1); 
output = VT(:,2:ths);

% train the NN
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input,output);

%% NN PREDICTION

% NN reconstruction 
x0 = VT(:,1);
VT_NN(:,1) = x0; 
for j = 2:um
    y0 = net(x0);
    VT_NN(:,j) = y0;
    x0 = y0; 
end

pu2_NN = Sr*VT_NN;
u2_NN = Ur*pu2_NN;
u_NN = reshape(u2_NN,[unA unB um]); 

figure(1); subplot(2,2,4);
plot(t(1:ths),VT_NN(:,1:ths),t(ths:end),VT_NN(:,ths:end),'-.');
title('NN reconstruction and forecast');
xlabel('Time')
ylabel('Amplitude')
grid on

%% RESULTS PLOTTING

figure(2); 

subplot(2,3,1)
pcolor(x,y,u(:,:,1)); shading interp; colormap(hot)
title(['Reaction-Diffusion u sol. at time = ' num2str(0)]);
xlabel('X');
ylabel('Y');

subplot(2,3,2)
pcolor(x,y,u(:,:,floor(um/2))); shading interp; colormap(hot)
title(['Reaction-Diffusion u sol. at time = ' num2str(floor(um/2)*dt)]);
xlabel('X');
ylabel('Y');

subplot(2,3,3)
pcolor(x,y,u(:,:,um)); shading interp; colormap(hot)
title(['Reaction-Diffusion u sol. at time = ' num2str(um*dt-dt)]);
xlabel('X');
ylabel('Y');

subplot(2,3,4)
pcolor(x,y,u_NN(:,:,1)); shading interp; colormap(hot)
title(['NN reconstruction at time = ' num2str(0)...
       ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y');

subplot(2,3,5)
pcolor(x,y,u_NN(:,:,floor(um/2))); shading interp; colormap(hot)
title(['NN reconstruction at time = ' num2str(floor(um/2)*dt)...
       ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y');

figure(2); subplot(2,3,6)
pcolor(x,y,u_NN(:,:,um)); shading interp; colormap(hot)
title(['NN forecast at time = ' num2str(um*dt-dt)...
       ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y')