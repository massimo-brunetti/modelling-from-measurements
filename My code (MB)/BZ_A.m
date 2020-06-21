%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This code developed by M.Brunetti - Politecnico di Milano in partial   % 
% fulfilment of "Modelling from Measurements" course #055461.            %
%                                                                        %
%                                                    Milano, June 2020   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NN trainining on data temporal modes (via SVD), then prediction %%%%%%

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

%% DATA RESHAPE AND SVD

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
VT = Vr'; % let's apply DMD on this 
pu2 = Ur'*u2;

subplot(2,2,2);
plot(t(1:ths),VT(:,1:ths),t(ths:end),VT(:,ths:end),'k:');
title(['Reshaped sol. temporal modes up to ' num2str(r)]);
xlabel('Time')
ylabel('Amplitude')
grid on

%% NN TRAINING

input = VT(:,1:end-1); 
output = VT(:,2:end);

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input,output);

%% NN PREDICTION

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

figure(1);

subplot(2,2,4);
plot(t(1:ths),VT_NN(:,1:ths),t(ths:end),VT_NN(:,ths:end),'-.');
title('NN reconstruction and forecast');
xlabel('Time')
ylabel('Amplitude')
grid on

%% RESULTS PLOTTING

figure(2); 

subplot(2,3,1)
pcolor(u(:,:,1)), shading interp;
title(['Belousov-Zhabotinsky sol. at time = ' num2str(0)]);
xlabel('X');
ylabel('Y');

subplot(2,3,2)
pcolor(u(:,:,floor(um/2))), shading interp;
title(['Belousov-Zhabotinsky sol. at time = ' num2str(floor(um/2)*dt)]);
xlabel('X');
ylabel('Y');

subplot(2,3,3)
pcolor(u(:,:,um)), shading interp;
title(['Belousov-Zhabotinsky sol. at time = ' num2str(um*dt)]);
xlabel('X');
ylabel('Y');

subplot(2,3,4)
pcolor(u_NN(:,:,1)), shading interp;
title(['NN reconstruction at time = ' num2str(0) ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y');

subplot(2,3,5)
pcolor(u_NN(:,:,floor(um/2))), shading interp;
title(['NN reconstruction at time = ' num2str(floor(um/2)*dt) ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y');

subplot(2,3,6)
pcolor(u_NN(:,:,um)), shading interp;
title(['NN forecast at time = ' num2str(um*dt) ' based on ' num2str(r) ' modes']);
xlabel('X');
ylabel('Y')