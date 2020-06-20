%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This code developed by M.Brunetti - Politecnico di Milano in partial   % 
% fulfilment of "Modelling from Measurements" course #055461.            %
%                                                                        %
%                                                    Milano, June 2020   %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NN training on Lorenz trajectory window set and transition forecast %%%

%% DATA INITIALIZATION

clear all, close all
mr = 20; % number of trajectories for training 
mrr = 5; % number of trajectories for testing 
dt = 0.01; T = 10; t = 0:dt:T; % time basis
C1 = 4; % past segment window (centre) 
W1 = 2; % past segment window (width) 
C2 = 7:1:9; % future segment window (centre) 
W2 = 2; % future segment window (width) 

b = 8/3; sig = 10; r = 28;
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);  

for i = 1:length(C2)
    
    t1 = C1-W1/2:dt:C1+W1/2; 
    t2 = C2(i)-W2/2:dt:C2(i)+W2/2;
    
    input = [];
    output = [];

    for j = 1:mr  

        x0 = 30*(rand(3,1)-0.5); 
        [~,y] = ode45(Lorenz,t,x0);
        yx = y(:,1);
        yx1 = yx(floor((C1-W1/2)/dt):floor((C1+W1/2)/dt));
        yx2 = yx(floor((C2(i)-W2/2)/dt):floor((C2(i)+W2/2)/dt));
    
        %% LABELLING FOR SML 
        
        transition = -1;
        for k = 2:length(t2) 
            if yx2(k)*yx2(k-1)<0
               transition = 1;
            end
        end
           
        input = [input; yx1'];
        output = [output; transition];
    
        figure(i); 
        
        subplot(2,3,1);
        plot3(y(:,1),y(:,2),y(:,3)), hold on
        plot3(x0(1),x0(2),x0(3),'ro'), hold on
        title(['Lorenz 3D trajectories for training, rho = ' num2str(r)]);
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        view(-23,18)
        grid on;
    
        subplot(2,3,2);   
        plot(t,yx,'k:',t1,yx1,t2,yx2,'r'), hold on
        title(['Projections for supervised learning, ' num2str(mr) ' samples' ]);
        legend({'ODE','Sampling','Labelling'},'Location','southeast');
        xlabel('Time');
        ylabel('X');
        grid on;
     
        subplot(2,3,3);
        bar(output,'r','BarWidth',0.5);
        title(['Transitioning labelling, horizon = ' num2str(C2(i)-C1)]);
        xlabel('Trajectory');
        ylabel('Label');
        xlim([0 mr+1])
        ylim([-1.5 1.5]);
        grid on;
      
    end

    %% NN TRAINING

    net = feedforwardnet([5 5 5]);
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'radbas';
    net.layers{3}.transferFcn = 'tansig';
    net = train(net,input.',output.');

    %% ODE SOLUTION Vs NN FORECAST

    trans_NN = [];
    trans_ode = [];
    
    for j = 1:mrr
        
        x0 = 30*(rand(3,1)-0.5); % new IC (not in the training)
        [~,y] = ode45(Lorenz,t,x0); % new trajectory (based on physics)
        yx = y(:,1);
        yx1 = yx(floor((C1-W1/2)/dt):floor((C1+W1/2)/dt));
        yx2 = yx(floor((C2(i)-W2/2)/dt):floor((C2(i)+W2/2)/dt));
        
        transition = -1;
        for k = 2:length(t2) 
            if yx2(k)*yx2(k-1)<0
               transition = 1;
            end
        end
        
        trans_ode = [trans_ode; transition]; 
        input = yx1; 
        trans_NN = [trans_NN; net(input)];     
        
        for w = 1:length(trans_NN)
            if trans_NN(w) <= 0
               trans_NN(w) = -1;
            else trans_NN(w) = 1;
            end    
        end
        
        %% RESULTS PLOTTING
        
        figure(i);
        
        subplot(2,3,4);
        plot3(y(:,1),y(:,2),y(:,3)), hold on
        plot3(x0(1),x0(2),x0(3),'ro'), hold on
        title(['Lorenz 3D trajectories for testing, rho = ' num2str(r)]);
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        view(-23,18)
        grid on;
    
        subplot(2,3,5);   
        plot(t,yx,'k:',t1,yx1,t2,yx2,'b'), hold on
        title(['Projections for classification, ' num2str(mrr) ' samples' ]);
        legend({'NN','Sampling','Labelling'},'Location','southeast');
        xlabel('Time');
        ylabel('X');
        grid on; 
            
        subplot(2,3,6);
        bar(trans_ode,'r','BarWidth',0.5); hold on;
        bar(trans_NN,'b','BarWidth',0.25); hold on;
        title(['NN transitioning forecast, horizon = ' num2str(C2(i)-C1)]);
        legend({'ODE','NN'},'Location','southeast');
        xlabel('Trajectory');
        ylabel('Label)');
        xlim([0 mrr+1])
        ylim([-1.5 1.5])
        grid on;
        
    end

end
