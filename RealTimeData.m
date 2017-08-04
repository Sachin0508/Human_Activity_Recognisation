function RealTimeData
sn = load('TrainedNetwork.mat','net');


sa = load('BufferedAccelerations.mat','atx','aty','atz','t','fs','actid','actnames');


g = 9.81;
fs=12.820128;

% h = plot(sa.t',zeros(size(sa.t,2),3),'LineWidth',1.5);
% xlim([0 sa.t(end)])
% ylim([-2*g 2*g])
% xlabel('Time offset (s)')
% ylabel('Acceleration (m \cdot s^{-2})')
% legend({'a_x','a_y','a_z'})
% grid on 

 for k = 1: 100
   m= mobiledev;
    m.logging=1;
    a= accellog(m);

%     [t,~] = accellog(m);
%      plot(a,t);
    
    
%     if(~ishandle(h))
%         break
%     end
    
    % Get data - one buffer for each acceleration component
    ax = a(:,1)'; 
    ay = a(:,2)';
    az = a(:,3)';

    
    % Extract feature vector
    f = featuresFromBuffer(ax, ay, az, fs);
    
    % Classify with neural network
    scores = sn.net(f');
    % Interpret result: use index of maximum score to retrieve the name of
    % the activity
    [~, maxidx] = max(scores);
    
     Activity = sa.actnames(maxidx)

%     printf('Estimated: %s\n',Activity)

% actualActivity = sa.actnames{sa.actid(k)};
    
%     displayPrediction
   clear m;
%     clear ax;
%     clear ay;
%     clear az;
%     clear a;
% clear all;
     pause(1)
%     drawnow
%         clear m
 end
    
%     function plotBuffer
% %         plot(length(ax),ax)
%         h(1).YData = g*ax'; 
%         h(2).YData = g*ay'; 
%         h(3).YData = g*az';
%     end
% 
%     function displayPrediction
%         title(sprintf('Estimated: %s\n',estimatedActivity));
%       drawnow
%     end

end
