function RecordedData

sn = load('TrainedNetwork.mat','net');
sa = load('BufferedAccelerations.mat',...
    'atx','aty','atz','t','fs','actid','actnames');
% Gravitational acceleration (m/s^2)
g = 9.81;

% Plot initialization
h = plot(sa.t',zeros(size(sa.t,2),3),'LineWidth',1.5);
xlim([0 sa.t(end)])
ylim([-2*g 2*g])
xlabel('Time offset (s)')
ylabel('Acceleration (m \cdot s^{-2})')
legend({'a_x','a_y','a_z'})
grid on

for k = 1:size(sa.atx,1)
    
    if(~ishandle(h))
        break
    end
    
    % Get data - one buffer for each acceleration component
    ax = sa.atx(k,:); %#ok<*NODEF>
    ay = sa.aty(k,:);
    az = sa.atz(k,:);
    
    % Plot three signals
    plotBuffer
    
    % Extract feature vector
    f = featuresFromBuffer(ax, ay, az, sa.fs);
    
    % Classify with neural network
    scores = sn.net(f');
    % Interpret result: use index of maximum score to retrieve the name of
    % the activity
    [~, maxidx] = max(scores);
    estimatedActivity = sa.actnames{maxidx}; %#ok<*USENS>
    
    % Display result as title in current plot along with ground truth
    actualActivity = sa.actnames{sa.actid(k)};
    displayPrediction
    pause(0.02)
    drawnow
   
end

    
    function plotBuffer
        h(1).YData = g*ax'; 
        h(2).YData = g*ay'; 
        h(3).YData = g*az';
    end

    function displayPrediction
        title(sprintf('Estimated: %s\nActually: %s\n', ...
            estimatedActivity,actualActivity))
        drawnow
    end

end


