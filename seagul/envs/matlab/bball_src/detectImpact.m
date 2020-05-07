function [preImpactState, impactTime] = detectImpact(tout,xout)

stateDimension = length(xout(1,:));
preImpactState = zeros(stateDimension,1,'single');
impactTime = -1; % if this is unchanged implies no impact

params;

q1 = xout(:,1); q2 = xout(:,2);
xb = xout(:,3); yb = xout(:,4);
if length(tout)>1
    x0 = 0;
    y0 = 0;
    th1 = q1;
    th2 = q2 + q1;


    x1 =  -l1*sin(th1);
    % x1cm = -p1*sin(th1);
    x2 = x1 - l2*sin(th2);
    % x2cm = x1 - p2*sin(th2);
    y1 = l1*cos(th1);
    % y1cm = p1*cos(th1);
    y2 = y1 + l2*cos(th2);

    % Only need to check the points after the ball has reached the maximum
    % height
    maxHeightIndex = find(yb == max(yb));
    maxHeightIndex = maxHeightIndex(1);
    
    if maxHeightIndex == 1
        maxHeightIndex = 2;
    end

    link2Slope = (y2(maxHeightIndex:end) - y1(maxHeightIndex:end))./(x2(maxHeightIndex:end) - x1(maxHeightIndex:end));
    link2Intercept = y2(maxHeightIndex:end) - link2Slope.*x2(maxHeightIndex:end);

    t = tout;
    previousDistance = 0;
    %% Detecting impacts
    for i = 1:1:length(link2Slope)
        actualIndex = i + maxHeightIndex - 1;
        distanceToLink2 = abs(yb(actualIndex) - xb(actualIndex)*link2Slope(i) - link2Intercept(i))/sqrt(1 + link2Slope(i)^2);
        if xb(actualIndex) > x2(actualIndex) && xb(actualIndex) < x1(actualIndex)
            if (distanceToLink2 < rb)
                impactTime = interp1([previousDistance distanceToLink2], tout(actualIndex-1: actualIndex), rb,'PCHIP'); 
                t = tout;
                preImpactState = zeros(stateDimension,1);
                for i = 1:1:stateDimension
                    preImpactState(i) = interp1(tout,xout(:,i),impactTime,'PCHIP');
                end
                break;
            end
        end
        previousDistance = distanceToLink2;
    end

end

