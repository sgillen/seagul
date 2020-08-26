function [preImpactState, impactTime] = detectImpact(tout,xout)

stateDimension = length(xout(1,:));
preImpactState = zeros(stateDimension,1,'single');
impactTime = -1; % if this is unchanged implies no impact

dt = 0.01;
params;

q1 = xout(:,1); q2 = xout(:,2); q3 = xout(:,3);
xb = xout(:,4); yb = xout(:,5);
if length(tout)>1
    x0 = 0;
    y0 = 0;
    th1 = q1;
    th2 = q2 + q1;
    th3 = q3 + q2 + q1;

    x1 =  -l1*sin(th1);
    % x1cm = -p1*sin(th1);
    x2 = x1 - l2*sin(th2);
    x3 = x2 - l3*sin(th3);
    % x2cm = x1 - p2*sin(th2);
    y1 = l1*cos(th1);
    % y1cm = p1*cos(th1);
    y2 = y1 + l2*cos(th2);
    y3 = y2 + l3*cos(th3);

    % Only need to check the points after the ball has reached the maximum
    % height
    maxHeightIndex = find(yb == max(yb));
    maxHeightIndex = maxHeightIndex(1);

    impLinkSlope = (y3(maxHeightIndex:end) - y2(maxHeightIndex:end))./(x3(maxHeightIndex:end) - x2(maxHeightIndex:end));
    impLinkIntercept = y3(maxHeightIndex:end) - impLinkSlope.*x3(maxHeightIndex:end);



    t = [tout(1):dt:tout(end)];
    previousDistance = 0;
    %% Detecting impacts
    for i = 1:1:length(impLinkSlope)
        actualIndex = i + maxHeightIndex - 1;
        distanceToImpLink = abs(yb(actualIndex) - xb(actualIndex)*impLinkSlope(i) - impLinkIntercept(i))/sqrt(1 + impLinkSlope(i)^2);
        if xb(actualIndex) > x3(actualIndex) && xb(actualIndex) < x2(actualIndex)
            if (distanceToImpLink < rb)
                impactTime = interp1([previousDistance distanceToImpLink], tout(actualIndex - 1: actualIndex), rb,'PCHIP')
                t = [tout(1):dt:impactTime];
                stateDimension = length(xout(1,:));
                preImpactState = zeros(stateDimension,1);
                for j = 1:1:stateDimension
                    preImpactState(j) = interp1(tout,xout(:,j),impactTime,'PCHIP');
                end
                break;
            end
        end
        previousDistance = distanceToImpLink;
    end
end