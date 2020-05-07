%% Function to animate the walker
function animate(tout,xout)
    params;

    t = tout;
    theta = 0:0.1:2*pi;
    % Animating the motion
    for n=1:length(t)
        q1 = interp1(tout,xout(:,1),t(n),'PCHIP'); 
        q2 = interp1(tout,xout(:,2),t(n),'PCHIP'); 
        xb = interp1(tout,xout(:,3),t(n),'PCHIP');
        yb = interp1(tout,xout(:,4),t(n),'PCHIP');

        xBall = xb + rb*sin(theta);
        yBall = yb + rb*cos(theta);

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


        x=[x0, x1, x2];
        y=[y0, y1, y2];

        if n==1
            p1 = plot(x,y,'b-','LineWidth',2);
            axis image; 
            hold on
            p2 = plot(xBall,yBall,'g-','LineWidth',2);
            xp = [-1:.01:1];
            yp = zeros(length(xp));
            plot(xp,yp,'k-')
            %keyboard
    %         dx = xp;
    %         pts = round(dx*2)/2;
    %         for xtext = min(pts):.5:max(pts)
    %             text(xtext, -.2,...
    %                 num2str(xtext,'%.1f'),'Color',[0 .6 0]);
    %             plot(xtext+[0 0], [0 -.05],...
    %             'r-','Color',[0 .6 0])
    %         end
            axis([x0 + [-1 1] y0 + [-1 1]]);
    %         grid on
            axis off
        else
            set(p1,'XData',x,'YData',y);
            set(p2,'XData',xBall,'YData',yBall);
            axis([x0 + [-1 1] y0 + [-1 1]]);
    %         grid on 
    %         axis off
            M(n) = getframe;
        end
        drawnow
        pause(0.05)
    end
end
        
    


