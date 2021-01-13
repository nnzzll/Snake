clear,clc;
I = imread('IM000001_0005.png');
I = rgb2gray(I);
I = im2double(I);
Igs = imgaussfilt(I,1);
figure(1),imshow(Igs);  
x=[];y=[];c=1;N=100; %����ȡ�����c,����N  
% ��ȡUser�ֶ�ȡ�������  
% [x,y]=getpts  
while c<N  
    [xi,yi,button]=ginput(1);  
    % ��ȡ��������  
    x=[x xi];  
    y=[y yi];  
    hold on  
    % text(xi,yi,'o','FontSize',10,'Color','red');  
    plot(xi,yi,'ro');  
    % ��Ϊ�һ�����ֹͣѭ��  
    if(button==3), break; end
    
    c=c+1;  
end

% ����һ���㸴�Ƶ�������󣬹���Snake��  
xy = [x;y];  
c=c+1;  
xy(:,c)=xy(:,1);  
% �������߲�ֵ  
t=1:c;  
ts = 1:0.1:c;  
xys = spline(t,xy,ts);  
xs = xys(1,:);  
ys = xys(2,:);  
% ������ֵЧ��  
hold on  
temp=plot(x(1),y(1),'ro',xs,ys,'b.');  
legend(temp,'ԭ��','��ֵ��');

% =========================================================================  
%                     Snakes�㷨ʵ�ֲ���  
% =========================================================================
NIter =2000; % ��������  
alpha=0.2; beta=0.2; gamma = 1; kappa = 0.1;  
wl = 0; we=0.4; wt=0;  
[row ,col] = size(Igs);  
  
% ͼ����-�ߺ���  
Eline = Igs;  
% ͼ����-�ߺ���  
[gx,gy]=gradient(Igs);  
Eedge = -1*sqrt((gx.*gx+gy.*gy));  
% ͼ����-�յ㺯��  
% �����Ϊ�����ƫ����������ɢ���ƫ����������  
m1 = [-1 1];   
m2 = [-1;1];  
m3 = [1 -2 1];   
m4 = [1;-2;1];  
m5 = [1 -1;-1 1];  
cx = conv2(Igs,m1,'same');  
cy = conv2(Igs,m2,'same');  
cxx = conv2(Igs,m3,'same');  
cyy = conv2(Igs,m4,'same');  
cxy = conv2(Igs,m5,'same');  
  
for i = 1:row  
    for j= 1:col  
        Eterm(i,j) = (cyy(i,j)*cx(i,j)*cx(i,j) -2 *cxy(i,j)*cx(i,j)*cy(i,j) + cxx(i,j)*cy(i,j)*cy(i,j))/((1+cx(i,j)*cx(i,j) + cy(i,j)*cy(i,j))^1.5);  
    end  
end  
  
%figure(3),imshow(Eterm);  
%figure(4),imshow(abs(Eedge));  
% �ⲿ�� Eext = Eimage + Econ  
Eext = wl*Eline + we*Eedge + wt*Eterm;  
% �����ݶ�  
[fx,fy]=gradient(Eext);  
  
xs=xs';  
ys=ys';  
[m, n] = size(xs);  
[mm, nn] = size(fx);  
  
% ������Խ�״����  
% ��¼: ��ʽ��14�� b(i)��ʾviϵ��(i=i-2 �� i+2)  
b(1)=beta;  
b(2)=-(alpha + 4*beta);  
b(3)=(2*alpha + 6 *beta);  
b(4)=b(2);  
b(5)=b(1);  
  
A=b(1)*circshift(eye(m),2);  
A=A+b(2)*circshift(eye(m),1);  
A=A+b(3)*circshift(eye(m),0);  
A=A+b(4)*circshift(eye(m),-1);  
A=A+b(5)*circshift(eye(m),-2);  
  
% ����������  
[L, U] = lu(A + gamma.* eye(m));  
Ainv = inv(U) * inv(L);

figure(3)
for i=1:NIter
ssx = gamma*xs - kappa*interp2(fx,xs,ys);
ssy = gamma*ys - kappa*interp2(fy,xs,ys);
xs = Ainv * ssx;
ys = Ainv*ssy;
imshow(I);
hold on;
plot([xs;xs(1)],[ys;ys(1)],'r-');
hold off;
pause(0.001)
end