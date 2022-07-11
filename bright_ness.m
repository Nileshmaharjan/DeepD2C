clc;
close all;
clear all;


x = [20,40,60,80,100];
y1 = [0.005,0.0008,0,0,0];
y2 = [0.01,0.005,0.002,0,0];
y3 = [0.015,0.007,0.0025,0,0];
y4 = [0.02,0.012,0.0065,0.003,0.0005];
%names = {'HiDDeN'; 'S.GAN(Residual)'; 'S.GAN(Dense)'; 'StegaStamp'; 'Ours'};
plot(x, y1, '-v');
hold on
plot(x, y2, '-o');
hold on
plot(x, y3, '-+');
hold on
plot(x, y4, '-*');


set(gca,'XLim',[20 100]);
set(gca,'YLim',[0 0.02]);
%set(gca,'XTick',[30 90 120])
set(gca,'YTick',[0 0.005 0.01 0.015 0.02])

ylabel('BER');
xlabel('Screen brightness (%)');
%ylabel('Achievable Throughput (bps)');
grid on
grid minor
legend('Ours (Normal light)', 'StegaStamp (Normal light)', 'Ours (Ambient light)', 'StegaStamp (Ambient light)');



