%�������������ٽ�����㷨����calcPlotErr_TPR
%����ǵõ�distance ���غϱȣ�������../ERRresults��
clear all;
close all;
clc;
basePath='/data_B/RGBT234/RGB-T234/';

sequences=dir(basePath);
sequences={sequences.name};
sequences=sequences(3:end);

trackers ={'SGT','DSST','SOWP','CSR','L1-PF','JSR','MEEM+RGBT','KCF+RGBT','CSR-DCF+RGBT','CFnet','CFnet+RGBT','SOWP+RGBT','ECO','C-COT','SRDCF','SAMF','CSR-DCF'};


bPlot=1;

calcPlotErr_TPR(sequences, trackers,  bPlot);
