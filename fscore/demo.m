%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------------------------------
% demo for precision recall and F-measure
% Author: Guangyu Zhong, Guangyuzhonghikari@gmail.com
%----------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear;close all;
% addpath(genpath('.'));

%% First method
input_path = '/home/iccd/data/msd9/test/image/';
truth_path = '/home/iccd/data/msd9/test/mask/';
% input_path = '/home/iccd/data/MSRA10K/DUT-OMRON/image/';
% truth_path = '/home/iccd/data/MSRA10K/DUT-OMRON/mask/';
% input_path = '/media/iccd/disk/saliency_benchmark/HKU-IS/image/';
% truth_path = '/media/iccd/disk/saliency_benchmark/HKU-IS/mask/';
% input_path = '/home/iccd/data/MSRA10K/PASCAL-S/image/';
% truth_path = '/home/iccd/data/MSRA10K/PASCAL-S/mask/';
% input_path = '/home/iccd/data/MSRA10K/HKU-IS/image/';
% truth_path = '/home/iccd/data/MSRA10K/HKU-IS/mask/';
% result_path = '/home/iccd/iccv/ckpt/TAYLOR5_MSRA_BCE/TAYLOR5_MSRA_BCE_100/E/';
result_path = '/home/iccd/iccv/msd9_results/msd9_BDRAR/';
% result_path = '/home/iccd/R3Net/ckpt/R3Net/(R3Net) HKU-IS_6000/';

[PreF,RecallF,FMeasureF] =  get_Fmeasure(input_path,result_path,truth_path);

disp(FMeasureF)

%% Second method
[Pre, Recall, fpr, AUC] = evaluate(result_path, truth_path);
FMeasure =  1.3 .* Pre .* Recall ./ (0.3 .* Pre + Recall + eps);
FScore = max(FMeasure);
sprintf('%.3f', FScore)

figure(1);set(gcf,'color','white'); xlabel('Recall'); ylabel('Precision');hold on;
grid on;axis equal;set(gca,'XTick',0:0.05:1);set(gca,'YTick',0:0.05:1.05);
plot(Recall,Pre,'r--');hold on;
