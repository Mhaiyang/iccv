function [precision, recall, fpr, AUC] = evaluate(result_path, truth_path)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------------------------------
% Evaluate result by using Pre-Recall
% Date: Jan 3th, 2013
% Author: Jinshan Pan, jspan@mail.dlut.edu.cn
% changed by Guangyu Zhong, Guangyuzhonghikari@gmail.com
%----------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_result = dir([result_path '*.png']);
dir_truth = dir([truth_path '*.png']);

COLOR_NUM = 256;
precision = zeros(COLOR_NUM,1);
recall = zeros(COLOR_NUM,1);
fpr = zeros(COLOR_NUM,1);
for i =1:length(dir_result)
    imName = dir_truth(i).name;
    truM = imread([truth_path imName]); 
    resS=imread([result_path imName]);

    truM = im2double(truM);
    tmp = 0.5*max(truM(:));
    truM(truM>=tmp) = 255;
    truM(truM<tmp) = 0;
    
    truM = double(truM);
    resS = double(resS);
    resS = (resS-min(resS(:)))/(max(resS(:))-min(resS(:)));
    resS = resS * 255;
    groundTruth = sum(truM(:));
   
    totalPixel = size(truM,1)*size(truM,2)*255;
    for thr = 1 : COLOR_NUM  
        resM = zeros(size(resS));
        resM(resS>=(thr-1)) = 255;
        %??
        %double res = sum(resM).val[0];
        res = sum(resM(:));
        %bitwise_and(resM, truM, resM)
        resM = reshape(uint8(resM),1,numel(resM));
        truM = reshape(uint8(truM),1,numel(truM));
        resM =  bitand(resM,truM);
        common = sum(resM(:));
        precision(thr) = precision(thr) + common/(res + 1e-8);
        recall(thr) = recall(thr) + common/(groundTruth + 1e-8);
        
        fpr(thr) = fpr(thr)+ (res-common)/(totalPixel-groundTruth + 1e-8);
    end
    i
    
end
precision = precision./length(dir_result);
recall = recall./length(dir_result);
fpr = fpr./length(dir_result);

AUC = 0;
[sort_fpr recall_index] = sort(fpr);
for i = 1:length(fpr)-1
    AUC = AUC+(sort_fpr(i+1)-sort_fpr(i))*(recall(recall_index(i+1))+recall(recall_index(i)))*0.5;
end