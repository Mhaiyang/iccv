clc;clear;

input_path = '/media/iccd/TAYLORMEI/saliency_dataset/HKU-IS';
output_path = '/home/iccd/data/MSRA10K/HKU-IS';

file = load('/media/iccd/TAYLORMEI/saliency_dataset/HKU-IS/testImgSet.mat');
imglist = cellstr(file.testImgSet);

for j = 1 : length(imglist)
    j
    imgname = imglist{j}
    copyfile([input_path '/image/' imgname], [output_path '/image/'])
    copyfile([input_path '/mask/' imgname], [output_path '/mask/'])
end;
    
