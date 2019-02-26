clc;clear;
path = '/home/iccd/data/2019/ylt_add_mask/';

list = dir(fullfile(path, '*_json'));

for i = 1 : length(list)
    fprintf(list(i).name);
    mask = imread(fullfile(path, list(i).name, 'label.png'));
    imwrite(mask, fullfile(path, list(i).name, 'label8.png'));

end
