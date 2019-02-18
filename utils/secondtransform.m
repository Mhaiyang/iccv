clc;clear;
path = '/home/iccd/data/2019beforetrue/ylt/';

list = dir(fullfile(path, '*_json'));

for i = 1 : length(list)
    fprintf(list(i).name);
    mask = imread(fullfile(path, list(i).name, 'label.png'));
    imwrite(mask, fullfile(path, list(i).name, 'label8.png'));

end
