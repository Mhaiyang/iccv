clc;clear;

algorithms = {
%      'C2SNet';
%      'RAS';
%      'PAGRN';
%      'DGRL';
%      'R3Net';
%      'BMPM';
%      'PiCANet-R';
%      'DSS';
%      'BASNet';
%      'CPD';
%      'PAGE-Net';
%      'PoolNet';
%      'AFNet';
%      'BANet';
%      'F3Net';
%      'DCENet';
%      'GDNet';


     'MirrorNet_NAC_resnet50_bie_four_ms_poly_v12'
    };

datasets = {
              'SOD';
              'PASCAL-S';
              'DUT-OMRON';
              'ECSSD';
              'HKU-IS';
              'DUTS-TE';
              
    };

tic
for i = 1:numel(algorithms)
    alg = algorithms{i};
    fprintf('%s\n', alg);
    txt_path = ['./mat/' alg '/'];
    if ~exist(txt_path, 'dir'), mkdir(txt_path); end
    fileID = fopen([txt_path 'results.txt'],'w');
    
    for j = 1:numel(datasets)
        dataset      = datasets{j};
%         predpath     = ['/media/iccd/disk/15/' alg '/' dataset '/'];
%         predpath     = ['../results/' alg '/' dataset '/'];
        predpath     = ['../results/' alg '/120/' dataset '/'];
        maskpath     = ['/media/iccd/disk1/saliency_benchmark/' dataset '/mask/'];
%         maskpath     = ['../../data/' dataset '/mask/'];
        if ~exist(predpath, 'dir'), continue; end

        names = dir(['/media/iccd/disk1/saliency_benchmark/' dataset '/mask/*.png']);
%         names = dir(['../../data/' dataset '/mask/*.png']);
        names = {names.name}';
        wfm          = 0; mae    = 0; sm     = 0; fm     = 0; prec   = 0; rec    = 0; em     = 0;
        score1       = 0; score2 = 0; score3 = 0; score4 = 0; score5 = 0; score6 = 0; score7 = 0;

        results      = cell(numel(names), 6);
        ALLPRECISION = zeros(numel(names), 256);
        ALLRECALL    = zeros(numel(names), 256);
        file_num     = false(numel(names), 1);
        
        
    
        
        for k = 1:numel(names)
            name          = names{k,1};
            results{k, 1} = name;
            file_num(k)   = true;
            fgpath        = [predpath name];
%             fgpath        = [predpath name(1:end-4) '.jpg'];
            fg            = imread(fgpath);

            gtpath = [maskpath name];
            gt = imread(gtpath);

            if length(size(fg)) == 3, fg = fg(:,:,1); end
            if length(size(gt)) == 3, gt = gt(:,:,1); end
            fg = imresize(fg, size(gt)); 
            fg = mat2gray(fg); 
            gt = mat2gray(gt);
%             if max(fg(:)) == 0 || max(gt(:)) == 0, continue; end
            
            gt(gt>=0.5) = 1; gt(gt<0.5) = 0; gt = logical(gt);
            score1                   = MAE(fg, gt);
            [score2, score3, score4] = Fmeasure(fg, gt, size(gt)); 
            score5                   = wFmeasure(fg, gt); 
            score6                   = Smeasure(fg, gt);
            score7                   = Emeasure(fg, gt);
            mae                      = mae  + score1;
            prec                     = prec + score2;
            rec                      = rec  + score3;
            fm                       = fm   + score4;
            wfm                      = wfm  + score5;
            sm                       = sm   + score6;
            em                       = em   + score7;
            results{k, 2}            = score1; 
            results{k, 3}            = score4; 
            results{k, 4}            = score5; 
            results{k, 5}            = score6;
            results{k, 6}            = score7;

        end

        file_num = double(file_num);
        fm       = fm  / sum(file_num);
        mae      = mae / sum(file_num); 
        wfm      = wfm / sum(file_num); 
        sm       = sm  / sum(file_num); 
        em       = em  / sum(file_num);
        fprintf(fileID, '%10s (%4d images): S:%6.3f, E:%6.3f, F:%6.3f, M:%6.3f\n', dataset, sum(file_num), sm, em, wfm, mae);
        fprintf('%10s (%4d images): S:%6.3f, E:%6.3f, F:%6.3f, M:%6.3f\n', dataset, sum(file_num), sm, em, wfm, mae);
        save_path = ['./mat' filesep alg filesep dataset filesep];
        if ~exist(save_path, 'dir'), mkdir(save_path); end
        save([save_path 'results.mat'], 'results');
        save([save_path 'prec.mat'], 'prec');
        save([save_path 'rec.mat'], 'rec');
    end
end
toc
