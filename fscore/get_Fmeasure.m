function [PreF,RecallF,FMeasureF] =  get_Fmeasure(imdir,outdir, gtdir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------------------------------
% Evaluate result by using F-measure
% Date: Jan 3th, 2013
% Author: Jinshan Pan, jspan@mail.dlut.edu.cn
% changed by Guangyu Zhong, Guangyuzhonghikari@gmail.com
%----------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%
imnames=dir([imdir '*' 'g']);

for ii=1:length(imnames)
    ii
    imname=[imdir imnames(ii).name];
    I=imread(imname);
    [m,n,z]=size(I);
    
    gtname=[gtdir imnames(ii).name(1:end-4) '.png'];
    gt=double(imread(gtname));
    [gm,gn]=size(gt);
    gt((gt>0))=1;

    salname=[outdir imnames(ii).name(1:end-4) '.png'];
    salImrc=double(imread(salname));
    if sum(salImrc(:))==0
        continue;
    end
    salimg=(salImrc-min(salImrc(:)))/(max(salImrc(:))-min(salImrc(:)));
   
    th1=2*mean(salimg(:));
    if th1>1
        th1=1;
    end
    bimg1=salimg;
    bimg1(bimg1<th1)=0;
    bimg1(bimg1>=th1)=1;
    bimg1=double(bimg1);
    and=sum(sum(bimg1.*gt));
    pa(ii)=0;
    ra(ii)=0;
    pa(ii)=pa(ii)+and/(sum(bimg1(:))+eps);
    ra(ii)=ra(ii)+and/sum(gt(:));
  
end
PreF=mean(pa);
RecallF=mean(ra);
FMeasureF=1.3*PreF*RecallF/(0.3*PreF+RecallF+eps);