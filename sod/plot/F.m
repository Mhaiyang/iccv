function F(path, names, colors, lines, years, human, dataset)
% Plot saliency precision/recall or f-measure results.
%
% USAGE
%  Plot( path, names, [colors], [lines], [years], [false])
%
% INPUTS
%  path        - algorithm result directory
%  names       - {nx1} algorithm names (for legend)
%  colors      - [{nx1}] algorithm colors
%  lines       - [{nx1}] line styles
%  years       - [{nx1}] the years when the algorithms are proposed
%  human       - [bool] if plot a fixed point
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(0,'defaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize',16);
set(0,'defaultTextFontName', 'Times New Roman')
set(0,'DefaultTextFontSize',16);

% parse inputs
if(~iscell(names)), names={names}; end
if(nargin<3||isempty(colors)), colors=repmat({'r','g','b','k','m','c','y'},1,100); end
if(nargin<4||isempty(lines)), lines=repmat({'-'},1,100); end
if(nargin<5||isempty(years)), years=repmat({''},1,100); end
if(~iscell(colors)), colors={colors}; end
if(~iscell(lines)), lines={lines}; end
if(~iscell(years)), years={years}; end

% setup basic plot (isometric contour lines and human performance)
clf; box on; grid on; hold on;
% line([0 1],[.5 .5],'LineWidth',2,'Color',.7*[1 1 1]);
% plot equal high line
% for f=0.1:0.1:0.9
%     r=f:0.01:1; 
%     p=f.*r./(2.*r-f); %f=2./(1./p+1./r)
%     plot(r,p,'Color',[0 1 0]); plot(p,r,'Color',[0 1 0]); 
% end
% plot a fixed point
if(human)
    h=plot(0.7235,0.9014,'o','MarkerSize',8,'Color',[0 .5 0],...
    'MarkerFaceColor',[0 .5 0],'MarkerEdgeColor',[0 .5 0]); 
end
% set axis
set(gca,'XTick',0:0.1:1,'YTick',0:0.1:1);
grid on; xlabel('Thresholds'); ylabel('F-measure');
axis equal; axis([0 1 0 1]);

% load results for every algorithm (pr=[T,R,P,F])
n=length(names); hs=zeros(1,n); fs=cell(1,n);
for i=1:n
    txt_path = fullfile(path, dataset, [names{i} '_trpf.txt']);
    f=dlmread(txt_path); 
    f=f(f(:,2)>=1e-3,:);
    fs{i}=f;
end

% plot results for every algorithm (plot best last)
for i=1:1:n
  hs(i)=plot(fs{i}(:,1),fs{i}(:,4),'-','LineWidth',2,'Color',colors{i},'LineStyle',lines{i});
end

% show legend if names provided (report best first)
hold off; if(isempty(names)), return; end
for i=1:n
    names{i}=[names{i} years{i}]; 
end
if(human)
    hs=[h hs]; 
    names=['Human'; names(:)]; 
end
legend(hs,names,'Location','sw');

set(gcf, 'PaperPosition', [-0.1 -0.25 8 8]);
set(gcf, 'PaperSize', [7.5 7.3]);
saveas(gcf, sprintf('curve_18/%s_F.pdf', dataset));

end
