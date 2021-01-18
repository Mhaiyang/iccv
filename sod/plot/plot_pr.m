clear;

% 15*2
colors = {
    [0.0000, 0.0000, 1.0000]
    [0.7098, 0.2000, 0.3608]
    [0.4902, 0.0706, 0.6863]
    [0.7059, 0.5333, 0.8824]
    [0.8000, 0.8000, 0.1000]
    [0.0588, 0.6471, 0.6471]
    [0.0392, 0.4275, 0.2667]
    [0.4157, 0.5373, 0.0824]
    [1.0000, 0.0000, 1.0000]
    [0.5490, 0.5490, 0.4549]
    [0.9373, 0.6863, 0.1255]
    [0.4471, 0.3333, 0.1725]
    [0.0000, 1.0000, 1.0000]
    [0.7176, 0.5137, 0.4392]
    [0.4685, 0.3421, 0.7256]
    [0.1767, 0.3751, 0.9243]
    [0.3345, 0.6834, 0.2567]
    [1.0000, 0.0000, 0.0000]
};

lines = {'-','-','-','-','-','-','-','-','--','--','--','--','--','--','--','--','--','-'};

names = {
    'C2SNet'
    'RAS'
    'PAGRN'
    'DGRL'
    'R3Net'
    'BMPM'
    'PiCANet-R'
    'DSS'
    'BASNet'
    'CPD'
    'PAGE-Net'
    'AFNet'
    'BANet'
    'GCPANet'
    'F3Net'
    'MINet-R'
    'ITSD'
    'GDNet-B-S'
};

years = {
    ' (ECCV''18)'
    ' (ECCV''18)'
    ' (CVPR''18)'
    ' (CVPR''18)'
    ' (IJCAI''18)'
    ' (CVPR''18)'
    ' (CVPR''18)'
    ' (TPAMI''19)'
    ' (CVPR''19)'
    ' (CVPR''19)'
    ' (CVPR''19)'
    ' (CVPR''19)'
    ' (ICCV''19)'
    ' (AAAI''20)'
    ' (AAAI''20)'
    ' (CVPR''20)'
    ' (CVPR''20)'
    ' (Ours)'
};

% dataset = 'SOD';
% dataset = 'PASCAL-S';
% dataset = 'DUT-OMRON';
% dataset = 'ECSSD';
% dataset = 'HKU-IS';
dataset = 'DUTS-TE';

PR('data_18', names, colors, lines, years, false, dataset);

title(sprintf('%s', dataset));
