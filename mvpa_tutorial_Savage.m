clear all; 
close all;

%% Dependencies:
% Neuroelf toolbox
% https://github.com/neuroelf/neuroelf-matlab

ione = 1;
if ione
    addpath(genpath('C:\Users\Ione Fine\Documents\code\neuroelf-matlab'))
end


paths.main = {'X:\TristramSavage\MT_xMod_2021\wjPilot_freesurf2BV\derivatives\brainvoyager\'};
paths.subject = {'sub-Pilot1'};

%% load ROI (aka .voi files)
roiFileName=['MT_L-and-R_from3mm_combo.voi'];
roi(1).name = {'MT_L_from3mm'};
roi(2).name = {'MT_R_from3mm'};

% modelName=['*2mm*24preds01.glm'];
% condFileName=['*2x2*.mat'];

modelName=['*3mm*24preds01.glm'];
condFileName=['*3x3*.mat'];




paths.roi = {'rois'}; % where are the roi files located inside the subject directory
for run = 1:length(roi); roi(run).predictors = []; end % initialize the roi struct to collate later

%% load beta weights or glm data
paths.session = fullfile(paths.main, paths.subject, {'ses-01', 'ses-02'});

%% define if using vmp or glm BOLD data
% dataformat = 'vmp';
% paths.data = fullfile('derivatives', '*3mm*24preds01.vmp'); % where the data files are located inside the subject directory



dataformat = 'glm';
paths.data = fullfile('derivatives', modelName); % where the data files are located inside the subject directory



%% setup experimental condition lists
factor(1).col = 2; factor(1).labels =  {'left', 'right'}; factor(1).chance = 1/2;
factor(2).col = NaN; factor(2).labels = {'session'}; factor(2).chance = NaN; % records session
factor(3).col = NaN; factor(3).labels = {'run'}; factor(3).chance = NaN; % records run

for f = 1:length(factor); factor(f).classlabels = [];  end % initialize the factors to collate later

%% collect all the rois
roi_xff = mvpa.load_roi(fullfile(paths.main, paths.subject,paths.roi, roiFileName)); % load the rois

if iscell(roi_xff); roi_xff=roi_xff{1}; disp(['Coerced roi_xff from cell to xff']); end


for sess = 1:length(paths.session) % for each session
    disp(sess)
    %     cond_filelist = dir(fullfile(paths.session{sess}, condFileName)); % each experimental condition file
    cond_filelist = dir(fullfile(paths.session{sess}, condFileName)); % each experimental condition file
    data_filelist = dir(fullfile(paths.session{sess}, paths.data)); % each data file
    if length(cond_filelist)~=length(data_filelist)
        error(['number of condition files ', num2str(length(condfilelist)), ...
            ' does not match the number of experimental files', num2str(length(data_filelist))]);
    end


    for run = 1:length(data_filelist) % for each vmp/glm file

        % deal with factors
        conditions = mvpa.load_exp(cond_filelist(run)); % load exp protocols

        factor = mvpa.collate_factor_labels(factor, cell2mat(conditions.mat(:, 1:5)), sess, run);        % save the class labels
        % deal with data
        data_xff = mvpa.load_data(data_filelist(run)); % load in vmp or glm
        data_roi = mvpa.subset_data_rois(data_xff, roi_xff, dataformat); % just save the data for ROIs, in a temp structure, nvoxels x n events/blocks
        roi = mvpa.collate_roi_predictors(roi, data_roi, dataformat); % now collate the roi data, over all the runs
    end
end

%% training time
model(1).desc = {'DiscrimType', 'linear', 'OptimizeHyperparameters', 'auto'};
%  Example models, increasing in power:
% 'DiscrimType: 'linear', 'quadratic', cubic
% If you want to go crazy and do SVM the terminology changes a little:
% model(2).desc = {'SVM, OptimizeHyperparameters, 'none'};

% 'OptimizeHyperparameters', 'auto' is slow, basically does something
% like PCA before classification

model(1).class_factor = 1; % which factor are you trying to classify?
<<<<<<< Updated upstream
% model(1).add_pred = {'session'};
model(1).add_pred = {'session','runs'};
=======
model(1).add_pred = {'session'};
>>>>>>> Stashed changes
% this adds additional predictors to the BOLD data. For example a non-classification factor, can also specify session and run as additional predictors
% model(1).add_pred ={}; , but you don't have to

model(1).CVstyle = {'Kfold', 10};
model(1).color = 'r'; model(1).sym = 's';
% Specifies cross validation style.
% Examples of sensible cross validation styles:
% {{'Kfold',  5}, {'Holdout', .1}, {'Leaveout', 'on'},

% if you want to Generalize over specific factors (e.g. use one
% factor to select a training set, and the other to select a test set
% the terminology is a little different:
% model(1).CVstyle= {'Generalize', 1, 'Seq', 'Onset'};
% define which factor you are using to select your train/tests sets
% using. Then specify the labels for train and test

model(1).Exclude = {}; % list of conditions to exclude, only works for non generalize right now

figure;
for r = 1:length(roi)
    for m = 1:length(model)
        predictors = mvpa.generate_predictors(model(m), factor, roi(r));

        predictors = mvpa.exclude_factors(predictors, model(m), factor);
        if strcmp(model(m).CVstyle, 'Generalize')
            [perf, Mdl, Mdl_CV] = mvpa.classify(model(m),  roi(r).predictors, ...
                factor(model(m).class_factor).classlabels, factor(model(m).CVstyle{2}).classlabels);
        else
            [perf, Mdl, Mdl_CV] = mvpa.classify(model(m),  roi(r).predictors, ...
                factor(model(m).class_factor).classlabels);
        end
        figure(10)
        h(m) = errorbar(r, perf.mean, perf.std, model(m).sym);
        set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color); hold on
    end
end
set(gca, 'XLim', [0 4])
