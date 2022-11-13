clear all;
close all;

%% Dependencies:
% Neuroelf toolbox
% https://github.com/neuroelf/neuroelf-matlab


% paths.main = {'C:\_Offline_Storage\deNoise\'};%local path

%paths.main = {'X:\TristramSavage\XMod\derivatives\deNoise'};%server path
ione = 1;
if ione
    addpath(genpath('C:\Users\Ione Fine\Documents\code\neuroelf-matlab'));
    paths.main = {'C:\Users\Ione Fine\Documents\code\tristram_classify\'};
end

paths.subject = {'sub-SRxGxKKx1950'};
paths.roi = {'rois'}; % where are the roi files located inside the subject directory
paths.session = fullfile(paths.main, paths.subject, {'ses-sr01', 'ses-sr02','ses-sr03','ses-sr04','ses-sr05'});

%% load ROI (aka .voi files)
roiFileName=['hMT_L+R.voi'];
roi(1).name = {'hMT_L'};
roi(2).name = {'hMT_R'};
for run = 1:length(roi); roi(run).predictors = []; end % initialize the roi struct to collate later

modelName=['*24-class.glm']; %name of glms produced
condFileName=['sub-SR-G-kk*.mat']; % condition files containing experimental info

%% load beta weights or glm data

%% define if using vmp or glm BOLD data
% dataformat = 'vmp';
% paths.data = fullfile('derivatives', '*3mm*24preds01.vmp'); % where the data files are located inside the subject directory

dataformat = 'glm';
paths.data = fullfile(modelName); % where the data files are located inside the subject directory


%% setup experimental condition lists
factor(1).col = 2; factor(1).name  = 'Direction'; factor(1).labels =  {'left', 'right'}; factor(1).chance = 1/2;
factor(2).col = 3; factor(2).name  = 'Modality'; factor(2).labels =  {'Aud', 'Vis'}; factor(2).chance = 1/2; 
factor(3).col = NaN; factor(3).name  = 'Modality + Direction'; factor(3).labels = {'Combo'}; factor(3).chance = 1/4;  % combines the other factors
factor(4).col = NaN;  factor(4).name  = 'Session'; factor(4).labels = {'Session'}; factor(4).chance = NaN;% records session
factor(5).col = NaN; factor(5).name  = 'Run'; factor(5).labels = {'Run'}; factor(5).chance = NaN;  % records run

for f = 1:length(factor); factor(f).classlabels = [];  end % initialize the factors to collate later

%% collect all the rois
roi_xff = mvpa.load_roi(fullfile(paths.main, paths.subject,paths.roi, roiFileName)); % load the rois
if iscell(roi_xff); roi_xff=roi_xff{1}; disp(['Coerced roi_xff from cell to xff']); end

%% add the model
model(1).desc = {'DiscrimType', 'linear',  'OptimizeHyperparameters', 'auto'};
model(1).class_factor = 1; % which factor are you trying to classify?
model(1).add_pred ={'Session'};% {'Session', 'Run'};  % see if you want to include session or run as predictors, but you don't have to
model(1).CVstyle = {'Kfold', 5}; % cross validation method
model(1).color = 'r'; model(1).sym = 's';

model(2)  = model(1); 
model(2).color = 'b'; model(1).sym = 'o';
m=1;
for sess = 1:length(paths.session)
    disp(['Session = ', num2str(sess)]) % for each session
    cond_filelist = dir(fullfile(paths.session{sess}, condFileName)); % each experimental condition file
    data_filelist = dir(fullfile(paths.session{sess}, paths.data)); % each data file

    if length(cond_filelist)~=length(data_filelist)
        error(['number of condition files ', num2str(length(condfilelist)), ...
            ' does not match the number of experimental files', num2str(length(data_filelist))]);
    end

    for run = 1:length(data_filelist) % for each vmp/glm file
        % deal with factors
        conditions = mvpa.load_exp(cond_filelist(run));                                            % load exp protocols
        % loads a mat file, conditions.mat is nevents x nconditions
        factor = mvpa.collate_factor_labels(factor, cell2mat(conditions.mat(:, 1:3)), sess, run);  % save the class labels#
        % creates a structure factor, containing class labels collated
        % across runs/sessions

        % deal with data
        data_xff = mvpa.load_data(data_filelist(run));                                             % load in vmp or glm
        data_roi = mvpa.subset_data_rois(data_xff, roi_xff, dataformat);              % just save the data for ROIs, in a temp structure, nvoxels x n events/blocks
        roi = mvpa.collate_roi_predictors(roi, data_roi, dataformat);                      % now collate the roi data, over all the runs     
    end
end

roi = mvpa.add_predictors(model(m), factor, roi);         % option to add session and run as predictors

figure(10)
for r = 1:2
subplot(2,2,r)
 imagesc(roi(r).predictors)
 roi(r).whitened = mvpa.whiten(roi(r).predictors, .0001);
 subplot(2,2,r+2)
 imagesc(roi(r).whitened)
colormap(gray); xlabel('voxels'); ylabel('events')
roi(r).predictors = roi(r).whitened;
end


%% look at classification across all sessions
for m = 1:2
    if m ==1
        model(m).Exclude = {2, 'Vis' }; % list of conditions to exclude##
    else
        model(m).Exclude = {2, 'Aud'}; % list of conditions to exclude##
    end
    [ tmp(m).roi, tmp(m).factor] = mvpa.exclude_predictors(roi, model(m), factor);
end
for m = 1:2
    for r = 1:length(roi)
        % within modality classification
        predictors = tmp(m).roi.predictors;
        classlabels  =  tmp(m).factor(model(m).class_factor).classlabels;
        [output(m,r).perf, output(m,r).Mdl, output(m,r).Mdl_CV] = mvpa.classify(model(m),double(predictors), classlabels);
        figure(m)
        e(m) =errorbar(r+.25, output(m,r).perf.mean,output(m,r).perf.std); hold on
        set(e(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color);
        h(m) = plot(r+.25, output(m,r).perf.mean,model(m).sym); hold on
        set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color, 'MarkerSize', 15); hold on

        % across modality classification
        mg = mod(m, 2)+1;
        predictorsg = tmp(mg).roi.predictors; % predictors for the other modality
        classlabelsg  =  tmp(mg).factor(model(mg).class_factor).classlabels; % class labels for the other modality
        [output(m, r).label,score,cost] = predict(output(m,r).Mdl,predictorsg);
        output(m, r).perfg = sum(strcmp(output(m, r).label, classlabelsg))/length(classlabelsg);
        h(m) = plot(2.25+r, output(m,r).perfg,model(m).sym); hold on
        set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color, 'MarkerSize', 15); hold on
    end
end

for m = 1:2
    figure(m)
    plot([0 4], [.5 .5], 'k--')
    set(gca, 'XLim', [0 6])
    set(gca, 'YLim', [ .4 .7])
    set(gca, 'XTick', 1.2:4.2)
    set(gca, 'XTickLabel', {'within L', 'within R', 'crossModal L', 'crossmodal R'})
    if m==1
        title('Auditory Training')
    else
        title('Visual Training')
    end
end


%% look at classification within individual sessions
for m = 1:2
    model(1).CVstyle = {'Kfold', 3}; % cross validation method
    model(1).add_pred ={};
    for sess = 1:4
        ind = setdiff(1:4, sess);
        if m ==1
            model(m).Exclude = {2, 'Vis'; 5, ind(1); 5, ind(2);  5, ind(3)}; % list of conditions to exclude
        else
            model(m).Exclude = {2, 'Aud'; 5,ind(1); 5, ind(2);  5,ind(3)}; % list of conditions to exclude##
        end
        [ tmp(m, sess).roi, tmp(m, sess).factor] = mvpa.exclude_predictors(roi, model(m), factor);
    end
end
for m = 1:2
    for sess = 1:4
        for r = 1:length(roi)
            % within modality classification
            predictors = tmp(m, sess).roi.predictors;
            classlabels  =  tmp(m,sess).factor(model(m).class_factor).classlabels;
            [output(m,r, sess).perf, output(m,r, sess).Mdl, output(m,r, sess).Mdl_CV] = mvpa.classify(model(m),double(predictors), classlabels);
            figure(m)
            h(m) = text(r+(sess*.1), output(m,r,sess).perf.mean,num2str(sess)); hold on
        %    set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color); hold on

            % across modality classification
            g = mod(m, 2)+1; 
            predictorsg = tmp(g, sess).roi.predictors; % predictors for the other modality
            classlabelsg  =  tmp(g, sess).factor(model(g).class_factor).classlabels; % class labels for the other modality
            [output(m, r, sess).label,score,cost] = predict(output(m,r, sess).Mdl,predictorsg);
            output(m, r, sess).perfg = sum(strcmp(output(m, r, sess).label, classlabelsg))/length(classlabelsg);
            h(m) = text(2+r+(sess*.1), output(m,r,sess).perfg, num2str(sess)); hold on
   %         set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color); hold on
        end
    end
end


