clear all;
close all;

%% Dependencies:
% Neuroelf toolbox
% https://github.com/neuroelf/neuroelf-matlab
% paths.main = {'C:\_Offline_Storage\deNoise\'};%local path

%paths.main = {'X:\TristramSavage\XMod\derivatives\deNoise'};%server path

addpath(genpath('C:\Users\Ione Fine\Documents\code\libsvm'));
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
model(1).desc = {'DiscrimType', 'linear'};
model(1).class_factor = 1; % which factor are you trying to classify?
model(1).add_pred ={ 'Run'};% {'Session', 'Run'};  % see if you want to include session or run as predictors, but you don't have to
model(1).CVstyle = {'Kfold', 5}; % cross validation method
model(1).color = 'r'; model(1).sym = 'o';

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
%% create separate Aud train and Vis train datasets
ct = 1;
for m = 1:2
    if m ==1
        model(m).Exclude = {2, 'Vis' }; % list of conditions to exclude
        model(m).name = 'Aud Train';
    else
        model(m).Exclude = {2, 'Aud'}; % list of conditions to exclude##
        model(m).name = 'Vis Train';
    end
    [ tmp(m).roi, tmp(m).factor] = mvpa.exclude_predictors(roi, model(m), factor);
end
%% look at variance in voxels
ct = 1;
for m = 1:2
    for r = 1:2
        figure(10);  subplot(2,2,ct);
        image(double(tmp(m).roi(r).predictors+700)/4); xlabel('voxels'); ylabel('events'); title([model(m).name, '/', roi(r).name ] );        colormap(gray);
        roi(r).var(m,:) = var(tmp(m).roi(r).predictors,0, 1);

        figure(11);  subplot(2,2,ct);
        roi(r).var(m,:) = var(tmp(m).roi(r).predictors,0, 1);
        histogram( roi(r).var(m,:), 0:250:8000); xlabel('Voxel var over events'); ylabel('Num'); title([model(m).name, '/', roi(r).name ]);

        figure(12);  subplot(2,2,ct);
        roi(r).var_events(m,:) = var(tmp(m).roi(r).predictors,0, 2);
        hist(roi(r).var_events(m,:), 0:250:6000); xlabel('Event var over voxels'); ylabel('Num'); title([model(m).name, '/', roi(r).name ]);
        ct = ct+1;
    end
end

%% subset ROI based on  variance over events
thr = [25 7000];
%thr = [0 inf];
for r = 1:2
    roi(r).idx = find(roi(r).var(1,:)>thr(1) & roi(r).var(2,:)>thr(1) & roi(r).var(1,:)<thr(2) & roi(r).var(2,:)<thr(2));
    disp(['original roi size = ', num2str(length(roi(r).var))]);
    disp(['num vox saved = ', num2str(length(roi(r).idx))]);
end



%% do the training and testing

nreps = 100;
nFolds = model(1).CVstyle{ 2} ;
for m = 1:2
    m_a = mod(m, 2)+1;
    for r = 1:length(roi)
        predictors_w = tmp(m).roi(r).predictors(:, roi(r).idx);
        classlabels_w  =  tmp(m).factor(model(m).class_factor).classlabels;
        predictors_a= tmp(m_a).roi(r).predictors(:, roi(r).idx); % predictors for the other modality
        classlabels_a = tmp(m_a).factor(model(m_a).class_factor).classlabels; % class labels for the other modality
        % cosmo
        dsw.samples =double(predictors_w);
        nvoxels = size(dsw.samples, 2);nsamples=size(dsw.samples,1); % should be half of 576 for this dataset
        dsw.sa.labels = classlabels_w;
        dsw.sa.targets  = zeros(length(classlabels_w));
        dsw.sa.targets =double(strcmp(classlabels_w, 'left'));
        dsa.samples =double(predictors_a);
        dsa.sa.labels =  classlabels_a;
        dsa.sa.targets  = zeros(length(classlabels_a));
        dsa.sa.targets =double(strcmp(classlabels_a, 'left'));
        opt = [];
    %    opt.normalization = 'zscore';
        chunks = ceil(linspace(.01, nFolds, nsamples));
        for rep = 1:nreps
            clear output_w output_cw output_a output_ca
            disp ([ 'rep = ', num2str(rep)])
            % within modality classification
          clear Mdl
          [output_w(m,r).perf, Mdl, Mdl_CV] = mvpa.classify(model(m),double(predictors_w), classlabels_w);
            sim.w_mean(m, r, rep) = output_w(m,r).perf.mean;
            sim.w_std(m, r, rep)=output_w(m,r).perf.std;

            % across modality classification
            [output_a(m, r).label,score,cost] = predict(Mdl,predictors_a);
            output_a(m, r).mean = sum(strcmp(output_a(m, r).label, classlabels_a))/length(classlabels_a);
            sim.a_mean(m, r, rep) =output_a(m,r).mean;

            %  cosmo
            dsw.sa.chunks = chunks(randperm(nsamples))';
            p=cosmo_nfold_partitioner(dsw);
            q=cosmo_balance_partitions(p,dsw);
            for fold  = 1:nFolds
                dsw_test=cosmo_slice(dsw,q.test_indices{fold});
                dsw_train=cosmo_slice(dsw,q.train_indices{fold});
                within_pred=cosmo_classify_lda(dsw_train.samples,dsw_train.sa.targets,...
                    dsw_test.samples); % also cosmo_classify_svm
                accuracy(fold)=mean(within_pred==dsw.sa.targets(q.test_indices{fold}));
                output_cw(m,r).perf.mean = mean(accuracy);
                output_cw(m,r).perf.std = std(accuracy)/sqrt(length(accuracy));
            end
            sim.cw_mean(m, r, rep) =output_cw(m,r).perf.mean;
            sim.cw_std(m, r, rep)=output_cw(m,r).perf.std;

            % cosmo
           dsa.sa.chunks = chunks(randperm(nsamples))';
            p=cosmo_nfold_partitioner(dsa);
            q=cosmo_balance_partitions(p,dsa);
            for fold  = 1:nFolds
                dsa_test=cosmo_slice(dsa,q.test_indices{fold});
                dsa_train=cosmo_slice(dsa,q.train_indices{fold});
                across_pred=cosmo_classify_lda(dsw_train.samples,dsw_train.sa.targets,...
                    dsa_test.samples);
                accuracy(fold)=mean(across_pred==dsa.sa.targets(q.test_indices{fold}));
                output_ca(m,r).perf.mean = mean(accuracy);
                output_ca(m,r).perf.std = std(accuracy)/sqrt(length(accuracy));
            end
            sim.ca_mean(m, r, rep) =output_ca(m,r).perf.mean;
            sim.ca_std(m, r, rep)=output_ca(m,r).perf.std;
        end
    end
end

%% plot the lot
for m = 1:2
    figure(m);
    if m==1;    set(gcf, 'Name', 'Auditory Training')
    else   set(gcf, 'Name', 'Visual Training');    end
    for  r = 1:length(roi)
        subplot(1,2,1)
        e(m) =errorbar(r, mean(sim.w_mean(m, r, :)),std(sim.w_mean(m, r, :))); hold on
        set(e(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color);
        h(m) = plot(r, mean(sim.w_mean(m, r, :)), model(m).sym); hold on
        set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color, 'MarkerSize', 15); hold on
        title('in house');

        subplot(1,2,2)
        e(m) =errorbar(r, mean(sim.cw_mean(m, r, :)),std(sim.cw_mean(m, r, :))); hold on
        set(e(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color);
        h(m) = plot(r, mean(sim.cw_mean(m, r, :)),'s'); hold on
        set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color, 'MarkerSize', 15); hold on
        title('cosmo')

        subplot(1,2,1);
        h(m) = plot(2+r, mean(sim.a_mean(m, r, :)), model(m).sym); hold on
        e(m) =errorbar(2+ r, mean(sim.a_mean(m, r, :)),std(sim.a_mean(m, r, :))); hold on
        set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color, 'MarkerSize', 15); hold on
        title('in house')

        subplot(1,2,2)
        e(m) =errorbar(r+2, mean(sim.ca_mean(m, r, :)),std(sim.ca_mean(m, r, :))); hold on
        set(e(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color);
        h(m) = plot(r+2, mean(sim.ca_mean(m, r, :)),'s'); hold on
        set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color, 'MarkerSize', 15); hold on
        title('cosmo')
    end
end
for m = 1:2
    figure(m)
    for sp = 1:2
        subplot(1,2, sp)
        plot([0 4], [.5 .5], 'k--')
        set(gca, 'XLim', [0 6]);  set(gca, 'YLim', [ .4 .7]); set(gca, 'XTick', 1.2:4.2)
        set(gca, 'XTIck', [1:4])
        set(gca, 'XTickLabel', {'within L', 'within R', 'crossModal L', 'crossmodal R'})
    end
end

return

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
            predictors_w = tmp(m, sess).roi.predictors;
            predictors_w = whiten(predictors_w,0.0001)
            classlabels  =  tmp(m,sess).factor(model(m).class_factor).classlabels;
            [output_w(m,r, sess).perf, output_w(m,r, sess).Mdl, output_w(m,r, sess).Mdl_CV] = mvpa.classify(model(m),double(predictors_w), classlabels);
            figure(m)
            h(m) = text(r+(sess*.1), output_w(m,r,sess).perf.mean,num2str(sess)); hold on
            %    set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color); hold on

            % across modality classification
            g = mod(m, 2)+1;
            predictorsg = tmp(g, sess).roi.predictors; % predictors for the other modality
            classlabelsg  =  tmp(g, sess).factor(model(g).class_factor).classlabels; % class labels for the other modality
            [output_w(m, r, sess).label,score,cost] = predict(output_w(m,r, sess).Mdl,predictorsg);
            output_w(m, r, sess).perfg = sum(strcmp(output_w(m, r, sess).label, classlabelsg))/length(classlabelsg);
            h(m) = text(2+r+(sess*.1), output_w(m,r,sess).perfg, num2str(sess)); hold on
            %         set(h(m), 'MarkerEdgeColor', model(m).color,'MarkerFaceColor', model(m).color, 'Color', model(m).color); hold on
        end
    end
end


