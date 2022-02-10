[audio, fs] = audioread('C:\Users\Jake\Documents\CS DegreeWork\Spring 2020\Senior Design\IRMAS-TrainingData\IRMAS-TrainingData\cla\[cla][cla]0152__2.wav');
ads = audioDatastore('C:\Users\Jake\Documents\CS DegreeWork\Spring 2020\Senior Design\IRMAS-TrainingData\IRMAS-TrainingData', 'LabelSource', 'foldernames', 'IncludeSubfolders', true);
sf = waveletScattering('SignalLength',length(audio(:,1)),'SamplingFrequency',fs,...
    'InvarianceScale',0.5);

 [fb,f,filterparams] = filterbank(sf);
 phi = ifftshift(ifft(fb{1}.phift));
 psiL1 = repelem(ifftshift(ifft(fb{2}.psift(:,end))),2);
 dt = 1/fs;
 time = -2^18*dt:dt:2^18*dt-dt;
 scalplt = plot(time,repelem(phi,2),'linewidth',1.5);
 hold on
 grid on
 ylimits = [-3e-4 3e-4];
 ylim(ylimits);
 plot([-0.25 -0.25],ylimits,'k--');
 plot([0.25 0.25],ylimits,'k--');
 xlim([-0.6 0.6]);
 xlabel('Seconds'); ylabel('Amplitude');
 wavplt = plot(time,[real(psiL1) imag(psiL1)]);
 legend([scalplt wavplt(1) wavplt(2)],{'Scaling Function','Wavelet-Real Part','Wavelet-Imaginary Part'});
 title({'Scaling Function';'Coarsest-Scale Wavelet First Filter Bank'})
 hold off

rng(100);
ads = shuffle(ads);
[adsTrain,adsTest] = splitEachLabel(ads,0.8);
countEachLabel(adsTrain)
countEachLabel(adsTest)
% 
Ttrain = tall(adsTrain);
Ttest = tall(adsTest);
% 
 scatteringTrain = cellfun(@(x)helperscatfeatures(x,sf,length(audio(:,1))),Ttrain,'UniformOutput',false);
 scatteringTest = cellfun(@(x)helperscatfeatures(x,sf,length(audio(:,1))),Ttest,'UniformOutput',false);
% 
 TrainFeatures = gather(scatteringTrain);
 TrainFeatures = cell2mat(TrainFeatures);
% 
 TestFeatures = gather(scatteringTest);
 TestFeatures = cell2mat(TestFeatures);

%  numTimeWindows = 32;
% trainLabels = adsTrain.Labels;
% numTrainSignals = numel(trainLabels);
% trainLabels = repmat(trainLabels,1,numTimeWindows);
% trainLabels = reshape(trainLabels',numTrainSignals*numTimeWindows,1);
% % 
% testLabels = adsTest.Labels;
% numTestSignals = numel(testLabels);
% testLabels = repmat(testLabels,1,numTimeWindows);
% testLabels = reshape(testLabels',numTestSignals*numTimeWindows,1);

trainLabels = adsTrain.Labels;
testLabels = adsTest.Labels;

trainLabels = repelem(trainLabels, 5);
testLabels = repelem(testLabels, 5);

 template = templateSVM(...
     'KernelFunction', 'polynomial', ...
     'PolynomialOrder', 3, ...
     'KernelScale', 'auto', ...
     'BoxConstraint', 1, ...
     'Standardize', true);
 Classes = {'cel','cla','flu','gac','gel','org',...
     'pia','sax','tru','vio','voi'};
classificationSVM = fitcecoc(...
    TrainFeatures, ...
    trainLabels, ...
    'Learners', template, ...
    'Coding', 'onevsone','ClassNames',categorical(Classes));




function [ClassVotes,ClassCounts] = helperMajorityVote(predLabels,origLabels,classes)
% This function is in support of wavelet scattering examples only. It may
% change or be removed in a future release.

% Make categorical arrays if the labels are not already categorical
predLabels = categorical(predLabels);
origLabels = categorical(origLabels);
% Expects both predLabels and origLabels to be categorical vectors
Npred = numel(predLabels);
Norig = numel(origLabels);
Nwin = Npred/Norig;
predLabels = reshape(predLabels,Nwin,Norig);
ClassCounts = countcats(predLabels);
[mxcount,idx] = max(ClassCounts);
ClassVotes = classes(idx);
% Check for any ties in the maximum values and ensure they are marked as
% error if the mode occurs more than once
modecnt = modecount(ClassCounts,mxcount);
ClassVotes(modecnt>1) = categorical({'NoUniqueMode'});
ClassVotes = ClassVotes(:);

%-------------------------------------------------------------------------
    function modecnt = modecount(ClassCounts,mxcount)
        modecnt = Inf(size(ClassCounts,2),1);
        for nc = 1:size(ClassCounts,2)
            modecnt(nc) = histc(ClassCounts(:,nc),mxcount(nc));
        end
    end
end

function features = helperscatfeatures(x,sf, len)
% This function is in support of wavelet scattering examples only. It may
% change or be removed in a future release.

features = featureMatrix(sf,x(1:len),'Transform','log');
features = features(:,1:8:end)';
end