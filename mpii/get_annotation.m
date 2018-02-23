load('annotation.mat')
%   seperate test and train
annolist_test = RELEASE.annolist(RELEASE.img_train == 0);
annolist_train = RELEASE.annolist(RELEASE.img_train == 1);

%   get valid single pose indexs
%rectidxs_test = RELEASE.single_person(RELEASE.img_train == 0);
%rectidxs_train = RELEASE.single_person(RELEASE.img_train == 1);

%   caching annotations
cache_annotations(annolist_train, rectidxs_train, 'train/');
fprintf('finished\n');