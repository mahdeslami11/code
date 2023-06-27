clear all;
addpath('..');

mnistfiles = 0;
if( exist('t10k-images-idx3-ubyte', 'file') == 2 )
    mnistfiles = mnistfiles + 1;
end
if( exist('t10k-labels-idx1-ubyte', 'file') == 2 )
    mnistfiles = mnistfiles + 1;
end
if( exist('train-images-idx3-ubyte', 'file') == 2 )
    mnistfiles = mnistfiles + 1;
end
if( exist('train-labels-idx1-ubyte', 'file') == 2 )
    mnistfiles = mnistfiles + 1;
end

if( mnistfiles < 4 )
    warning('Can not find mnist data files. Please download four data files from http://yann.lecun.com/exdb/mnist/ . Unzip and copy them to same folder as testMNIST.m');
    return;
end

[TrainImages TrainLabels TestImages TestLabels] = mnistread();

% if you want to test with small number of samples, 
% please uncomment the following
%{
TrainNum = 10000; % up to 60000
TestNum = 100; % up to 10000

TrainImages = TrainImages(1:TrainNum,:);
TrainLabels = TrainLabels(1:TrainNum,:);

TestImages = TestImages(1:TestNum,:);
TestLabels = TestLabels(1:TestNum,:);
%}

nodes = [784 800 800 10]; % just example!
bbdbn = randDBN( nodes, 'BBDBN' );
nrbm = numel(bbdbn.rbm);

opts.MaxIter = 1000;
opts.BatchSize = 100;
opts.Verbose = true;
opts.StepRatio = 0.1;
opts.object = 'CrossEntropy';

opts.Layer = nrbm-1;
bbdbn = pretrainDBN(bbdbn, TrainImages, opts);
bbdbn= SetLinearMapping(bbdbn, TrainImages, TrainLabels);

opts.Layer = 0;
bbdbn = trainDBN(bbdbn, TrainImages, TrainLabels, opts);

save('mnistbbdbn.mat', 'bbdbn' );




