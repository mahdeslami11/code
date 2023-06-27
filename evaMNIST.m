clear all;

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

load mnistbbdbn;

rmse= CalcRmse(bbdbn, TrainImages, TrainLabels);
ErrorRate= CalcErrorRate(bbdbn, TrainImages, TrainLabels);
fprintf( 'For training data:\n' );
fprintf( 'rmse: %g\n', rmse );
fprintf( 'ErrorRate: %g\n', ErrorRate );

rmse= CalcRmse(bbdbn, TestImages, TestLabels);
ErrorRate= CalcErrorRate(bbdbn, TestImages, TestLabels);
fprintf( 'For test data:\n' );
fprintf( 'rmse: %g\n', rmse );
fprintf( 'ErrorRate: %g\n', ErrorRate );

