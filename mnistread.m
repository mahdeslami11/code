function [ TrainImages, TrainLabels, TestImages, TestLabels ] = mnistread( mnistfilenames )

if( ~exist('mnistfilenames') )
    mnistfilenames = cell(4,1);
    mnistfilenames{1} = 'train-images-idx3-ubyte';
    mnistfilenames{2} = 'train-labels-idx1-ubyte';
    mnistfilenames{3} = 't10k-images-idx3-ubyte';
    mnistfilenames{4} = 't10k-labels-idx1-ubyte';
end

TrainImages = mnistimageread( mnistfilenames{1} );
TrainLabels = mnistlabelread( mnistfilenames{2} );
TestImages = mnistimageread( mnistfilenames{3} );
TestLabels = mnistlabelread( mnistfilenames{4} );

TrainImages = single(TrainImages)/255.0;
TestImages = single(TestImages)/255.0;

TrainLabels = single(TrainLabels);
TestLabels = single(TestLabels);

end

function images = mnistimageread( imagefile )
    fid = fopen( imagefile, 'rb');
    magic = fread(fid, 1, '*int32',0,'b');
    nimgs = fread(fid, 1, '*int32',0,'b');
    nrows = fread(fid, 1, '*int32',0,'b');
    ncols = fread(fid, 1, '*int32',0,'b');
    images = fread(fid, inf, '*uint8',0,'b');
    fclose( fid );
    
    if( magic ~= 2051 )
        warning( sprintf( '%s is not MNIST image file.', imagefile ) );
        images = [];
        return;
    end
    
    images = reshape( images, [nrows*ncols, nimgs] )';
    for i=1:nimgs
        img = reshape( images(i,:), [nrows ncols] )';
        images(i,:) = reshape(img, [1 nrows*ncols]);
    end
end

function labels = mnistlabelread( labelfile )
    fid = fopen( labelfile, 'rb');
    magic = fread(fid, 1, '*int32',0,'b');
    nlabels = fread(fid, 1, '*int32',0,'b');
    ind = fread(fid, inf, '*uint8',0,'b');
    fclose( fid );
    
    if( magic ~= 2049 )
        warning( sprintf( '%s is not MNIST label file.', labelfile ) );
        labels = [];
        return;
    end
    
    labels = zeros( nlabels, 10 );
    ind = ind + 1;
    for i=1:nlabels
        labels(i, ind(i)) = 1;
    end
end


