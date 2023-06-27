clear

warning off

[x,t]=iris_dataset;

x = x';
t = t';

nodes = [4 3];

dnn = randDBN(nodes);

opts.MaxIter = 1000;
opts.BatchSize = 150/4;
opts.Verbose = true;
opts.StepRatio=0.1;
opts.DropOutRate=.05;



dnn =pretrainDBN(dnn,x,opts);
dnn =SetLinearMapping(dnn,x,t);

opts.Layer = 0;
dnn = trainDBN(dnn,x,t,opts);

rmse = CalcRmse(dnn,x,t)