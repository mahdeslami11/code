clear

addpath('DeepNeuralNetwork')


num = 1000;

IN = rand(num,32);
OUT = rand(num,4);

nodes =[32 16 4];

dnn = randDBN(nodes);


opts.MaxIter = 1000;
opts.BatchSize = num/4;
opts.Verbose = true;
opts.StepRatio=0.1;
opts.DropOutRate=.05;
opts.object = 'CrossEntropy';



dnn =pretrainDBN(dnn,IN,opts);
dnn =SetLinearMapping(dnn,IN,OUT);

dnn = trainDBN(dnn,IN,OUT,opts);

rmse = CalcRmse(dnn,IN,OUT)