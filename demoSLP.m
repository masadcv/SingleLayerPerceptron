% demonstration of a SingleLayeredPerceptron
% data files come from: http://cis.jhu.edu/~sachin/digit/digit.html
% help taken from: http://www.saedsayad.com/artificial_neural_network_bkp.htm
%
% The task is to differentiate between hand written digits 2 and 5
%
% We evaluate and show only the misclassification results
% 
% The architecture is simple - using just one single perceptron
%
%

close all
clear all
clc


% load the data
fid5 = fopen('data5', 'r');
fid2 = fopen('data2', 'r');

% lets just read 10 samples from each file
N = 100;

data = [];
dataLabel = [];

for i = 1:N
    [t1, N] = fread(fid5, [28 28], 'uint8');
    data = [data reshape(t1, N, 1)];
    dataLabel = [dataLabel 1];
    
    [t1, N] = fread(fid2, [28 28], 'uint8');
    data = [data reshape(t1, N, 1)];
    dataLabel = [dataLabel 0];
end

% lets define out SLP

% initialize randomly
w = rand(N, 1)-0.5;
b = rand;
learningRate = 0.1;
nEpochs = 100;

for i = 1:nEpochs
    for j = 1:size(data,2)
%         w
        predValD = w'*data(:, j) + b;
        predVal = 1/(1 + exp(-predValD));
        error = predVal - dataLabel(:, j);
        delW = (learningRate * error) .* data(:, j);
        w = w-delW;
    end
end


% make prediction on rest of the samples and see if our single layered
% perceptron worked?
correctPredict = [];
misClassifiedData = [];
while ~feof(fid5)
    [x, N] = fread(fid5, [28 28], 'uint8');
    if(~isempty(x))
        xR = reshape(x, N, 1);
        
        % run this through SLP
        predValD = w'*xR + b;
        predVal = 1/(1 + exp(-predValD));
        
        if predVal == 1
            correctPredict = [correctPredict 0];
        else
            correctPredict = [correctPredict 1];
            misClassifiedData = [misClassifiedData xR];
        end
    end
end

while ~feof(fid2)
    [x, N] = fread(fid2, [28 28], 'uint8');
    if(~isempty(x))
        xR = reshape(x, N, 1);
        
        % run this through SLP
        predValD = w'*xR + b;
        predVal = 1/(1 + exp(-predValD));
        
        if predVal == 0
            correctPredict = [correctPredict 0];
        else
            correctPredict = [correctPredict 1];
            misClassifiedData = [misClassifiedData xR];
        end
    end
end
fclose(fid5);
fclose(fid2);
overallAcc = (1-sum(correctPredict)/ size(correctPredict, 2))*100

% if you want to see the misclassified samples uncomment below
% for i = 1:size(misClassifiedData, 2)
%     imshow(reshape(misClassifiedData(:, i), 28, 28)); drawnow;
% end
