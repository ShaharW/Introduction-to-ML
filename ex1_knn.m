data = importdata('data.m');

% get number of examples and the data dimension
n = size(data,1);
m = size(data,2);

% shuffle entries and separate predictors from predicted values
P = randperm(n);
X = data(P,2:m-1);
Y = data(P,m);

test_size = ceil(n/10); % split the data to 10 fold cross
false_percent = zeros(1,25); % initialize the returned vector

for k = 1:25 % run on each k number of nearest neighbors
	total_penalty = 0;
	total_tests = 0;
	for i=1:10 % run on each fold cross
		fc_penalty = 0;
		
		% create a different test and training samples
		if test_size*i > n % in order not to exceed the matrix size
			UB = n;
			LB = n - test_size + 1;
		else
			UB = test_size*i;
			LB = test_size*(i-1) + 1;
		end
		XTR = X;
		XTS = X(LB:UB,:); % test set
		XTR(LB:UB,:) = []; % training set
		YTR = Y;
		YTS = Y(LB:UB,:);
		YTR(LB:UB,:) = [];
		
		% go over each test sample
		for j=1:test_size
			predicted = 0;
			knn = zeros(1,k);
			xs = XTS(j,:);
			ys = YTS(j,:);
			% D(i,:) is XTR(i,:) - xs
			D = XTR - repmat(xs, size(XTR,1), 1);
			% get the knn values for the test sample
			for t = 1:k
				[value,index] = min(sum(D.^2,2)); % minimal value in D
				knn(1,t) = YTR(index); % insert the value to the knn array
				D(index,:) = []; % delete the row from D in order to get to the next one
			end
			predicted = mode(knn); % the predicted value is the most common value in knn

			% add 1 to the fold cross penalty if the nearest neighbor label is not equal to the test label
			if ys == predicted
				penalty = 0;
			else
				penalty = 1;
			end
			fc_penalty = fc_penalty + penalty;
		end
		% add the fold cross results to the total sum
		total_penalty = total_penalty + fc_penalty;
		total_tests = total_tests + test_size;
	end
	false_percent(1,k) = total_penalty/total_tests;
end
plot(false_percent);
xlabel('K - # of neighbors');
ylabel('Percent of false prediction');
title('False Predictions','FontWeight','bold');

false_percent
