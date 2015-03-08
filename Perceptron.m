% get data and initialize parameters
x_train_data = importdata('x_train.m');
y_train_data = importdata('y_train.m');
x_test_data = importdata('x_test.m');
y_test_data = importdata('y_test.m');

n = size(x_train_data,1);
tn = size(x_test_data,1);
% adding -1 for each row in order to create a threshold
x_train = [-1*ones(n,1),x_train_data];
x_test = [-1*ones(tn,1),x_test_data]; 

m = size(x_train,2);
p = randperm(n); % randomize the training set
x_train = x_train(p,:);
y_train = y_train_data(p,:);
w = zeros(m,1);
train_errors = [];
test_errors = 0;
loop_errors = 1;

% perceptron loop
while loop_errors > 0
	loop_errors = 0;
	for i = 1:n
		x_train(i,:) = x_train(i,:)/norm(x_train(i,:)); % normalize x
		if x_train(i,:)*w >= 0
			if y_train(i) == -1 % error detected
				w = w - x_train(i,:)';
				train_errors = train_errors + 1;
			end
		else	
			if y_train(i) == 1 % error detected
				w = w + x_train(i,:)';
				loop_errors = loop_errors + 1;
			end
		end
	end
	train_errors = [train_errors, loop_errors];
end
train_errors

% test perceptron
for i = 1:tn
	x_test(i,:) = x_test(i,:)/norm(x_test(i,:)); % normalize x
	if x_test(i,:)*w >= 0
		prediction = 1;
	else	
		prediction = -1;
	end
	if prediction ~= y_test_data(i)
		test_errors = test_errors + 1;
	end
end
accuracy = 1-test_errors/tn