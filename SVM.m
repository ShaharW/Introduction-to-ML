clc
% get data, normalize it and split it to x and y
fileID = fopen('data.txt');
C = textscan(fileID,'%s %f %f %f%f %f %f %f %f %s');
fclose(fileID);
predicted = C{10};
x = zeros(size(predicted,1),8);
for i = 2:9
        x(:,i-1) = C{i};
end

y = ones(size(predicted,1),1);
m = size(x,2);
n = size(x,1);

% [-1,1] normalization
for i = 1:m
	minimum = min(x(:,i));
	maximum = max(x(:,i));
	factor = 2/(maximum-minimum);
	x(:,i) = ones(n,1)*(-1) + factor*(x(:,i)-ones(n,1)*minimum);
end

% set predicted vector
for i = 1:n
	if strcmp(predicted(i) ,'CYT')==1
		y(i) = 1;
	else
		y(i) = -1;
	end
end

test_size = ceil(n/10);
lowest_err = 100;
% run svm
addpath('/a/home/cc/students/csguests/shaharwi/Public/Ex4/libsvm-3.17/matlab')
best_logc_per_deg = [];
for deg = 1:4
    best_err_per_deg = 1;
    % plot arrays
    x_axis = [];
    ub = [];
    err = [];
    lb = [];
    for logc = 1:11
		crs_val_err = [];
		for i = 1:10 % cross validation
			% create a different test and training samples
			if test_size*i > n % in order not to exceed the matrix size
				UB = n;
				LB = n - test_size + 1;
			else
				UB = test_size*i;
				LB = test_size*(i-1) + 1;
			end
			XTR = x;
			XTS = x(LB:UB,:); % test set
			XTR(LB:UB,:) = []; % training set
			YTR = y;
			YTS = y(LB:UB,:);
			YTR(LB:UB,:) = [];
			
			% run svm
			opt = ['-c ',num2str(2^(logc-1)),' -t 1 -d ',num2str(deg),' -g 1'];
			model = svmtrain(YTR, XTR, opt);
			[predict_label, accuracy, dec_values] = svmpredict(YTS, XTS, model); % test the training data
            crs_val_err = [crs_val_err,(100-accuracy(1))/100];
        end
		M = mean(crs_val_err);
		STD = sqrt((1/9)*sum((M.*ones(1,10) - crs_val_err).^2));
        x_axis = [x_axis logc-1];
        ub = [ub M+STD];
        err = [err M];
        lb = [lb M-STD];
        
        if M < best_err_per_deg
            best_err_per_deg = M;
            best_logc_per_deg(deg) = logc-1;
            if M < lowest_err
               lowest_err = M;
               best_logc = logc-1;
               best_deg = deg;
            end
        end
    end
    figure('Name',['Polynomial degree = ',num2str(deg)])
    plot(x_axis,lb,x_axis,ub,x_axis,err);
    title(['Polynomial degree = ',num2str(deg)])
    xlabel('LogC')
    ylabel('Error')
end

% second assignment - plot the error as a function of d with the best c
% found previously
% plot arrays
x_axis = [];
test_err = [];
train_err = [];
nSV = [];
nsv_margin = [];
margin = [];

for deg = 1:4
    test_err_CV = [];
    train_err_CV = []; 
    nsv_CV = [];
	margin_CV = [];
	nsv_margin_CV = [];
    for i = 1:10 % cross validation
        % create a different test and training samples
        if test_size*i > n % in order not to exceed the matrix size
            UB = n;
            LB = n - test_size + 1;
        else
            UB = test_size*i;
            LB = test_size*(i-1) + 1;
        end
        XTR = x;
        XTS = x(LB:UB,:); % test set
        XTR(LB:UB,:) = []; % training set
        YTR = y;
        YTS = y(LB:UB,:);
        YTR(LB:UB,:) = [];

        % run svm
        opt = ['-c ',num2str(2^best_logc),' -t 1 -d ',num2str(deg),' -g 1'];
        model = svmtrain(YTR, XTR, opt);
		[predict_label, accuracy, dec_values] = svmpredict(YTR, XTR, model); % test the training data
		train_err_CV = [train_err_CV,(100-accuracy(1))/100];
        [predict_label, accuracy, dec_values] = svmpredict(YTS, XTS, model); % test the test data
        test_err_CV = [test_err_CV,(100-accuracy(1))/100];
		
		%support vectors
		nsv_CV = [nsv_CV sum(model.nSV)];
		nsv_margin_CV = [nsv_margin_CV sum(abs(model.sv_coef) < 2^best_logc)];
		% calculate coefficient for margin calculation
		coefficient = zeros(size(model.sv_coef, 1), 1);		
		for j = 1 : size(coefficient, 1)
			if abs(model.sv_coef(j)) ~= 2^best_logc
				coefficient(j) = model.sv_coef(j); 
			end 
		end
		margin_CV = [margin_CV (1/norm(model.SVs'*coefficient))];
    end
	% arrays for plot
    x_axis = [x_axis deg];
	train_err = [train_err mean(train_err_CV)];
	test_err = [test_err mean(test_err_CV)];
	nSV = [nSV mean(nsv_CV)];
	nsv_margin = [nsv_margin mean(nsv_margin_CV)];
	margin = [margin mean(margin_CV)];
	
end
% plot the error for each degree
figure('Name','Error for each polynomial degree');
plot(x_axis,train_err,x_axis,test_err);
hleg1 = legend('training error','testing error');
title('Error for each polynomial degree');
xlabel('degree');
ylabel('Error');

% plot the support vectors
figure('Name','Support Vectors');
plot(x_axis,nSV,x_axis,nsv_margin);
hleg1 = legend('Support Vectors','SVs on Margin');
title('Number of Support Vectors');
xlabel('degree');
ylabel('Number of SVs');
margin