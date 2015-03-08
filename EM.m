% get data and insert it to the matrix g
data = importdata('data.m');
n = size(data,1);
g = zeros(4,5);
% this is for inverting the numbers into vectors
for i=1:n
	for j=1:5
		g(i,j)=floor(data(i)/10^(5-j));
		data(i)=data(i)-floor(data(i)/10^(5-j))*10^(5-j);
	end
end

% generate t matrix were each row is a binary number from 1 to 32 by order
t = zeros(32,5);
for i=1:31
	t(i+1,:) = de2bi(i,5);
end

% initialize p - the probability of each binary number to appear
p = ones(1,32)/32;

% initialize a and f
a = zeros(n,32);
f = zeros(n,32);

% calculation of f
for i = 1:n
	for j = 1:32
		if sum(g(i,:)-t(j,:)==0) == 3
			f(i,j)=0.25;
		end
	end
end

% ######### start iterating over EM ###########
num_of_iter = 0;
p_old = zeros(1,32);
while norm(p_old-p)>0.001
	num_of_iter = num_of_iter + 1;
	% calculation of a
	for i = 1:n
		a(i,:) = p.*f(i,:);
		a(i,:) = a(i,:)./sum(a(i,:),2);
	end

	% maximization of p
	p_old = p;
	p = (sum(a))/n;
end
num_of_iter
p'
