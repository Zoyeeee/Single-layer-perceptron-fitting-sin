# Single-layer-perceptron-fitting-sin
单层感知机拟合sin函数
```matlab
clear all;clc
rng('default');
%%
%训练集
x_Test=[0:0.00001:(2*3.14)];
IDtest=randperm(length(x_Test));
x_Test=x_Test(IDtest);

%y_Test=sin(x_Test)+randn(1,length(x_Test));
y_Test=sin(x_Test);
y_Test=BecomeFalseValue(y_Test);

figure(1)
plot(y_Test);
hold on
plot((sin(x_Test)+1)/2,'r');
hold off
%%
%验证集
x_Validation=[0:0.01:2*3.14];
y_Validation=sin(x_Validation);
y_Validation=BecomeFalseValue(y_Validation);


figure(2)
plot(y_Validation);
hold on
%%
%设定参数部分（调参全部在这边调）
%神经元细胞数目设定
num_cell=10;
%误差多少的时候停止(可不填)
Tolerance=0.0001;
%学习率
alpha=0.01;
%%
%bp神经网络部分
%%
%初始化各参数;(一边写一边加,防遗漏)
w1=randn(1,num_cell);
b1=randn(1,num_cell);
w2=randn(1,num_cell);
b2=randn(1,num_cell);
%%
%%
%时间百分比条的加入（百分比条网上代码）
bar = waitbar(0,'读取数据中...');    % waitbar显示进度条
%训练模型
for i=1:length(x_Test)
    MyFlag=1;
    counter=0;
    var_RemberDisdencecriterion=0;
    RemberDisdence=[99999,999999,9999999];
    while(MyFlag)
        tempy1=HindLayer(w1,b1,x_Test(i));
        y=Hind2Out(w2,b2,tempy1);
        if mod(counter,3)==0
            RemberDisdence(1)=y;
        elseif mod(counter,3)==1
            RemberDisdence(2)=y;
        elseif mod(counter,3)==2
            RemberDisdence(3)=y;
        end
            
        if (IfStop(y_Test(i),y,Tolerance)||VarianceCondition(y,RemberDisdence))%||(counter==100000)%暂停条件不止这个，应该再加入差距不再变化,以及计算次数
            MyFlag=0;
        else
            [w1,b1,w2,b2]=UpdateParameter(w1,b1,w2,b2,y_Test(i),tempy1,y,alpha,x_Test(i));
            counter=counter+1;
        end
    end
    str=['计算中...',num2str(100*i/length(x_Test)),'%'];    % 百分比形式显示处理进程,不需要删掉这行代码就行
    waitbar(i/length(x_Test),bar,str)                       % 更新进度条bar，配合bar使用
end

close(bar)
%%
%测试模型
fy=nan(1,length(x_Validation));
for i=1:length(x_Validation)
    tempy1=HindLayer(w1,b1,x_Test(i));
    fy(i)=Hind2Out(w2,b2,tempy1);
end

figure(2)
plot(fy,'r');
hold off
%%
%输出正确率
sum=0;
value=0;
for i=1:length(x_Validation)
    if abs(fy(i)-y_Validation)<=0.01
        value=1;
    else
        value=0;
    end
    sum=sum+value;
end

%输出最后百分比化的正确率
acc=(sum/length(x_Validation))
%%
%还原函数
function y=BecomeTrueValue(x)
    y=-1+x.*2;
end
%%
%变成Sigmoid函数
function y=BecomeFalseValue(x)
    y=(x+1)./2;
end
%%
%输入-隐藏层
%w=(1*10),b=(1,10),x=(1,1);y=(1,10)
function y=HindLayer(w,b,x)
    y=x.*w+b;
    y=1./(1+exp(-y));
end
%%
%隐藏层-输出-----注意这边真实用到了元的个数
%x=(1,10),b(1,10),w(1,10);y(1,1)
function y=Hind2Out(w,b,x)
    Sigma=0;
    for i=1:length(w)
        Sigma=(x(i)*w(i)+b(i))+Sigma;
    end
    y=1/(1+exp(-Sigma));
end
%%
%权重更新层
%w1,2=(1*10),b1,2=(1,10),TotalError=(1,1)
%此处x=(1,10),为隐藏层的基本输出，还没有wx+b那次的输出
%out=(1,1),为此次的输出（假）
%o=(1,1),为此次的输出，（真），标签数据
%alpha为学习率
%input为输入的数据
%各参数初始化部分，未写；直接随机就行，开局随机，然后直接带入也不存在问题感觉
function [w1,b1,w2,b2]=UpdateParameter(w1,b1,w2,b2,o,x,out,alpha,input)
    w2_old=w2;
    %w2的更新,基于Sigmoid
    for i=1:length(x)
        w2(i)=w2(i)-alpha*(-(o-out)*(out*(1-out)*x(i)));
    end
    %b2的更新,基于Sigmoid
    for i=1:length(x)
        b2(i)=b2(i)-alpha*(-(o-out)*(out*(1-out)));
    end
    %w1的更新,基于Sigmoid
    for i=1:length(x)
        w1(i)=w1(i)-alpha*((-(o-out)*(out*(1-out)))*w2_old(i)*x(i)*(1-x(i))*input);
    end
    %b1的更新,基于Sigmoid
    for i=1:length(x)
        b1(i)=b1(i)-alpha*((-(o-out)*(out*(1-out)))*w2_old(i)*x(i)*(1-x(i)));
    end
end
%%
%暂停条件1
%Tolerance指真实值和输出值的差，容忍的范围,其中容忍的范围为可选设置参数，对于最后准确率以及计算时间有关
function y=IfStop(x_false,x_true,Tolerance)

    if nargin<3
            Tolerance=0.0001;
    end

    if abs(x_true-x_false)<=Tolerance
        y=1;
    else
        y=0;
    end

end
%%
%暂停条件2
function var_RemberDisdencecriterion=VarianceCondition(y,RemberDisdence)
    if sum(RemberDisdence==[99999,999999,9999999])~=0
        var_RemberDisdencecriterion=0;
    elseif var(RemberDisdence)<=0.000001
        var_RemberDisdencecriterion=1;
        %最后需要调试的数据
    else
        var_RemberDisdencecriterion=0;
    end
end
%%
%输出误差计算;此处使用最简单的损失函数->TotalError
function y=OutputError(x_false,x_true)
    y=(1/2)*((x_false-x_true)^2);
end
```
