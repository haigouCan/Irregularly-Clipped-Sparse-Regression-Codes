clc; close all; clear;
% rng(43,'twister')
rng('shuffle')

load alpha.mat
CR =[-20:20];
SNRdB = ones(1,length(CR)) ;


alpha = ones(1,length(SNRdB));

max_noise_order = [1:length(SNRdB)];
% max_noise_order is using to choose which SNRs would be run. A longer step
% would helps saving time.

 
v_final_c = cell(1,length(max_noise_order));
u_final_c = cell(1,length(max_noise_order)); 
MSE_final_c = cell(1,length(max_noise_order));
SER_final_c = cell(1,length(max_noise_order));

v_final_c_test = cell(1,length(max_noise_order));
u_final_c_test = cell(1,length(max_noise_order));
MSE_final_c_test = cell(1,length(max_noise_order));
SER_final_c_test = cell(1,length(max_noise_order));



clip_in_final_c = cell(1,length(max_noise_order));
clip_out_final_c = cell(1,length(max_noise_order));
clip_pos_final_c = cell(1,length(max_noise_order));
N_in_final_c = cell(1,length(max_noise_order));
N_out_final_c = cell(1,length(max_noise_order));
a2=zeros(31,0);
a2(15)=1;
for h = 1:length(max_noise_order)
h

%% initialize parameters
B = 64;                     %length of block
L = 2048;                  %number of blocks
N = B*L;                   %length of whole message 
R = 1;                 %information rate
M = fix(L * log2(B) / R) ;       %legnth of sent message
rate = M/N;
dec_x = zeros(N,1);


alph_tem = alpha(:,max_noise_order(h));
% alph_tem is using to choose opted coefficients. All SNRs have their own
% coefficients so it's okay to use the same order as max_noise_order.
% But usually there is one best combination so it's also okay to use only
% one like:
% alph_tem = alpha(:,8);

alph_tem = max(0,alph_tem);

rep_times = 50 ;           %repeat times
N_ite_floor =30 ;                %iteration times


SNRdB_tem = SNRdB(max_noise_order(h));
v_n = (10.^(SNRdB_tem/10)).^(-1);      % variance of noise
sigma = sqrt(v_n);

% CR = [-22, -20, -15, -13, -10, -5, -2, 0, 1, 3, 5, 6, 10, 20, 30];        %clipping ratio
lambda = 10.^(CR/20);
   

    
full_clip_x = zeros(N,length(CR));
full_x_dct = zeros(N,1);

clip_in_final = zeros(N_ite_floor,length(v_n));
clip_out_final = zeros(N_ite_floor,length(v_n));
clip_pos_final = zeros(N_ite_floor,length(v_n));
N_in_final = zeros(N_ite_floor,length(v_n));
N_out_final = zeros(N_ite_floor,length(v_n));

clip_in_final_test = zeros(N_ite_floor,length(v_n));
clip_out_final_test = zeros(N_ite_floor,length(v_n));
clip_pos_final_test = zeros(N_ite_floor,length(v_n));
N_in_final_test = zeros(N_ite_floor,length(v_n));
N_out_final_test = zeros(N_ite_floor,length(v_n));



v_final = zeros(N_ite_floor,length(v_n));
u_final = zeros(N,length(sigma));
MSE_final = zeros(N_ite_floor,length(v_n));
SER_final = zeros(N_ite_floor,length(v_n));

v_final_test = zeros(N_ite_floor,length(v_n));
u_final_test = zeros(N,length(sigma));
MSE_final_test = zeros(N_ite_floor,length(v_n));
SER_final_test = zeros(N_ite_floor,length(v_n));

label = 0;
for k2=1:rep_times
  
% length4M =fix( alph_tem * M );
% % length4each = [0 0 0 M];
% num = find(length4M > 0);
% dif = M - sum(length4M);
% length4M(num(1:min(dif,length(num)))) = length4M(num(1:min(dif,length(num)))) + ceil(dif/sum(length(num)));

length4N =fix( alph_tem * N );
length4N(find(length4N<=10)) = 0;
% length4each = [0 0 0 M];
numN = find(length4N > 0);
difN = N - sum(length4N);
length4N(numN(1:min(difN,length(numN)))) = length4N(numN(1:min(difN,length(numN)))) + fix(difN/sum(length(numN)));
difN = N - sum(length4N);
length4N(numN(1:min(difN,length(numN)))) = length4N(numN(1:min(difN,length(numN)))) + ceil(difN/sum(length(numN)));

length4M = fix(length4N * rate);
num = find(length4M > 0);
M = sum(length4M);
%% source          
data=randi([1,B],L,1);

%% Encoder
[full_x_dct,i_dct,x,D]=enc(data,B,L,M);   % x_dct = x(i_dct)
rand_x = x(i_dct);
x_dct = full_x_dct(i_dct);

%% clip
index_M = randperm(M);
last_v_ext = 1;
iN = cell(1,length(numN));  % xdct_i = x(iN{i})
iM = cell(1,length(numN));  % xdct_i = x_dct(iM{i})
for i = 1:length(numN)

    ind = index_M(last_v_ext:last_v_ext + length4M(numN(i)) - 1);
    last_v_ext = last_v_ext + length4M(numN(i)) ;
    iN{i} = i_dct(ind) ; % for N
    iM{i} = ind;  % for M
end

index_0 = [1:N];
for i = 1:length(iN)
    index_0 = setdiff(index_0,iN{i});
end
index_0 = index_0(randperm(N-M));
last_v_ext = 1;

N_index = cell(1,length(numN));
length4n = length4N - length4M;
for i = 1:length(numN)
    ind = index_0(last_v_ext:last_v_ext + length4n(numN(i)) - 1);
    last_v_ext = last_v_ext + length4n(numN(i)) ;
    N_index{i} = ind;
end

%% test
%  for i = 1:length(num)
%      a{i} = union( N_index{i},index_cr{i});
%  end
% b=[];
%  for i = 1:length(num)
%     b = union(b,a{i});
%  end
% a2 = intersect(N_index{1},N_index{2});

%%
full_clip_x = clip_res(full_x_dct,iN,lambda(h));  
clip_x = full_clip_x(i_dct);
% a = [];
% for i = 1:length(index_cr)
%     a = [a,index_cr{i}];
% end

%% channel noise and norm
[y,v_neq] = channel_noise(clip_x,v_n,iM,B);

%% OAMP Decoding
clip_in = zeros(N_ite_floor,length(v_n));
clip_out = zeros(N_ite_floor,length(v_n));
clip_pos = zeros(N_ite_floor,length(v_n));
N_in = zeros(N_ite_floor,length(v_n));
N_out = zeros(N_ite_floor,length(v_n));


v_ite = ones(N_ite_floor + 1,length(v_n));
u_ite = zeros(N,length(sigma));
v_ite_test = ones(N_ite_floor + 1,length(v_n));
u_ite_test = zeros(N,length(sigma));

u_pos_clip = zeros(M,length(v_n));
v_pos_clip = zeros(N_ite_floor,length(v_n));
v_pos_clip_tem = zeros(length(iN),length(v_n));
v_ext = ones(N_ite_floor ,length(v_n));%%
v_ext_tem = ones(length(iN) ,length(v_n));%%
MSE_clip = zeros(N_ite_floor,length(v_n));
u_pos_clip_test = zeros(M,length(v_n));
v_ext_test = ones(N_ite_floor ,length(v_n));%%
MSE_clip_test = zeros(N_ite_floor,length(v_n));

v_pos_N = zeros(N_ite_floor,length(v_n));
MSE_N = zeros(N_ite_floor,length(v_n));
SER_N = zeros(N_ite_floor,length(v_n));
SER_N_test = zeros(N_ite_floor,length(v_n));
v_pos_N_test = zeros(N_ite_floor,length(v_n));
MSE_N_test = zeros(N_ite_floor,length(v_n));

v_result = ones(N_ite_floor,length(v_n));
MSE_result = zeros(N_ite_floor,length(v_n));

u_ext_tem = zeros(N,length(iN));

last_v_ext = 10;
last_v_ite = 10;
    for j=1:N_ite_floor 

    %% clip test
        % de-clip for every different CR
%         
%         
%         for i = 1:length(index_cr)
%             index_lambda = index_cr{i};
%             y_in = y(M_index{i},:);
%             [u_pos_clip_test(M_index{i},:), v_pos_clip_tem(i,:) ] = Z_APP_Clip(lambda(num(i)), y_in, u_ite_test(index_lambda,:), v_ite_test(j,:), v_neq(i));  
%             [u_ext_tem(:,i),v_ext_tem(i,:)]  = ext(N,length4N(num(i)),u_ite_test, v_ite_test(j,:),u_pos_clip_test(M_index{i},:),v_pos_clip_tem(i,:),[],index_lambda,N_index{i});
%         end
% 
% 
%         v_ext_test(j,:) = alphe_9(num)' * v_ext_tem;
%         u_ext_test = sum(u_ext_tem,2);
% 
%         
%         MSE_clip_test(j,:) = mean((u_pos_clip_test - x_dct).^2);
        
            %%  clip

                for i = 1:length(iN)
                    index_lambda = iN{i};
                    y_in = y(iM{i},:);
                    [u_pos_clip(iM{i},:), v_pos_clip_tem(i,:) ] = Z_APP_Clip(lambda(h), y_in, u_ite(index_lambda,:), v_ite(j,:), v_neq(i));  
                    
                end
                % average the variance of each CR

                v_pos_clip(j,:) = var_average(length4M(numN),v_pos_clip_tem); 
        %                 v_pos_clip(j,:) = max(v_pos_clip(j,:),1e-7);
                if j==19
                     ssss=1;
                end
                        last = last_v_ext;
                        [u_ext,last_v_ext]  = external_information(u_ite, v_ite(j,:),u_pos_clip,v_pos_clip(j,:),i_dct);
                        v_ext(j,:) = last_v_ext;
        %                 if v_ext(j,:)>last
        %                     v_ext(j,:) = v_ext(j-1,:);
        %                     last_v_ext = last;
        %                 end

                        MSE_clip(j,:) = mean((u_pos_clip - x_dct).^2);
                        if isnan(v_pos_clip(j,:))
                            assss=1;
                end

    %% NLD test
%         u_ext_test = idct(u_ext_test);
% 
%         
%         [u_pos_N_test, v_pos_N_test(j,:),decision_test] = NLD_LOOP(u_ext_test,v_ext_test(j,:),B,N);  
%         v_pos_N_test(j,:) = max(v_pos_N_test(j,:),1e-06);
%         [u_ite_test,v_ite_test(j + 1,:)]  = external_information(u_ext_test, v_ext_test(j,:),u_pos_N_test,v_pos_N_test(j,:));
% 
%         u_ite_test = dct(u_ite_test);
%         MSE_N_test(j,:) = mean((u_pos_N_test - x).^2);
%         a_tem = ((0:L-1)'*B+data - decision_test);
%         SER_N_test(j,:) = SER_N_test(j,:) + length(find(a_tem ~= 0));
%         
% 
%         if v_pos_N(j,:) <= 1e-5
%           break
%         end
 %% NLD
                u_ext = D .* idct(u_ext);


                [u_pos_N, v_pos_N(j,:),decision] = NLD_LOOP(u_ext,v_ext(j,:),B,N);  
                v_pos_N(j,:) = max(v_pos_N(j,:),1e-8);
                [u_ite, last_v_ite]  = external_information(u_ext, v_ext(j,:),u_pos_N,v_pos_N(j,:));
                v_ite(j + 1,:) = last_v_ite;
                u_ite = dct(D .* u_ite);
                MSE_N(j,:) = mean((u_pos_N - x).^2);
                a_tem = ((0:L-1)'*B+data - decision);
                SER_N(j,:) = SER_N(j,:) + length(find(a_tem ~= 0));
               if v_pos_N(j,:) <= 1e-5
                    break
               end
    end
v_pos_N([j+1:end],:) = v_pos_N(j,:);
MSE_N([j+1:end],:) = MSE_N(j,:);
SER_N([j+1:end],:) = SER_N(j,:);

if v_pos_N(end) > 1e-3
    label = label + 1;
end

v_final = v_final + v_pos_N/rep_times;
u_final = u_final + u_pos_N/rep_times;
MSE_final = MSE_final + MSE_N/rep_times;
SER_final = SER_final + SER_N/rep_times;
% 
% v_final_test = v_final_test + v_pos_N_test/rep_times;
% u_final_test = u_final_test + u_pos_N_test/rep_times;
% MSE_final_test = MSE_final_test + MSE_N_test/rep_times;
% SER_final_test = SER_final_test + SER_N_test/rep_times;

clip_in_final = clip_in_final + v_ite(1:end-1,:)/rep_times;
clip_out_final = clip_out_final + v_ext/rep_times;
clip_pos_final = clip_pos_final + v_pos_clip/rep_times;
N_in_final = N_in_final + v_ext/rep_times;
N_out_final = N_out_final + v_ite(2:end,:)/rep_times;
end
v_final_c{h} = v_final;
u_final_c{h} = u_final;
MSE_final_c{h} = MSE_final;
SER_final_c{h} = SER_final/L;

% v_final_c_test{h} = v_final_test;
% u_final_c_test{h} = u_final_test;
% MSE_final_c_test{h} = MSE_final_test;
% SER_final_c_test{h} = SER_final_test/L;


clip_in_final_c{h} = clip_in_final;
clip_out_final_c{h} = clip_out_final;
clip_pos_final_c{h} = clip_pos_final;
N_in_final_c{h} = N_in_final;
N_out_final_c{h} = N_out_final;
end
save test.mat
%% plot
% load opt_clip_data_12
minv = zeros(length(max_noise_order),1);
minMSE = zeros(length(max_noise_order),1);
minSER = zeros(length(max_noise_order),1);
for i = 1:length(max_noise_order)
    v_tem = v_final_c{i};
    [minv_tem,ind_tem] = mink(v_tem,5);
    minv(i) = mean(minv_tem);
    MSE_tem = MSE_final_c{i};
    minMSE(i) = mean(MSE_tem(ind_tem));
    SER_tem = SER_final_c{i};
    minSER(i) = mean(mink(SER_tem,5));
end
a= [1,4,6:9,11,12];
semilogy(SNRdB(a),minSER(a),'r.-','LineWidth',2,'MarkerSize',25)
xlim([0,5])
ylim([1e-5,1])
% MSE_final = max(MSE_final,1e-6);
% SER_final = max(SER_final_c{1},1e-6);
% 
%     figure
%     semilogy([1:N_ite_floor],v_final','LineWidth',2)
%     hold on
%     semilogy([1:N_ite_floor],MSE_final','*')
%     semilogy([1:N_ite_floor],SER_final','s')
%     xlabel('iteration times')
%     ylabel('MSE')
%     legend('v','MSE','SER')
    

function [x_dct,index,x,D]= enc(data,B,L,M)
N = L * B;
sq_B=B^0.5;
x=zeros(L*B,1);
s=(0:L-1)';
s=s*B+data;
x(s) = sq_B;
% for i=1:L
%     x(data(i)+1+B*(i-1))=sq_B;    
% end
D = 2 * randi(2,N,1) - 3;
x_ten = D .* x;
%% dct
x_dct=dct(x_ten);
%% random
index = randperm(N,M);
% x_dct = x_dct(index);
end
function [clip_x] = clip_res(x,index_cr,lambda)
clip_x = zeros(size(x));
    for ii = 1:length(lambda)
        index_lambda = index_cr{ii};
        A = lambda(ii);
        clip_x(index_lambda)  = max(min(x(index_lambda),A), -A); 
    end
end
function [receive,v_n_eq] = channel_noise(clip_x,v_n,index_cr,B)
    noise = zeros(length(clip_x), length(v_n));
    v_n_eq = zeros(length(index_cr),length(v_n));
    for iii = 1:length(index_cr)
        index = index_cr{iii};
        v_n_eq(iii,:) =  mean(clip_x(index).^2/1)' .* v_n;
        noise(index,:) = sqrt(v_n_eq(iii,:))  .* randn(length(index),length(v_n)); 
    end
    receive=clip_x + noise ; 
end

function [m_out, v_out,v_pos,m_pos] = LD_LOOP(m_pri,v_pri,N,M,sigma,y_tem,index)
a=v_pri<=2.186881032889819e-6;
v_pri(a)=2.186881032889819e-6;

v_out = zeros(size(sigma));
m_out = zeros(N, length(sigma));
v_pos = zeros(size(sigma));
m_pos = zeros(N, length(sigma));

for i=1:length(sigma) 
    
    v_pos(i) = ( (N-M)*v_pri(i) + M/(sigma(i)^-1 + v_pri(i)^-1) )/N; %%%%%%
   
    Dia_inv = (sigma(i)^-1 + v_pri(i)^-1)^-1 * ones(N,1);
    Dia_inv(index(1:N-M)) = v_pri(i) * ones(N-M,1);
    
    m_pos(:,i) = sigma(i)^-1 * y_tem(:,i) + v_pri(i)^-1*dct(m_pri(:,i));
    m_pos(:,i) = idct( Dia_inv .* m_pos(:,i) );
    
    [m_out(:,i),v_out(i)] = Post_pri(m_pos(:,i), v_pos(i), m_pri(:,i), v_pri(i));
    
end
end

function [mean_pos,var_pos,Posi] = NLD_LOOP(m_pri,var_pri,B,N)
%% initialize parameters 
len_sigma=length(var_pri);

Posi = zeros(N/B,len_sigma);
o=B^0.5*eye(B);     %B type of source codeword
P_pos = zeros(size(m_pri));

%% compute posteerior probability

for j=1:len_sigma
    sigma=var_pri(j);
    position = zeros(1,N/B);
    for i=1:B:N
        
        
        block_B=m_pri(i:i+B-1,j);
        block_B=repelem(block_B,1,B);
        % take one block and copy to the same number with sigma
        
        P=exp(-(block_B-o).^2/(2*sigma));  
        P=sum(log10(P));
        
        sa=P<-90;
        P(sa)=-90;
        sb=P>=90;
        P(sb)=90;
        P=10.^P; 
        % set a threshold to keep probability not too small
        
        sum_P=sum(P);
        P=P./sum_P;
        [~,p]=max(P);
        position((i-1)/B+1)=p+((i-1)/B)*B;
        
        % normalization
        %var((i-1)/B+1,j)=sum(P.*B.*(1-P))/B;
        
        P_pos(i:i+B-1,j)=P;
      
         
    end
    Posi(:,j)=position';
end

        
%% post output  
mean_pos = B^0.5*P_pos;                      %posterior mean
var_pos = mean(B * P_pos.*(1-P_pos));        %posterior variacne averaged by N bits
% var_pos = max(var_pos,1e-6);
%% orthogonal output 
% var_out = (var_pos.^-1-var_pri.^-1).^-1;
% for i=1:length(var_out)
%     mean_out(:,i) = (var_pos(i).^-1.*mean_pos(:,i)-var_pri(i).^-1.*m_pri(:,i)) ... 
%         .*var_out(i);
% end 

end



