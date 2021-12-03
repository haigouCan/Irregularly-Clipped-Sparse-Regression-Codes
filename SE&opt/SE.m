clear,clc,close all;  
rng('shuffle')
R = 0.5 ;   
L = 2048;
B = 64 ;
N = L * B;
M =fix( L * log2(B) / R);         

rep_times = 80;
SNRdB = [11:0.1:13];
v_n = (10.^(SNRdB/10)).^(-1);
v_x = 1;

CR = [-300,-30:30,300];

lambda = 10.^(CR/20);

rho_pri =[0:60];      %dB
v_pri = 1./( 10.^(rho_pri/10) );  % rho =v^-1
x_pri = zeros(N, length(rho_pri));

len_lambda = length(lambda);
len_v_pri = length(v_pri);
r = zeros(N,1);

v_pos_output = zeros(length(lambda),length(rho_pri),length(v_n));
v_ext_output = zeros(length(lambda),length(rho_pri),length(v_n));
MSE_ext_output = zeros(length(lambda),length(rho_pri),length(v_n));



% rng('default'); 

%% Quantization
for k = 1:rep_times
    k
    lambda = 10.^(CR/20);
    MSE_ext = zeros(length(lambda),length(rho_pri),length(v_n));
    v_ext = zeros(length(lambda),length(rho_pri),length(v_n));
    v_pos = zeros(length(lambda),length(rho_pri),length(v_n));
    for ii = 1:length(v_n)
        lambda = 10.^(CR/20);
        v_pri = 1./( 10.^(rho_pri/10) );
        v_pos_tem_2 = zeros(length(lambda),length(rho_pri));
        v_ext_tem_2 = zeros(length(lambda),length(rho_pri));
        MSE_ext_tem_2 = zeros(length(lambda),length(rho_pri));
        for j = 1:len_lambda
            %lambda(j)
            A = lambda(j);
            v_pos_tem = zeros(1,length(rho_pri));
            v_ext_tem = zeros(1,length(rho_pri));
            MSE_ext_tem = zeros(1,length(rho_pri));
            for i = 1:len_v_pri
               
                %% source
                x_pri = sqrt(1-v_pri(i)) * randn(N,1);
                n_pri = sqrt(v_pri(i)) * randn(N,1);
                full_x = x_pri + n_pri;
                %% FFT 
                index = randperm(N);
                index = index(1:M);
                x = full_x(index);

                %% clipping 
                clip_x  = max(min(x,A), -A);  

                %% channel noise
                v_n_eq =  mean(clip_x.^2) * v_n(ii); 
                noise = sqrt(v_n_eq) * randn(M,1); 
                y = clip_x + noise; 

                %% de-clipping

                [u_pos, v_pos_tem(i)] = Z_APP_Clip(A, y, x_pri(index), v_pri(i), v_n_eq);
                [u_ext,v_ext_tem(i)]  = external_information(x_pri,v_pri(i),u_pos,v_pos_tem(i),index);
                
                %% error  
                MSE_ext_tem(i) = mean((u_ext - full_x).^2); 
            end   
            v_pos_tem_2(j,:) = v_pos_tem;
            v_ext_tem_2(j,:) = v_ext_tem;
            MSE_ext_tem_2(j,:) = MSE_ext_tem;
        end
        v_pos(:,:,ii) = v_pos_tem_2;
        v_ext(:,:,ii) = v_ext_tem_2;
        MSE_ext(:,:,ii) = MSE_ext_tem_2;
    end
    v_pos_output = v_pos_output + v_pos / rep_times;
    v_ext_output = v_ext_output + v_ext / rep_times;
    MSE_ext_output = MSE_ext_output + MSE_ext / rep_times;
end

%% plot
% 
save result_SE.mat
%  figure
% for i = 1:length(CR)
%     semilogy(v_ext_output(i,:,1),v_pri)
%     hold on
% end
%  hold on
% semilogy(MSE_ext(:,:,10),v_pri,'*')
%  save r=0.2.mat;



