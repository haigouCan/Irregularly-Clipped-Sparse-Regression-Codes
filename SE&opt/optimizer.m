clear,clc
load  result_SE.mat
load Table_SR_B64.mat
rho = Table(:,1);
v_pri_table=Table(:,1).^-1;
v_post_table=Table(:,2);
v_ext_table = (v_post_table.^(-1) - v_pri_table.^(-1)).^(-1);
v_post_clip = v_pos_output;
a=v_pri_table;
b=v_ext_table;




sigma = M/N;
zero_range = [];
zero_ind = [];
v_post_clip(setdiff(zero_range,zero_ind),:,:) = [];
CR(setdiff(zero_range,zero_ind)) = [];
[cr,~,~] = size(v_post_clip);
x_res = cell(length(v_n),1);
y_res = cell(length(v_n),1);
for k = 1:length(SNRdB)
    v_post_clip_log = log10(v_post_clip(:,:,k));
    v_pri_log = log10(v_pri);
    v_pri_Table_log = log10(v_pri_table(50:end));
    VAR_ext_Table_log = log10(v_ext_table(50:end));
%     figure
%     plot(SNR_Table_log,VAR_POST_Table_log)
%     hold on

    inf_check = isinf(VAR_ext_Table_log);
    inf_label = find(inf_check == 1);
    VAR_ext_Table_log(inf_label) = [];
    v_pri_Table_log(inf_label) = [];
    
    y_log = [-5:0.01:0];

    sample_clip_x = interp1(v_pri_log,v_post_clip_log',y_log);  
    sample_mod = interp1(VAR_ext_Table_log,v_pri_Table_log,y_log); 

;
    sample_x_clip = 10.^sample_clip_x;
    sample_mod = 10.^sample_mod;

    N_index = find(sum(isnan(sample_x_clip)) == cr);
    sample_x_clip(:,N_index) = [];
    sample_mod(:,N_index) = [];
    y_log(N_index) = [];
    y_real = 10.^y_log;
%     plot(y_real,sample_linear,'r','LineWidth',2)
%     hold on
%     plot(y_real,sample_x_clip)   

    cnan_ind = isnan(sample_x_clip);
    lnan_ind = isnan(sample_mod);
    nan_ind = find(sum(cnan_ind' +lnan_ind) ~= 0);
    
    sample_x_clip(nan_ind,:) = [];
    sample_mod(nan_ind) = [];
    y_real(nan_ind) = [];
    sample_x_clip=sample_x_clip';
    
    
    % opt parameters
    Aeq = ones(1,cr);
    beq = 1;
%     A = -1 * eye(cr);
%     b = zeros(cr,1);
    x0 = zeros(cr,1);
    lb = zeros(1,cr)';
    ub = ones(1,cr)';
    
    
    op = optimoptions('fminimax','StepTolerance',1e-15,'ConstraintTolerance',1e-15);

    
    fun = @(x) [-1 * (sample_mod - y_real .* (sigma^(-1) * (1 - x' * sample_x_clip ./ y_real).^(-1) - 1))];
    [alpha(:,k),fval{k},maxfval(k),exit(k),~] = fminimax(fun,x0,[],[],Aeq,beq,lb,ub,[],op);
    x_res{k} = fval{k} + sample_mod;
    y_res{k} = y_real;
    
end

save alpha.mat

%% plot
% order = 16;
% x = v_ext_output(:,:,order);
% x2 = alpha(:,order)' * x;
% 
% v_ext_comb = x_res{order};
% z = fval{order};
% y_real = y_res{order};
% % semilogx(v_pri,x2,'b','LineWidth',2)
% sample_mod = v_ext_comb - z;
% v_ext_table = (y_real.^-1 - sample_mod.^-1).^-1;
% % v_ext_comb = z + sample_mod;
% figure
% % subplot(2,1,1)
% % semilogx(v_ext_table,sample_mod,'r','LineWidth',2);  %%youwenti
% semilogx(b,a,'b','LineWidth',2);
% hold on
% % semilogx(v_pri,x2,'g','LineWidth',2)
% semilogx(y_real,v_ext_comb,'r','LineWidth',2)
% xlim([1e-6,1])
% ylim([0,16])
% title('Irregular, SNR = 12.5')
% legend('v_{\phi}','v_{\gamma}')
% xlabel('v_{z}')
% ylabel('v_{x}')
% set(gca,'FontName','Times','FontSize',25,'Xlim',[9.99e-6,1],'Ylim',[0,10],'LineWidth',2)
% ax = axes('Position',[0.16,0.6,0.2,0.2]);
% box on
% semilogx(ax,y_real(410:448),sample_mod(410:448),'b','LineWidth',1)
% hold on
% semilogx(ax,y_real(410:448),v_ext_comb(410:448),'r','LineWidth',1)
% xticks([])
% yticks([])
% % 
% % %% CR -13 graph
% 
% z = x_res{41}; % 2.0dB
% clip_sample = z(find(CR == -13),:);
% c= zeros(length(CR),1);
% c(find(CR == -13))=1;
% clip_sample = y_real .* (sigma^(-1) * (1 -  clip_sample ./ y_real).^(-1) - 1);
% ext_clip = (clip_sample.^-1 - y_real.^-1).^-1;
% subplot(2,1,2)
% semilogx(v_ext_table,sample_mod,'b','LineWidth',2);
% hold on
% semilogx(y_real,clip_sample,'r','LineWidth',2)
% xlabel('v_{z}','FontName','Times')
% ylabel('v_{x}')
% set(gca,'FontName','Times','FontSize',20,'Xlim',[9.99e-6,1],'Ylim',[0,10],'LineWidth',2)



