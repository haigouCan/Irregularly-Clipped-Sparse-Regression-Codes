function [ app_mean, app_var ] = Z_APP_Clip( clip, y_in, pri_mean, pri_var, sigma_in )
%% To calculate the posterior mean and variance
% y_in = Q(z) + n
% Q(z) = z if |z| < clip,
% Q(z) = sign(z)clip, otherwise.
% z          N(pri_mean, pri_var)
% n          N(0, sigma_in)
% app_mean   estimate of z
% app_var    variance of estimate app_mean
%
%                        --by Shansuo Liang 2019
% 
%%
u_star = (y_in .* pri_var + pri_mean .* sigma_in) ./ (pri_var+sigma_in); %M,1
sigma_star = (pri_var .* sigma_in) ./ (pri_var+sigma_in);
% sigma_star(sigma_star < 1e-7) = 1e-7;
% c_0 = (u_star.^2 - ((y_in.^2) .* max(pri_var,1e-4) + pri_mean.^2 .* sigma_in)./(max(pri_var,1e-4)+sigma_in)) ./ (2 .* sigma_star);
% % c_0(c_0 >= 1e1) = 1e1;
% c_1 = exp(c_0);
% 
% c_2 = sqrt(sigma_star./(2.*pi.*sigma_in.*pri_var)) ;
% c_star_2 = c_1 .* c_2;
c_star = exp((u_star.^2 - ((y_in.^2) .* pri_var + pri_mean.^2 .* sigma_in)./(pri_var+sigma_in)) ./ (2 .* sigma_star)) .*sqrt(sigma_star./(2.*pi.*sigma_in.*pri_var));%M,1

%%  A = 1 
alpha_y = (-y_in - clip)./sqrt(2.*sigma_in); %M,1
beta_y = (clip-y_in)./sqrt(2.*sigma_in); %M,1

alpha_star = (-u_star - clip)./sqrt(2.*sigma_star); %M,1
beta_star = (clip-u_star)./sqrt(2.*sigma_star); %M,1

alpha_x = (-pri_mean - clip)./sqrt(2.*pri_var); %M,1
beta_x = (clip-pri_mean)./sqrt(2.*pri_var); %M,1

%%
p_y_gx = (exp(-alpha_y.^2).*(2-erfc(alpha_x))+exp(-beta_y.^2).*erfc(beta_x))./sqrt(8.*pi.*sigma_in) ...,
    + (c_star.*(erfc(alpha_star)-erfc(beta_star)))./2; %M,1
p_y_gx = max(p_y_gx,1e-12); %M,1


integral_x = exp(-alpha_y.^2).*(sqrt(pi/2).*pri_mean.*(2-erfc(alpha_x))-exp(-alpha_x.^2).*sqrt(pri_var))./(2*pi*sqrt(sigma_in))...,
    + exp(-beta_y.^2).*(sqrt(pi/2).*pri_mean.*erfc(beta_x) + exp(-beta_x.^2).*sqrt(pri_var))./(2*pi*sqrt(sigma_in))...,
    + c_star.*(sqrt(sigma_star).*(exp(-alpha_star.^2)-exp(-beta_star.^2)) + sqrt(pi/2).*u_star.*(erfc(alpha_star)-erfc(beta_star)))./sqrt(2*pi);


integral_x2 = exp(-alpha_y.^2).*(sqrt(pi)./2.*(pri_mean.^2+pri_var).*(2-erfc(alpha_x)) + beta_x.*exp(-alpha_x.^2).*pri_var)./(pi*sqrt(2*sigma_in))...,
     + sigma_star.*c_star.*(alpha_star.*exp(-beta_star.^2) - beta_star.*exp(-alpha_star.^2))./sqrt(pi)...,
     + 0.5*c_star.*(u_star.^2+sigma_star).*(erfc(alpha_star)-erfc(beta_star)) ...,
     + exp(-beta_y.^2).*(sqrt(pi)./2.*(pri_mean.^2+pri_var).*erfc(beta_x) - alpha_x.*exp(-beta_x.^2).*pri_var)./(pi*sqrt(2*sigma_in)); %M,1

 
%  %%
%  p_y_gx_2 = (exp(-alpha_y.^2).*(2-erfc(alpha_x))+exp(-beta_y.^2).*erfc(beta_x))./sqrt(8.*pi.*sigma_in) ...,
%     + (c_star_2.*(erfc(alpha_star)-erfc(beta_star)))./2; %M,1
% p_y_gx_2 = max(p_y_gx_2,1e-12); %M,1
% 
% 
% integral_x_2 = exp(-alpha_y.^2).*(sqrt(pi/2).*pri_mean.*(2-erfc(alpha_x))-exp(-alpha_x.^2).*sqrt(pri_var))./(2*pi*sqrt(sigma_in))...,
%     + exp(-beta_y.^2).*(sqrt(pi/2).*pri_mean.*erfc(beta_x) + exp(-beta_x.^2).*sqrt(pri_var))./(2*pi*sqrt(sigma_in))...,
%     + c_star_2.*(sqrt(sigma_star).*(exp(-alpha_star.^2)-exp(-beta_star.^2)) + sqrt(pi/2).*u_star.*(erfc(alpha_star)-erfc(beta_star)))./sqrt(2*pi);
% 
% 
% integral_x2_2 = exp(-alpha_y.^2).*(sqrt(pi)./2.*(pri_mean.^2+pri_var).*(2-erfc(alpha_x)) + beta_x.*exp(-alpha_x.^2).*pri_var)./(pi*sqrt(2*sigma_in))...,
%      + sigma_star.*c_star_2.*(alpha_star.*exp(-beta_star.^2) - beta_star.*exp(-alpha_star.^2))./sqrt(pi)...,
%      + 0.5*c_star_2.*(u_star.^2+sigma_star).*(erfc(alpha_star)-erfc(beta_star)) ...,
%      + exp(-beta_y.^2).*(sqrt(pi)./2.*(pri_mean.^2+pri_var).*erfc(beta_x) - alpha_x.*exp(-beta_x.^2).*pri_var)./(pi*sqrt(2*sigma_in)); %M,1
 
%%
app_mean = integral_x./p_y_gx;

app_var = mean(integral_x2./p_y_gx - app_mean.^2);

% %%
% app_mean_2 = integral_x_2./p_y_gx_2;
% 
% app_var_2 = mean(integral_x2_2./p_y_gx_2 - app_mean_2.^2);

end

