function [u_dem_ext,v_dem_ext_m]  = external_information(u_pri,v_pri,u_post,v_post,index)
if nargin == 5
    M = length(u_post);
    N = length(u_pri);  % N >= M
    len_sec = size(u_post,2);
    
    u_post_hat = zeros(N,len_sec);
    u_post_hat(index,:) = u_post;
    u_post_hat(setdiff([1:N],index),:) = u_pri(setdiff([1:N],index),:);

    v_post_hat = M / N * v_post + (N - M) / N * v_pri;
else
    u_post_hat = u_post;
    v_post_hat = v_post;
end
v_dem_ext_m = 1 ./ ( 1./v_post_hat - 1./v_pri );                             
u_dem_ext = v_dem_ext_m .* (u_post_hat./v_post_hat - u_pri./v_pri); 
end

