clc
clear 
close all

category = 'Urban2';
% category = 'RubberWhale';
% Compute rmse from ground truth:
gtflow = readFlowFile( strcat('gt_data/',category,'_gt.flo'));
g_u = gtflow(:,:,1);
g_v = gtflow(:,:,2);

% fix unknown flow
UNKNOWN_FLOW_THRESH = 1e9;
idxUnknown = (abs(g_u)> UNKNOWN_FLOW_THRESH) | (abs(g_v)> UNKNOWN_FLOW_THRESH) ;
g_u(idxUnknown) = 0;
g_v(idxUnknown) = 0;

data_path = strcat('../8-image-data/',category,'/*.png');
files = dir(data_path);
data = imread(strcat(files(1).folder, "/", files(1).name));
img_volume = zeros(size(data,1), size(data,2), length(files), 'uint8');
img_volume(:,:,1) = data; 
for k = 2:length(files)
    filepath = strcat(files(k).folder, "/", files(k).name);
    img_volume(:,:,k) = imread(filepath);
end

% pick a method for gradient computation: 'sobel', 'prewitt', 'central', 'intermediate'
method = 'sobel';
fprintf("Using '%s' method for gradient computation\n\n", method);
[Ix,Iy,It] = imgradientxyz(img_volume, method);

% Normalize gradient output:
switch method
    case 'sobel'
        Ix = Ix * (1/44);
        Iy = Iy * (1/44);
        It = It * (1/44);
    case 'prewitt'
        Ix = Ix * (1/18);
        Iy = Iy * (1/18);
        It = It * (1/18);        
end

%% First method: Using same delta tau for entire volume 

% Find the maximum x and y gradients for each frame:
max_Ix          = squeeze(max(Ix,[],'all'));
max_Iy          = squeeze(max(Iy,[],'all'));

% Parameters:
lambda_s        = [100, 10, 5, 1, 0.5, 0.1, 0.01, 0.001];
lambda_t        = [100, 10, 5, 1, 0.5, 0.1, 0.01, 0.001];
% lambda_s        = [100];
% lambda_t        = [0.0001];
threshold       = 1e-3; % difference threshold for convergence

% Constants:
ones_matrix     = ones(size(Ix));
nr              = size(Ix,1); % number of rows
nc              = size(Ix,2); % number of columns
nf              = size(Ix,3); % number of frames
Ix_sq           = Ix.^2;
Iy_sq           = Iy.^2;

all_rmse_u      = zeros(length(lambda_s),length(lambda_t)); 
all_rmse_v      = zeros(length(lambda_s),length(lambda_t)); 

flog = fopen(strcat('log_',category,'.txt'), 'w'); 
for n_s = 1:length(lambda_s)
    for n_t = 1:length(lambda_t)
        
        % Initialize variables:
        u               = zeros(size(Ix));
        v               = zeros(size(Iy));
        diff_u          = Inf;
        diff_v          = Inf;

        % Step size based on CFL condition:
        dt_u            = 2 / (max_Ix^2 + 8 * lambda_s(n_s) + 4 * lambda_t(n_t));
        dt_v            = 2 / (max_Iy^2 + 8 * lambda_s(n_s) + 4 * lambda_t(n_t));

%         iter = 1;
        
        fprintf(flog, "Params used: n_s=%d/%d, n_t=%d/%d, \nlambda_s: %f\nlambda_t: %f\ndt_u: %e\ndt_v: %e\n\n", ...
                n_s, length(lambda_s), n_t, length(lambda_t),lambda_s(n_s), lambda_t(n_t), dt_u, dt_v);
        fprintf("Params used: n_s=%d/%d, n_t=%d/%d, \nlambda_s: %f\nlambda_t: %f\ndt_u: %e\ndt_v: %e\n\n", ...
                n_s, length(lambda_s), n_t, length(lambda_t),lambda_s(n_s), lambda_t(n_t), dt_u, dt_v);

            while( (diff_u > threshold) || (diff_v > threshold) )

                % Hold v constant and take a "u-step": 
                current_u   = u;
                u_xx        = [u(:,2:end,:), zeros(nr,1,nf)] + [zeros(nr,1,nf), u(:,1:end-1,:)]; % x-axis is along columns
                u_yy        = [u(2:end,:,:); zeros(1,nc,nf)] + [zeros(1,nc,nf); u(1:end-1,:,:)]; % y-axis is along rows
                u_tt        = cat(3,u(:,:,2:end),zeros(nr,nc,1)) + cat(3,zeros(nr,nc,1),u(:,:,1:end-1));
                u           = (ones_matrix - Ix_sq.*dt_u - 4.*lambda_s(n_s).*dt_u - 2.*lambda_t(n_t).*dt_u).* u ...
                              -(Iy .* Ix .* v + It .* Ix) .* dt_u ... 
                              + lambda_s(n_s) .* dt_u .* (u_xx + u_yy) + lambda_t(n_t) .* dt_u .* u_tt;
                diff_u      = max(abs(u - current_u),[],'all'); 

                % Hold u constant and take a "v-step": 
                current_v   = v;
                v_xx        = [v(:,2:end,:), zeros(nr,1,nf)] + [zeros(nr,1,nf), v(:,1:end-1,:)]; % x-axis is along columns
                v_yy        = [v(2:end,:,:); zeros(1,nc,nf)] + [zeros(1,nc,nf); v(1:end-1,:,:)]; % y-axis is along rows
                v_tt        = cat(3,v(:,:,2:end),zeros(nr,nc,1)) + cat(3,zeros(nr,nc,1),v(:,:,1:end-1));
                v           = (ones_matrix - Iy_sq.*dt_v -4.*lambda_s(n_s).*dt_v - 2.*lambda_t(n_t).*dt_v).* v ...
                              -(Iy .* Ix .* u + It .* Iy) .* dt_v ... 
                              + lambda_s(n_s) .* dt_v .* (v_xx + v_yy) + lambda_t(n_t) .* dt_v .* v_tt;
                diff_v      = max(abs(v - current_v),[],'all');     

%                 fprintf("Iteration: %d, u_diff = %f, v_diff = %f\n", iter, diff_u, diff_v);
%                 iter        = iter +1;  
                if ((diff_u > 1e3) || (diff_v > 1e3))
                    fprintf(flog, "*************Failed for lambda_s=%f, lambda_t=%f ***********\n\n", lambda_s(n_s), lambda_t(n_t));
                    fprintf("*************Failed for lambda_s=%f, lambda_t=%f ***********\n\n", lambda_s(n_s), lambda_t(n_t));
                    break;
                end
            end
            
        maxu = max(u, [], 'all');
        minu = min(u, [], 'all');

        maxv = max(v, [], 'all');
        minv = min(v, [], 'all');

        fprintf(flog, "Flow range for u: %f to %f\n",minu, maxu);
        fprintf(flog, "Flow range for v: %f to %f\n",minv, maxv);

        mag = sqrt(u.^2+v.^2);
        maxmag = max(mag, [], 'all');

        fprintf(flog, "Max flow magnitude: %f\n",maxmag);

        rmse_u = sqrt(mean((g_u - u(:,:,4)).^2,'all'));
        rmse_v = sqrt(mean((g_v - v(:,:,4)).^2,'all'));
        fprintf(flog, "RMSE_u: %f\nRMSE_v: %f\n\n", rmse_u, rmse_v);
        
        all_rmse_u(n_s, n_t) = rmse_u;
        all_rmse_v(n_s, n_t) = rmse_v;
           
    end
end
fclose(flog);
save('rmse_errors.mat','all_rmse_u','all_rmse_v')



















