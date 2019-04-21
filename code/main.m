clc
clear 
close all

category = 'Urban2/*.png';
data_path = strcat('../8-image-data/',category);
files = dir(data_path);
data = imread(strcat(files(1).folder, "/", files(1).name));
img_volume = zeros(size(data,1), size(data,2), length(files), 'uint8');
img_volume(:,:,1) = data; 
for k = 2:length(files)
    filepath = strcat(files(k).folder, "/", files(k).name);
    img_volume(:,:,k) = imread(filepath);
%     figure(1);
%     imshow(all_data(:,:,k))
end

% pick a method for gradient computation: 'sobel', 'prewitt', 'central', 'intermediate'
method = 'sobel';
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

% for k = 1:length(files)
%    figure(1);
%    imshow(Gx(:,:,k))
% end


%% First method: Using same delta tau for entire volume 

% Find the maximum x and y gradients for each frame:
max_Ix          = squeeze(max(Ix,[],'all'));
max_Iy          = squeeze(max(Ix,[],'all'));

% Parameters:
lambda          = 0.1;
threshold       = 1e-3; % difference threshold for convergence

% Step size based on CFL condition:
dt_u            = 2 ./ (max_Ix.^2 + 12 * lambda);
dt_v            = 2 ./ (max_Iy.^2 + 12 * lambda);

% Constants:
ones_matrix     = ones(size(Ix));
nr              = size(Ix,1); % number of rows
nc              = size(Ix,2); % number of columns
nf              = size(Ix,3); % number of frames
Ix_sq           = Ix.^2;
Iy_sq           = Iy.^2;

% Initialize variables:
u               = zeros(size(Ix));
v               = zeros(size(Iy));
diff_u          = Inf;
diff_v          = Inf;

iter = 1;
fprintf("Params used:\nlambda: %f\nThreshold: %f\nmax Ix: %f\ndelta tau for u: %f\n\n",lambda, threshold, max_Ix, dt_u);
while( (diff_u > threshold) || (diff_v > threshold) )
    
    % Hold v constant and take a "u-step": 
    current_u   = u;
    u_xx        = [u(:,2:end,:), zeros(nr,1,nf)] - [zeros(nr,1,nf), u(:,1:end-1,:)]; % x-axis is along columns
    u_yy        = [u(2:end,:,:); zeros(1,nc,nf)] - [zeros(1,nc,nf); u(1:end-1,:,:)]; % y-axis is along rows
    u_tt        = cat(3,u(:,:,2:end),zeros(nr,nc,1)) - cat(3,zeros(nr,nc,1),u(:,:,1:end-1));
    u           = (ones_matrix - Ix_sq.*dt_u - 6.*lambda.*dt_u).* u ...
                  -(Iy .* Ix .* v + It .* Ix) .* dt_u ... 
                  + lambda .* dt_u .* (u_xx + u_yy + u_tt);
    diff_u      = max(abs(u - current_u),[],'all'); 

    % Hold u constant and take a "v-step": 
    current_v   = v;
    v_xx        = [v(:,2:end,:), zeros(nr,1,nf)] - [zeros(nr,1,nf), v(:,1:end-1,:)]; % x-axis is along columns
    v_yy        = [v(2:end,:,:); zeros(1,nc,nf)] - [zeros(1,nc,nf); v(1:end-1,:,:)]; % y-axis is along rows
    v_tt        = cat(3,v(:,:,2:end),zeros(nr,nc,1)) - cat(3,zeros(nr,nc,1),v(:,:,1:end-1));
    v           = (ones_matrix - Iy_sq.*dt_v - 6.*lambda.*dt_v).* v ...
                  -(Iy .* Ix .* u + It .* Iy) .* dt_v ... 
                  + lambda .* dt_v .* (v_xx + v_yy + v_tt);
    diff_v      = max(abs(v - current_v),[],'all');     
    
    fprintf("Iteration: %d, u_diff = %f, v_diff = %f\n", iter, diff_u, diff_v);
    iter        = iter +1;    
end



























