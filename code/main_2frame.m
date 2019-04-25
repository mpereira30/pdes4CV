clc
clear 
close all

category = 'Urban2';
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
%     figure(1);
%     imshow(all_data(:,:,k))
end

img_volume = img_volume(:,:,4:5);  

% pick a method for gradient computation: 'sobel', 'prewitt', 'central', 'intermediate'
method = 'central';
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

% for k = 1:length(files)
%    figure(1);
%    imshow(Gx(:,:,k))
% end


%% First method: Using same delta tau for entire volume 

% Find the maximum x and y gradients for each frame:
max_Ix          = squeeze(max(Ix,[],'all'));
max_Iy          = squeeze(max(Iy,[],'all'));

% Parameters:
lambda          = 1.0;
threshold       = 1e-3; % difference threshold for convergence

% Step size based on CFL condition:
dt_u            = 2 / (max_Ix^2 + 8 * lambda);
dt_v            = 2 / (max_Iy^2 + 8 * lambda);

% Constants:
ones_matrix     = ones(size(Ix,1), size(Ix,2));
nr              = size(Ix,1); % number of rows
nc              = size(Ix,2); % number of columns
nf              = size(Ix,3); % number of frames

Ix              = Ix(:,:,1);
Iy              = Iy(:,:,1);
It              = It(:,:,1);
Ix_sq           = Ix.^2;
Iy_sq           = Iy.^2;

% Initialize variables:
u               = zeros(size(Ix,1), size(Ix,2));
v               = zeros(size(Iy,1), size(Iy,2));
diff_u          = Inf;
diff_v          = Inf;

iter = 1;
fprintf("Params used:\nlambda: %e\nThreshold: %e\nmax Ix: %f\nmax Iy: %f\ndt_u: %e\ndt_v: %e\n\n", ...
        lambda, threshold, max_Ix, max_Iy, dt_u, dt_v);
while( (diff_u > threshold) || (diff_v > threshold) )
    
    % Hold v constant and take a "u-step": 
    current_u   = u;
    u_xx        = [u(:,2:end), zeros(nr,1)] + [zeros(nr,1), u(:,1:end-1)]; % x-axis is along columns
    u_yy        = [u(2:end,:); zeros(1,nc)] + [zeros(1,nc); u(1:end-1,:)]; % y-axis is along rows
    u           = (ones_matrix - Ix_sq.*dt_u - 4.*lambda.*dt_u).* u ...
                  -(Iy .* Ix .* v + It .* Ix) .* dt_u ... 
                  + lambda .* dt_u .* (u_xx + u_yy);
    diff_u      = max(abs(u - current_u),[],'all'); 

    % Hold u constant and take a "v-step": 
    current_v   = v;
    v_xx        = [v(:,2:end), zeros(nr,1)] + [zeros(nr,1), v(:,1:end-1)]; % x-axis is along columns
    v_yy        = [v(2:end,:); zeros(1,nc)] + [zeros(1,nc); v(1:end-1,:)]; % y-axis is along rows
    v           = (ones_matrix - Iy_sq.*dt_v - 4.*lambda.*dt_v).* v ...
                  -(Iy .* Ix .* u + It .* Iy) .* dt_v ... 
                  + lambda .* dt_v .* (v_xx + v_yy);
    diff_v      = max(abs(v - current_v),[],'all');     
    
    fprintf("Iteration: %d, u_diff = %f, v_diff = %f\n", iter, diff_u, diff_v);
    iter        = iter +1;    
end


maxu = max(u, [], 'all');
minu = min(u, [], 'all');

maxv = max(v, [], 'all');
minv = min(v, [], 'all');

fprintf("\nFlow range for u: %f to %f\n",minu, maxu);
fprintf("Flow range for v: %f to %f\n",minv, maxv);

mag = sqrt(u.^2+v.^2);
maxmag = max(mag, [], 'all');

fprintf("Max flow magnitude: %f\n",maxmag);

rmse_u = sqrt(mean((g_u - u).^2,'all'))
rmse_v = sqrt(mean((g_v - v).^2,'all'))


















