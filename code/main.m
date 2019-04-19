clc
clear 
close all

cat = 'Urban2/*.png';
data_path = strcat('../8-image-data/',cat);
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
method = 'prewitt';
[Gx,Gy,Gz] = imgradientxyz(img_volume, method);

% Normalize gradient output:
switch method
    case 'sobel'
        Gx = Gx * (1/44);
        Gy = Gy * (1/44);
        Gz = Gz * (1/44);
    case 'prewitt'
        Gx = Gx * (1/18);
        Gy = Gy * (1/18);
        Gz = Gz * (1/18);        
end

% for k = 1:length(files)
%    figure(1);
%    imshow(Gx(:,:,k))
% end

max_Ix = squeeze(max(Gx,[],[1 2]))
max_Iy = squeeze(max(Gx,[],[1 2]))

lambda = 0.01;
% delta_tau = 2 / (max_Ix^2 + 12 * lambda)


