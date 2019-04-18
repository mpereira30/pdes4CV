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

[Gx,Gy,Gz] = imgradientxyz(img_volume,'central');
% for k = 1:length(files)
%    figure(1);
%    imshow(Gz(:,:,k))
% end

max_Ix = max(max(max(Gx)))
max_Iy = max(max(max(Gy)))
max_It = max(max(max(Gz)))