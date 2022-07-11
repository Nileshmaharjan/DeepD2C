clc
close all
clear all

num_img = 20;
PSNR_avg = 0 ;
PSNR_vec = zeros(1, num_img) ;
for kk=1:num_img
    file1=sprintf('%d.jpg',kk);
    fullFileName1 = fullfile('C:\Research\COdes\DeepD2C\test_images', file1)
    I = imread(fullFileName1);

    file2=sprintf('%d_encoded_.png',kk);
    fullFileName2 = fullfile('C:\Research\COdes\DeepD2C\21DEC04\data_embedded_images', file2)
%    fullFileName2 = fullfile('C:\Research\COdes\DeepD2C\test_final', file2)
    Ihat = imread(fullFileName2);

    % Read the dimensions of the image.
    [rows columns ~] = size(I);

    % Calculate mean square error of R, G, B.   
    mseRImage = (double(I(:,:,1)) - double(Ihat(:,:,1))) .^ 2;
    mseGImage = (double(I(:,:,2)) - double(Ihat(:,:,2))) .^ 2;
    mseBImage = (double(I(:,:,3)) - double(Ihat(:,:,3))) .^ 2;

    mseR = sum(sum(mseRImage)) / (rows * columns);
    mseG = sum(sum(mseGImage)) / (rows * columns);
    mseB = sum(sum(mseBImage)) / (rows * columns);

    % Average mean square error of R, G, B.
    mse = (mseR + mseG + mseB)/3;

    % Calculate PSNR (Peak Signal to noise ratio).
    PSNR_vec(kk) = 10 * log10( 256^2 / mse);
    PSNR_avg = PSNR_avg + PSNR_vec(kk);
end
PSNR_avg = PSNR_avg/num_img

