function [img_local,position] = mgrid(img,msk,stride,ps)

% clear
% stride = 32;
% ps = 48; %patch size
% load('temp.mat');

slicenum = 50;
img = img(:,:,end:-1:1);
msk = msk(:,:,end:-1:1);
hmsk = squeeze(sum(msk,[1,2]));
z0 = size(img,3);
idx = [1:z0;zeros(1,z0)];
idx = resizem(idx,[2,slicenum]);
sliceidx = idx(1,:);
% prevent missing sampling
for i = 1:slicenum-1
    if sliceidx(i) ~= sliceidx(i+1)+1
        for j = sliceidx(i):sliceidx(i+1)
            if hmsk(j)
                sliceidx(i) = j;
                continue
            end
        end
    end
end
imgn = []; mskn = [];
for i = 1:slicenum
    imgs = imresize(img(:,:,sliceidx(i)),[256,256]);
    msks = imresize(msk(:,:,sliceidx(i)),[256,256]);
    msks = im2bw(msks);
    imgn(:,:,i) = imgs;
    mskn(:,:,i) = msks;
    
%     imgs = mat2gray(imgs,[-1000,200]);
%     subplot 121; imshow(imgs);
%     subplot 122; imshow(msks);
end

[x,y,z] = size(imgn);

n = 1;
img_local = -2000*ones(1,ps,ps);
position = ones(1,3);
for k = 1:z
    imgs = imgn(:,:,k);
    imgs_local = padarray(imgs,[ps/2,ps/2],'symmetric');
    
%     imshow(imgs_global);
    msks = mskn(:,:,k);
    if sum(msks) == 0
        continue
    end
    
%     imgs = mat2gray(imgs,[-1400,200]);
%     imgs_lobal = mat2gray(imgs_lobal,[-1400,200]);
%     subplot(121); imshow(imgs);
%     subplot(122); imshow(msks);
    for i = 1:x/stride
        for j = 1:y/stride
            if sum(msks((i-1)*stride+1:i*stride,(j-1)*stride+1:j*stride),'all') > 0
                img_local(n,:,:) = imgs_local((i-1)*stride+1:(i-1)*stride+ps,...
                    (j-1)*stride+1:(j-1)*stride+ps);
                
                position(n,:) = [i,j,k];
                
%                 figure(); imshow(squeeze(img_local(n,:,:)));
                
                n = n+1;
            end
        end
    end
end
k;