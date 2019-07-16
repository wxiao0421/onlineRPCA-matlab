clc; clear; close all;
addpath('..\omwRPCA');

% data preparation
video = [];
pic = 255*im2double(imread ('\Lobby\SwitchLight1344.bmp'));

wid = floor(0.75*size(pic,2));
hei = floor(0.5*size(pic,1));
virtual_camera_speed = 10; % the camera moves one pixel per x frames.
frame_cnt=0;
fprintf('\t\tchange point is 400 and 800\n');
frame_begin = 1250;
frame_len = 200;

for fold_iter = 1 : 5
    fold_dir = ['\Lobby\'];
    for file_iter = frame_begin: frame_begin+frame_len
        pic = 255*im2double(rgb2gray(imread ([fold_dir 'SwitchLight' num2str(file_iter,'%04d') '.bmp'])));
        virtual_camera_start = mod(floor(frame_cnt / virtual_camera_speed), floor(0.25*size(pic,2))) + 1;
        virtual_camera_end = virtual_camera_start + wid - 1;
        frame_cnt = frame_cnt + 1;
        temp = pic(0.5*hei:0.5*hei+hei-1, virtual_camera_start:virtual_camera_end);
        video = [video reshape(temp', [], 1)];
    end
end

% parameter setting
startTime = tic;
pms.burnin=100 ;
pms.win_size=30;
pms.track_cp_burnin=100;
pms.n_check_cp=20;
pms.alpha=0.01;
pms.proportion=0.2;
pms.n_positive=3;
pms.min_test_size=100;
pms.tolerance_num=0;
pms.lambda1=1.0/sqrt(200);
pms.lambda2=2.0/sqrt(200)*(10^2);
pms.factor=1;

[ Lhat_cp, Shat_cp, rank, cp, num_sparses ] = omwrpca_cp( video,pms);

% show the result 
% running time
fprintf('omwrpca_cp finished. total time is %f.\n', toc(startTime));

% the changing curve of the number of sparse points
figure;
plot(num_sparses);

% the original image, the foreground and the background 
figure; hold on
for iter=1:frame_cnt
    img=[];
    temp=reshape((video(:,iter)),[],hei);
    minT=min(video(:,iter));
    maxT=max(video(:,iter));
    temp=(temp-minT)*(1)/(maxT-minT);
    img=[img temp];
    
    temp=[reshape((Lhat_cp(:,iter)),[],hei)];
    minT=min(Lhat_cp(:,iter));
    maxT=max(Lhat_cp(:,iter));
    temp=(temp-minT)*(1)/(maxT-minT);
    img=[img temp];
    
    temp=[reshape((Shat_cp(:,iter)),[],hei)];
    minT=min(Shat_cp(:,iter));
    maxT=max(Shat_cp(:,iter));
    temp=(temp-minT)*(1)/(maxT-minT);
    img=[img temp];
    
    imshow(img',[]);
end