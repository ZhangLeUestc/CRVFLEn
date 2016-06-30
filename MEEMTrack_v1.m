%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Implemetation of the tracker described in paper
%	"MEEM: Robust Tracking via Multiple Experts using Entropy Minimization", 
%   Jianming Zhang, Shugao Ma, Stan Sclaroff, ECCV, 2014
%	
%	Copyright (C) 2014 Jianming Zhang
%
%	This program is free software: you can redistribute it and/or modify
%	it under the terms of the GNU General Public License as published by
%	the Free Software Foundation, either version 3 of the License, or
%	(at your option) any later version.
%
%	This program is distributed in the hope that it will be useful,
%	but WITHOUT ANY WARRANTY; without even the implied warranty of
%	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%	GNU General Public License for more details.
%
%	You should have received a copy of the GNU General Public License
%	along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%	If you have problems about this software, please contact: jmzhang@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = MEEMTrack_v1(input, ext, show_img, init_rect, start_frame, end_frame)

addpath(genpath('.'));

% parse input arguments
D = dir(fullfile(input,['*.', ext]));
file_list={D.name};

if nargin < 4
    init_rect = -ones(1,4);
end
if nargin < 5
    start_frame = 1;
end
if nargin < 6
    end_frame = numel(file_list);
end

% declare global variables
global sampler
global CNN_tracker
global experts
global config

global finish % flag for determination by keystroke


config.display = true;
sampler = createSampler();

experts = {};
finish = 0;


timer = 0;
result.res = nan(end_frame-start_frame+1,4);
result.len = end_frame-start_frame+1;
result.startFrame = start_frame;
result.type = 'rect';

if show_img
    figure(1); set(1,'KeyPressFcn', @handleKey); 
end
% pos_update=[];
% cost_pos=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output = zeros(1,4);

for frame_id = start_frame:end_frame
    if finish == 1
        break;
    end

    if ~config.display
        clc
        display(input);
        display(['frame: ',num2str(frame_id),'/',num2str(end_frame)]);
    end
    
    %% read a frame
    I_orig=imread(fullfile(input,file_list{frame_id}));
   
   
%     if numel(S)<3
%       I_orig1=zeros([S,1]);
%       
%       I_orig1(:,:,1)=I_orig;
%       I_orig=I_orig1;
%     end
%     
    %% intialization
    if frame_id==start_frame
        
        % crop to get the initial window
        if isequal(init_rect,-ones(1,4))
            assert(config.display)
            figure(1)
            imshow(I_orig);
            [InitPatch init_rect]=imcrop(I_orig);
        end
        init_rect = round(init_rect);
        
        config = makeConfig(I_orig,init_rect,true,true,true,show_img);
        CNN_tracker.output = init_rect*config.image_scale;
        CNN_tracker.output(1:2) = CNN_tracker.output(1:2) + config.padding;
        CNN_tracker.output_exp = CNN_tracker.output;
        
        output = CNN_tracker.output;
    end
        
    %% compute ROI and scale image
    [I_scale]= getFrame2Compute(I_orig);
    
    %% crop frame
    if frame_id == start_frame
        sampler.roi = rsz_rt(CNN_tracker.output,size(I_scale),5*config.search_roi,false);
    else%if CNN_tracker.confidence > config.svm_thresh
        sampler.roi = rsz_rt(output,size(I_scale),config.search_roi,true);
    end
    I_crop = I_scale(round(sampler.roi(2):sampler.roi(4)),round(sampler.roi(1):sampler.roi(3)),:);
    %I_crop=I_crop-median(I_crop(:));
    %I_crop=I_crop/255;
    %% compute feature images
   % [BC F] = getFeatureRep(I_crop,config.hist_nbin);
   
    %% tracking part
    
    tic
    
    if frame_id==start_frame
        initSampler(CNN_tracker.output,I_crop,[],config.use_color);
        train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
        label = sampler.costs(train_mask,1)<config.thresh_p;
        fuzzy_weight = ones(size(label));
        createCNNTracker
          
        %initSvmTracker(sampler.patterns_dt(train_mask,:), label, fuzzy_weight);
        
       
        initCNNTracker(sampler.patterns_dt(:,:,:,train_mask), label, fuzzy_weight);
     
      
       % size(sampler.patterns_dt(train_mask,:))
        if config.display
            figure(1);
            imshow(I_orig);
            res = CNN_tracker.output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            rectangle('position',res,'LineWidth',2,'EdgeColor','b')
        end
    else
        % testing
        if config.display
            figure(1)
            imshow(I_orig);       
            roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2)+1;
            roi_reg(1:2) = roi_reg(1:2) - config.padding;
            rectangle('position',roi_reg/config.image_scale,'LineWidth',1,'EdgeColor','r');
        end
        if mod((frame_id - start_frame + 1),config.expert_update_interval) == 0% CNN_tracker.update_count >= config.update_count_thresh
            updateTrackerExperts;
        end

        expertsDo(I_crop,config.expert_lambda,config.label_prior_sigma);
      %  CNN_tracker.confidence 
%         if CNN_tracker.confidence > config.RF_thresh
%             output = CNN_tracker.output;
%         end
        output = CNN_tracker.output;
        
        if config.display
            figure(1) 
            res = output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            if CNN_tracker.best_expert_idx ~= numel(experts)
                % red rectangle: the prediction of current tracker
                res_prev = CNN_tracker.output_exp;
                res_prev(1:2) = res_prev(1:2) - config.padding;
                res_prev = res_prev/config.image_scale;
                rectangle('position',res_prev,'LineWidth',2,'EdgeColor','r') %
                % yellow rectangle: the prediction of the restored tracker
                rectangle('position',res,'LineWidth',2,'EdgeColor','y')   
            else
                % blue rectangle: indicates no restoration happens 
                rectangle('position',res,'LineWidth',2,'EdgeColor','b') 
            end
        end
        
 CNN_tracker.update_count=0;
 % if mod( frame_id,3)==0    
        
    %    if CNN_tracker.confidence > config.CNN_thresh || CNN_tracker.update_count>30%&& ~CNN_tracker.failure config.RF_thresh
    %if CNN_tracker.confidence <0.9 || CNN_tracker.update_count>15
     if CNN_tracker.confidence <0.9
           % disp('updating')
            train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
           label = sampler.costs(train_mask) < config.thresh_p;
%             pos_mask=(sampler.costs<config.thresh_p);
%             neg_mask=(sampler.costs>=config.thresh_n);
%             pos_update=cat(4,pos_update,sampler.patterns_dt(:,:,:,pos_mask));
%             
%             cost_pos=[cost_pos;sampler.costs(pos_mask)];
            skip_train = false;
%             if CNN_tracker.confidence > 1.0 
%                 score_ = -(sampler.patterns_dt(train_mask,:)*CNN_tracker.w'+CNN_tracker.Bias);
%                 if prod(double(score_(label) > 1)) == 1 && prod(double(score_(~label)<1)) == 1
%                     skip_train = true;
%                 end
%             end
        
            if ~skip_train
              
                costs = sampler.costs(train_mask);
                fuzzy_weight = ones(size(label));
                fuzzy_weight(~label) = 2*costs(~label)-1;
                updateCNNTracker (sampler.patterns_dt(:,:,:,train_mask),label,fuzzy_weight);  
                 
%                  cost=[ cost_pos;sampler.costs(neg_mask)];
%                  data_update=cat(4,pos_update,sampler.patterns_dt(:,:,:,neg_mask));
%                  nSam=numel(cost);
%                  nPos=numel(cost_pos);
%                  label=zeros(nSam,1);
%                  label(1:nPos)=1;
%                  
%                  fuzzy_weight = ones(size(label));
%                  fuzzy_weight(~label) = 2*cost(~label)-1;
%                  updateCNNTracker (data_update,label,fuzzy_weight);  
                CNN_tracker.update_count=0;
%                 pos_update=[];
%                 cost_pos=[];
                
            end
        else % clear update_count
            CNN_tracker.update_count = CNN_tracker.update_count+1;
        end
   % end
% toc
    end

    timer = timer + toc;
    res = output;
    res(1:2) = res(1:2) - config.padding;
    result.res(frame_id-start_frame+1,:) = res/config.image_scale;
end

%% output restuls
result.fps = result.len/timer;


clearvars -global sampler CNN_tracker experts config finish 
