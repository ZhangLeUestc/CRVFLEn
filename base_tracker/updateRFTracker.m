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

function updateRFTracker(sample,label,fuzzy_weight)
global RF_tracker
global experts;
global mtry
global ntree

pos_mask = label>0.5;
pos=sample(pos_mask,:);
pos_label=label(pos_mask);
nSample=numel(label);
N_temp=round(nSample/sum(pos_mask));
pos=repmat(pos,round(2*N_temp),1);
pos=pos+0.005*randn(size(pos));
pos_label=repmat(pos_label,round(2*N_temp),1);
sample=[sample;pos];

label=[label;pos_label];
model=RF_tracker.clsf;
nSample=size(sample,1);
[y1,y2]=ClusteringRF_predict(sample,model);
y2=reshape(y2,ntree,nSample);
y2=2*y2-3;
err=zeros(ntree,1);
% idx_temp=find(label>0.5);
% y2=y2(:,idx_temp);
% label=label(idx_temp);


for i=1:ntree
  err(i)=length(find(y2(i,:)==label'))/nSample;  
end
% err
% numel(idx_temp)
[val,idx_temp]=sort(err,'descend');
idx_temp=idx_temp(1:round(0.3*ntree));
% idx_temp=find(err~=0);

% idx_temp
%    idx_temp=find(err>0.5);
% sample = [RF_tracker.pos_sv;RF_tracker.neg_sv; sample];
% label = [ones(size(RF_tracker.pos_sv,1),1);zeros(size(RF_tracker.neg_sv,1),1);label];% positive:1 negative:0
% sample_w = [RF_tracker.pos_w;RF_tracker.neg_w;fuzzy_weight];
       
% pos_mask = label>0.5;
% neg_mask = ~pos_mask;
% s1 = sum(sample_w(pos_mask));
% s2 = sum(sample_w(neg_mask));
%         
% sample_w(pos_mask) = sample_w(pos_mask)*s2;
% sample_w(neg_mask) = sample_w(neg_mask)*s1;
        
%C = max(RF_tracker.C*sample_w/sum(sample_w),0.001);
%mtry=round(sqrt(size(sample,2)));

if numel(idx_temp)>0
model1=ClusteringRF_train(sample,label,'nvartosample',mtry,'ntrees',numel(idx_temp),'replace',0);
RF_tracker.clsf(idx_temp)=model1;
end


% if config.verbose
%    % RF_tracker.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{RF_tracker.struct_mat},...
%        %'boxconstraint',C,'autoscale','false','options',statset('Display','final','MaxIter',5000));
%    % fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(RF_tracker.clsf.Alpha,1)); 
%    RF_tracker.clsf=OnlineRF_train(model,sample,label,'nvartosample',mtry,'ntrees',ntree);
%   
%  % RF_tracker.clsf=BatchRF_train(sample,label,'nvatrosample',mtry,'ntrees',ntree);
% else
%     %.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{RF_tracker.struct_mat},...
%        %'boxconstraint',C,'autoscale','false','options',statset('MaxIter',5000));
% %  RF_tracker.clsf=BatchRF_train(sample,label,'nvatrosample',mtry,'ntrees',ntree);
%   RF_tracker.clsf=OnlineRF_train(model,sample,label,'nvartosample',mtry,'ntrees',ntree);
% end
% Y=OnlineRF_predict(sample,RF_tracker.clsf);

%**************************
% RF_tracker.w = RF_tracker.clsf.Alpha'*RF_tracker.clsf.SupportVectors;
% RF_tracker.Bias = RF_tracker.clsf.Bias;
% RF_tracker.clsf.w = RF_tracker.w;
% % get the idx of new svs
% sv_idx = RF_tracker.clsf.SupportVectorIndices;
% sv_old_sz = size(RF_tracker.pos_sv,1)+size(RF_tracker.neg_sv,1);
% sv_new_idx = sv_idx(sv_idx>sv_old_sz);
% sv_new = sample(sv_new_idx,:);
% sv_new_label = label(sv_new_idx,:);
%         
% num_sv_pos_new = sum(sv_new_label);
%         
% update pos_dis, pos_w and pos_sv
% pos_sv_new = sv_new(sv_new_label>0.5,:);
% if ~isempty(pos_sv_new)
%     if size(pos_sv_new,1)>1
%         pos_dis_new = squareform(pdist(pos_sv_new));
%     else
%         pos_dis_new = 0;
%     end
%     pos_dis_cro = pdist2(RF_tracker.pos_sv,pos_sv_new);
%     RF_tracker.pos_dis = [RF_tracker.pos_dis, pos_dis_cro; pos_dis_cro', pos_dis_new];
%     RF_tracker.pos_sv = [RF_tracker.pos_sv;pos_sv_new];
%     RF_tracker.pos_w = [RF_tracker.pos_w;ones(num_sv_pos_new,1)];
% end
        
% update neg_dis, neg_w and neg_sv
% neg_sv_new = sv_new(sv_new_label<0.5,:);
% if ~isempty(neg_sv_new)
%     if size(neg_sv_new,1)>1
%         neg_dis_new = squareform(pdist(neg_sv_new));
%     else
%         neg_dis_new = 0;
%     end
%     neg_dis_cro = pdist2(RF_tracker.neg_sv,neg_sv_new);
%     RF_tracker.neg_dis = [RF_tracker.neg_dis, neg_dis_cro; neg_dis_cro', neg_dis_new];
%     RF_tracker.neg_sv = [RF_tracker.neg_sv;neg_sv_new];
%     RF_tracker.neg_w = [RF_tracker.neg_w;ones(size(sv_new,1)-num_sv_pos_new,1)];
% end
        
% RF_tracker.pos_dis = RF_tracker.pos_dis + diag(inf*ones(size(RF_tracker.pos_dis,1),1));
% RF_tracker.neg_dis = RF_tracker.neg_dis + diag(inf*ones(size(RF_tracker.neg_dis,1),1));
%         
%         
% compute real margin
% pos2plane = -RF_tracker.pos_sv*RF_tracker.w';
% neg2plane = -RF_tracker.neg_sv*RF_tracker.w';
% RF_tracker.margin = (min(pos2plane) - max(neg2plane))/norm(RF_tracker.w);
        
% shrink svs
% check if to remove
% if size(RF_tracker.pos_sv,1)+size(RF_tracker.neg_sv,1)>RF_tracker.B
%     pos_score_sv = -(RF_tracker.pos_sv*RF_tracker.w'+RF_tracker.Bias);
%     neg_score_sv = -(RF_tracker.neg_sv*RF_tracker.w'+RF_tracker.Bias);
%     m_pos = abs(pos_score_sv) < RF_tracker.m2;
%     m_neg = abs(neg_score_sv) < RF_tracker.m2;
%             
%     if config.verbose
%         fprintf('remove svs: pos %d, neg %d \n',sum(~m_pos),sum(~m_neg));
%     end
%     if sum(m_pos) > 0
%         RF_tracker.pos_sv = RF_tracker.pos_sv(m_pos,:);
%         RF_tracker.pos_w = RF_tracker.pos_w(m_pos,:);
%         RF_tracker.pos_dis = RF_tracker.pos_dis(m_pos,m_pos);
%     end
% 
%     if sum(m_neg)>0
%         RF_tracker.neg_sv = RF_tracker.neg_sv(m_neg,:);
%         RF_tracker.neg_w = RF_tracker.neg_w(m_neg,:);
%         RF_tracker.neg_dis = RF_tracker.neg_dis(m_neg,m_neg);
%     end
% end
        
% check if to merge
% while size(RF_tracker.pos_sv,1)+size(RF_tracker.neg_sv,1)>RF_tracker.B
%     [mm_pos,idx_pos] = min(RF_tracker.pos_dis(:));
%     [mm_neg,idx_neg] = min(RF_tracker.neg_dis(:));
%             
%     if mm_pos > mm_neg || size(RF_tracker.pos_sv,1) <= RF_tracker.B_p% merge negative samples
%         if config.verbose
%             fprintf('merge negative samples: %d \n', size(RF_tracker.neg_w,1))
%         end
%                 
%         [i,j] = ind2sub(size(RF_tracker.neg_dis),idx_neg);
%         w_i= RF_tracker.neg_w(i);
%         w_j= RF_tracker.neg_w(j);
%         merge_sample = (w_i*RF_tracker.neg_sv(i,:)+w_j*RF_tracker.neg_sv(j,:))/(w_i+w_j);                
%                 
%         RF_tracker.neg_sv([i,j],:) = []; RF_tracker.neg_sv(end+1,:) = merge_sample;
%         RF_tracker.neg_w([i,j]) = []; RF_tracker.neg_w(end+1,1) = w_i + w_j;
%                 
%         RF_tracker.neg_dis([i,j],:)=[]; RF_tracker.neg_dis(:,[i,j])=[];
%         neg_dis_cro = pdist2(RF_tracker.neg_sv(1:end-1,:),merge_sample);
%         RF_tracker.neg_dis = [RF_tracker.neg_dis, neg_dis_cro;neg_dis_cro',inf];                
%     else
%         if config.verbose
%             fprintf('merge positive samples: %d \n', size(RF_tracker.pos_w,1))
%         end
% 
%         [i,j] = ind2sub(size(RF_tracker.pos_dis),idx_pos);
%         w_i= RF_tracker.pos_w(i);
%         w_j= RF_tracker.pos_w(j);
%         merge_sample = (w_i*RF_tracker.pos_sv(i,:)+w_j*RF_tracker.pos_sv(j,:))/(w_i+w_j);                
% 
%         RF_tracker.pos_sv([i,j],:) = []; RF_tracker.pos_sv(end+1,:) = merge_sample;
%         RF_tracker.pos_w([i,j]) = []; RF_tracker.pos_w(end+1,1) = w_i + w_j;
%                 
%         RF_tracker.pos_dis([i,j],:)=[]; RF_tracker.pos_dis(:,[i,j])=[];
%         pos_dis_cro = pdist2(RF_tracker.pos_sv(1:end-1,:),merge_sample);
%         RF_tracker.pos_dis = [RF_tracker.pos_dis, pos_dis_cro;pos_dis_cro',inf]; 
%                 
%                 
%     end
%             
% end
%         
% update experts
experts{end}.snapshot = RF_tracker;
% experts{end} = RF_tracker.Bias;
        
RF_tracker.update_count = RF_tracker.update_count + 1;
