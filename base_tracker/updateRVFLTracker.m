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

function updateRVFLTracker(sample,label,sample_w)
global config;
global RVFL_tracker;
global experts;
Block=size(sample,1);
% positive:1 negative:0
 
% positive:1 negative:0      
pos_mask = label>0.5;
neg_mask = ~pos_mask;
 s1 = sum(sample_w(pos_mask));
 s2 = sum(sample_w(neg_mask));
      
%  sample_w(pos_mask) = sample_w(pos_mask)*s2;
%  sample_w(neg_mask) = sample_w(neg_mask)*s1;
% W=eye(numel(label));
% for i=1:numel(pos_mask)
% 
% W(pos_mask(i),pos_mask(i))=sample_w(pos_mask(i));
% end
% for i=1:numel(neg_mask)
% W(neg_mask(i),neg_mask(i))=sample_w(neg_mask(i));
% end
 N=round(1+s2/s1);  
 pos=sample(pos_mask,:);
 pos_label=label(pos_mask);
 
 pos=repmat(pos,N,1);
 pos=pos+0.05*randn(size(pos));
 pos_label=repmat(pos_label,N,1);
 sample=[sample;pos];
%  sample=sample-repmat(mean(sample,1),size(sample,1),1);
%  Norm_s=sqrt(sum(sample.^2,1));
%  sample=sample./repmat(Norm_s,size(sample,1),1);
%  sample(isnan(sample))=0;
label=[label;pos_label]; 
% sample = [RVFL_tracker.pos_sv;RVFL_tracker.neg_sv; sample];
% label = [ones(size(RVFL_tracker.pos_sv,1),1);zeros(size(RVFL_tracker.neg_sv,1),1);label];
%C = max(RVFL_tracker.C*sample_w/sum(sample_w),0.001);

% if config.verbose
% %     RVFL_tracker.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{RVFL_tracker.struct_mat},...
% %        'boxconstraint',C,'autoscale','false','options',statset('Display','final','MaxIter',5000));
% %     fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(RVFL_tracker.clsf.Alpha,1)); 
% else
% %     RVFL_tracker.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{RVFL_tracker.struct_mat},...
% %        'boxconstraint',C,'autoscale','false','options',statset('MaxIter',5000));
% end
M=RVFL_tracker.clsf.M;
beta=RVFL_tracker.clsf.beta;
weight=RVFL_tracker.clsf.weight;
bias=RVFL_tracker.clsf.bias;
Block=size(sample,1);
H=sample*weight+repmat(bias,Block,1);
H=1./(1+exp(-H));
H=[sample,H];

M=M-M*H'*((eye(Block)+H*M*H')\H*M);
beta=beta+M*H'*(label-H*beta);
RVFL_tracker.clsf.M=M;
RVFL_tracker.clsf.beta=beta;
% weight=RVFL_tracker.clsf.weight;
% bias=RVFL_tracker.clsf.bias;
% nSam=size(sample,1);
% H=sample*weight+repmat(bias,nSam,1);
% H=1./(1+exp(-H));
% H=[sample,H];
% lambda=RVFL_tracker.lambda;
% %beta=(H'*H+lambda*eye(size(H,2)))\H'*label;
% %beta=H'*((lambda*eye(size(H,1))+H*H')\label);
% if size(H,1)>size(H,2)
% beta=(H'*H+lambda*eye(size(H,2)))\H'*label;
% else
% beta=H'*((lambda*eye(size(H,1))+H*H')\label);
% end
RVFL_tracker.clsf.beta=beta;
%     idx_new=size(RVFL_tracker.pos_sv,1)+size(RVFL_tracker.neg_sv,1)+1:numel(label);
%     y=H*beta;
%     y=y(idx_new,:);
%     Nsv=0.1*nSam;
%      Nsv=round(0.1*nSam);
%     if Nsv<1
%        Nsv=1; 
%     end
%     
%     label_temp=label(idx_new);
%     err=abs(y-label_temp);
%     idx_pos=find(label_temp>0.5);
%     idx_neg=find(label_temp<=0.5);
%     err_pos=err(idx_pos);
%     err_neg=err(idx_neg);
%     [val,idx_temp]=sort(err_pos,'descend');
%     idx_pos=idx_pos(idx_temp(1:Nsv));
%     idx_pos=idx_new(idx_pos);
%     [val,idx_temp]=sort(err_neg,'descend');
%     idx_neg=idx_neg(idx_temp(1:Nsv));
%     idx_neg=idx_new(idx_neg);
%     pos_sv_new=sample(idx_pos,:);
%     neg_sv_new=sample(idx_neg,:);
%    size(pos_sv_new,1)
%     size(neg_sv_new,1)
%    idx_new(1)
%    size(sample)
%**************************
% RVFL_tracker.w = RVFL_tracker.clsf.Alpha'*RVFL_tracker.clsf.SupportVectors;
% RVFL_tracker.Bias = RVFL_tracker.clsf.Bias;
% RVFL_tracker.clsf.w = RVFL_tracker.w;
% get the idx of new svs
% sv_idx = RVFL_tracker.clsf.SupportVectorIndices;
% sv_old_sz = size(RVFL_tracker.pos_sv,1)+size(RVFL_tracker.neg_sv,1);
% sv_new_idx = sv_idx(sv_idx>sv_old_sz);
% sv_new = sample(sv_new_idx,:);
% sv_new_label = label(sv_new_idx,:);
        
% num_sv_pos_new = sum(sv_new_label);
        
% update pos_dis, pos_w and pos_sv
%pos_sv_new = sv_new(sv_new_label>0.5,:);
% if ~isempty(pos_sv_new)
%     if size(pos_sv_new,1)>1
%         pos_dis_new = squareform(pdist(pos_sv_new));
%     else
%         pos_dis_new = 0;
%     end
%     pos_dis_cro = pdist2(RVFL_tracker.pos_sv,pos_sv_new);
%     
%     RVFL_tracker.pos_dis = [RVFL_tracker.pos_dis, pos_dis_cro; pos_dis_cro', pos_dis_new];
%     RVFL_tracker.pos_sv = [RVFL_tracker.pos_sv;pos_sv_new];
% %    RVFL_tracker.pos_w = [RVFL_tracker.pos_w;ones(num_sv_pos_new,1)];
% end
%         
% % update neg_dis, neg_w and neg_sv
% % neg_sv_new = sv_new(sv_new_label<0.5,:);
% if ~isempty(neg_sv_new)
%     if size(neg_sv_new,1)>1
%         neg_dis_new = squareform(pdist(neg_sv_new));
%     else
%         neg_dis_new = 0;
%     end
%     neg_dis_cro = pdist2(RVFL_tracker.neg_sv,neg_sv_new);
%   
%     
%     RVFL_tracker.neg_dis = [RVFL_tracker.neg_dis, neg_dis_cro; neg_dis_cro', neg_dis_new];
%     RVFL_tracker.neg_sv = [RVFL_tracker.neg_sv;neg_sv_new];
%   %  RVFL_tracker.neg_w = [RVFL_tracker.neg_w;ones(size(sv_new,1)-num_sv_pos_new,1)];
% end
        
% RVFL_tracker.pos_dis = RVFL_tracker.pos_dis + diag(inf*ones(size(RVFL_tracker.pos_dis,1),1));
% RVFL_tracker.neg_dis = RVFL_tracker.neg_dis + diag(inf*ones(size(RVFL_tracker.neg_dis,1),1));
%         
        
% compute real margin
% pos2plane = -RVFL_tracker.pos_sv*RVFL_tracker.w';
% neg2plane = -RVFL_tracker.neg_sv*RVFL_tracker.w';
% RVFL_tracker.margin = (min(pos2plane) - max(neg2plane))/norm(RVFL_tracker.w);
%         
% % shrink svs
% % check if to remove
% if size(RVFL_tracker.pos_sv,1)+size(RVFL_tracker.neg_sv,1)>RVFL_tracker.B
% %     pos_score_sv = -(RVFL_tracker.pos_sv*RVFL_tracker.w'+RVFL_tracker.Bias);
% %     neg_score_sv = -(RVFL_tracker.neg_sv*RVFL_tracker.w'+RVFL_tracker.Bias);
%     H= RVFL_tracker.pos_sv*RVFL_tracker.clsf.weight+repmat(RVFL_tracker.clsf.bias,size(RVFL_tracker.pos_sv,1),1);
%     H=1./(1+exp(-H));
%     H=[RVFL_tracker.pos_sv,H];
%     pos_score_sv=H*beta;
%     %%
%     H= RVFL_tracker.neg_sv*RVFL_tracker.clsf.weight+repmat(RVFL_tracker.clsf.bias,size(RVFL_tracker.neg_sv,1),1);
%     H=1./(1+exp(-H));
%     H=[RVFL_tracker.neg_sv,H];
%     neg_score_sv=H*beta;
%     m_pos = pos_score_sv < RVFL_tracker.m2;
%     m_neg = neg_score_sv >RVFL_tracker.m2;
%             
%     if config.verbose
%         fprintf('remove svs: pos %d, neg %d \n',sum(~m_pos),sum(~m_neg));
%     end
%     if sum(m_pos) > 0
%         RVFL_tracker.pos_sv = RVFL_tracker.pos_sv(m_pos,:);
%       %  RVFL_tracker.pos_w = RVFL_tracker.pos_w(m_pos,:);
%         RVFL_tracker.pos_dis = RVFL_tracker.pos_dis(m_pos,m_pos);
%     end
% 
%     if sum(m_neg)>0
%         RVFL_tracker.neg_sv = RVFL_tracker.neg_sv(m_neg,:);
%       %  RVFL_tracker.neg_w = RVFL_tracker.neg_w(m_neg,:);
%         RVFL_tracker.neg_dis = RVFL_tracker.neg_dis(m_neg,m_neg);
%     end
% end
%         
% check if to merge
% while size(RVFL_tracker.pos_sv,1)+size(RVFL_tracker.neg_sv,1)>RVFL_tracker.B
%     [mm_pos,idx_pos] = min(RVFL_tracker.pos_dis(:));
%     [mm_neg,idx_neg] = min(RVFL_tracker.neg_dis(:));
%             
%     if mm_pos > mm_neg || size(RVFL_tracker.pos_sv,1) <= RVFL_tracker.B_p% merge negative samples
%         if config.verbose
%             fprintf('merge negative samples: %d \n', size(RVFL_tracker.neg_w,1))
%         end
%                 
%         [i,j] = ind2sub(size(RVFL_tracker.neg_dis),idx_neg);
% %         w_i= RVFL_tracker.neg_w(i);
% %         w_j= RVFL_tracker.neg_w(j);
%        w_i=1;
%        w_j=1;
%         merge_sample = (w_i*RVFL_tracker.neg_sv(i,:)+w_j*RVFL_tracker.neg_sv(j,:))/(w_i+w_j);                
%                 
%         RVFL_tracker.neg_sv([i,j],:) = []; RVFL_tracker.neg_sv(end+1,:) = merge_sample;
%        % RVFL_tracker.neg_w([i,j]) = []; RVFL_tracker.neg_w(end+1,1) = w_i + w_j;
%                 
%         RVFL_tracker.neg_dis([i,j],:)=[]; RVFL_tracker.neg_dis(:,[i,j])=[];
%         neg_dis_cro = pdist2(RVFL_tracker.neg_sv(1:end-1,:),merge_sample);
%         RVFL_tracker.neg_dis = [RVFL_tracker.neg_dis, neg_dis_cro;neg_dis_cro',inf];                
%     else
%         if config.verbose
%             fprintf('merge positive samples: %d \n', size(RVFL_tracker.pos_w,1))
%         end
% 
%         [i,j] = ind2sub(size(RVFL_tracker.pos_dis),idx_pos);
% %         w_i= RVFL_tracker.pos_w(i);
% %         w_j= RVFL_tracker.pos_w(j);
% w_i=1;
% w_j=1;
%         merge_sample = (w_i*RVFL_tracker.pos_sv(i,:)+w_j*RVFL_tracker.pos_sv(j,:))/(w_i+w_j);                
% 
%         RVFL_tracker.pos_sv([i,j],:) = []; RVFL_tracker.pos_sv(end+1,:) = merge_sample;
%        % RVFL_tracker.pos_w([i,j]) = []; RVFL_tracker.pos_w(end+1,1) = w_i + w_j;
%                 
%         RVFL_tracker.pos_dis([i,j],:)=[]; RVFL_tracker.pos_dis(:,[i,j])=[];
%         pos_dis_cro = pdist2(RVFL_tracker.pos_sv(1:end-1,:),merge_sample);
%         RVFL_tracker.pos_dis = [RVFL_tracker.pos_dis, pos_dis_cro;pos_dis_cro',inf]; 
%                 
%                 
%     end
% %             
% end
        
% update experts
% experts{end}.w = RVFL_tracker.clsf.w;
% experts{end}.Bias = RVFL_tracker.Bias;
experts{end}.snapshot=RVFL_tracker;      
RVFL_tracker.update_count = RVFL_tracker.update_count + 1;

