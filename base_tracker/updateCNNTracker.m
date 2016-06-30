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

function updateCNNTracker(sample,label,sample_w)
global config;
global CNN_tracker;
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
 N=round(1+2*s2/s1);  

 pos=sample(:,:,:,pos_mask);
 
 pos_label=label(pos_mask);
 
 pos=repmat(pos,1,1,1,N);
 pos=pos+0.05*randn(size(pos));
 pos_label=repmat(pos_label,N,1);
 %sample=[sample;pos];
 sample=cat(4,sample,pos);
%  sample=sample-repmat(mean(sample,1),size(sample,1),1);
%  Norm_s=sqrt(sum(sample.^2,1));
%  sample=sample./repmat(Norm_s,size(sample,1),1);
%  sample(isnan(sample))=0;
label=[label;pos_label]; 
% sample = [CNN_tracker.pos_sv;CNN_tracker.neg_sv; sample];
% label = [ones(size(CNN_tracker.pos_sv,1),1);zeros(size(CNN_tracker.neg_sv,1),1);label];
%C = max(CNN_tracker.C*sample_w/sum(sample_w),0.001);

% if config.verbose
% %     CNN_tracker.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{CNN_tracker.struct_mat},...
% %        'boxconstraint',C,'autoscale','false','options',statset('Display','final','MaxIter',5000));
% %     fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(CNN_tracker.clsf.Alpha,1)); 
% else
% %     CNN_tracker.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{CNN_tracker.struct_mat},...
% %        'boxconstraint',C,'autoscale','false','options',statset('MaxIter',5000));
% end
M=CNN_tracker.M;
beta=CNN_tracker.beta;
sample=single(sample);
H=vl_simplenn(CNN_tracker,sample);
H=H(end).x;
size_temp=size(H);
H=reshape(H,[],size_temp(4))';
sample1=reshape(sample,[],size_temp(4))';
sample1=normalize_feature(sample1);
%H=normalize_feature(H);

H=[H,sample1];
%H=1./exp(-1*H); 
%H=normalize_feature(H);
Block=size(H,1);
M=M-M*H'*((eye(Block)+H*M*H')\H*M);
beta=beta+M*H'*(label-H*beta);
CNN_tracker.M=M;
CNN_tracker.beta=beta;
% weight=CNN_tracker.clsf.weight;
% bias=CNN_tracker.clsf.bias;
% nSam=size(sample,1);
% H=sample*weight+repmat(bias,nSam,1);
% H=1./(1+exp(-H));
% H=[sample,H];
% lambda=CNN_tracker.lambda;
% %beta=(H'*H+lambda*eye(size(H,2)))\H'*label;
% %beta=H'*((lambda*eye(size(H,1))+H*H')\label);
% if size(H,1)>size(H,2)
% beta=(H'*H+lambda*eye(size(H,2)))\H'*label;
% else
% beta=H'*((lambda*eye(size(H,1))+H*H')\label);
% end


experts{end}.snapshot=CNN_tracker;      
%CNN_tracker.update_count = CNN_tracker.update_count + 1;

