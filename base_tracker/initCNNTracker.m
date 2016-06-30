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

function initCNNTracker (sample,label,fuzzy_weight)

global CNN_tracker;
global experts;


sample_w = fuzzy_weight;
 pos_mask = label>0.5;
 neg_mask = ~pos_mask;
 s1 = sum(sample_w(pos_mask));
 s2 = sum(sample_w(neg_mask));
 lambda=CNN_tracker.lambda;    
 N=round(1+2*s2/s1);  
 S=size(sample);
 
 pos=sample(:,:,:,pos_mask);
 pos=repmat(pos,1,1,1,N);
 
 pos_label=label(pos_mask);
 
 pos=double(pos)+double(0.05*randn(size(pos)));
 pos_label=repmat(pos_label,N,1);
 %sample=[sample;pos];
 sample=cat(numel(S),sample,pos);
 label=[label;pos_label];
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
if size(H,1)>size(H,2)
beta=(H'*H+lambda*eye(size(H,2)))\H'*label;
else
beta=H'*((lambda*eye(size(H,1))+H*H')\label);
end
CNN_tracker.beta=beta; 
M=(H'*H+lambda*eye(size(H,2)))\eye(size(H,2));
CNN_tracker.M=M;
%     beta=model.beta;
%     weight=model.weight;
%     bias=model.bias;
%     Sample=patterns';
%     Sample=Sample-repmat(mean(Sample,1),size(Sample,1),1);
%  Norm_s=sqrt(sum(Sample.^2,1));
%  Sample=Sample./repmat(Norm_s,size(Sample,1),1);
%  Sample(isnan(Sample))=0;
%     nSam=size(sample,1);
%     H=sample*weight+repmat(bias,nSam,1);
%     H=1./(1+exp(-H));
%     H=[sample,H];
%     y=H*beta;
%     Nsv=round(0.1*nSam);
%     if Nsv>100
%        Nsv=10; 
%     end
%     
%     err=abs(y-label);
%     idx_pos=find(label>0.5);
%     idx_neg=find(label<=0.5);
%     err_pos=err(idx_pos);
%     err_neg=err(idx_neg);
%     [val,idx_temp]=sort(err_pos,'descend');
%     idx_pos=idx_pos(idx_temp(1:Nsv));
%     [val,idx_temp]=sort(err_neg,'descend');
%     idx_neg=idx_neg(idx_temp(1:Nsv));
%    
%     CNN_tracker.pos_sv=sample(idx_pos,:);
%     CNN_tracker.neg_sv=sample(idx_neg,:);
%   
% if size(CNN_tracker.pos_sv,1)>1
%     CNN_tracker.pos_dis = squareform(pdist(CNN_tracker.pos_sv));
% else
%     CNN_tracker.pos_dis = inf;
% end
% CNN_tracker.neg_dis = squareform(pdist(CNN_tracker.neg_sv)); 
    
% CNN_tracker.w = CNN_tracker.clsf.w;
% CNN_tracker.Bias = CNN_tracker.clsf.Bias;
% CNN_tracker.sv_label = label(CNN_tracker.clsf.SupportVectorIndices,:);
% CNN_tracker.sv_full = sample(CNN_tracker.clsf.SupportVectorIndices,:);
%         
% CNN_tracker.pos_sv = CNN_tracker.sv_full(CNN_tracker.sv_label>0.5,:);
% CNN_tracker.pos_w = ones(size(CNN_tracker.pos_sv,1),1);
% CNN_tracker.neg_sv = CNN_tracker.sv_full(CNN_tracker.sv_label<0.5,:);
% CNN_tracker.neg_w = ones(size(CNN_tracker.neg_sv,1),1);
        
% compute real margin
% pos2plane = -CNN_tracker.pos_sv*CNN_tracker.w';
% neg2plane = -CNN_tracker.neg_sv*CNN_tracker.w';
% CNN_tracker.margin = (min(pos2plane) - max(neg2plane))/norm(CNN_tracker.w);
        
% calculate distance matrix
% if size(CNN_tracker.pos_sv,1)>1
%     CNN_tracker.pos_dis = squareform(pdist(CNN_tracker.pos_sv));
% else
%     CNN_tracker.pos_dis = inf;
% end
% CNN_tracker.neg_dis = squareform(pdist(CNN_tracker.neg_sv)); 
        
%% intialize tracker experts
% experts{1}.w = CNN_tracker.w;
% experts{1}.Bias = CNN_tracker.Bias;
experts{1}.score = [];
experts{1}.snapshot = CNN_tracker;
        
 experts{2} = experts{1};