function feature=normalize_feature(feature)
tiny_value=exp(-10);
feature1=feature.^2;
nSam=size(feature,1);
feature1=sqrt(sum(feature1))+tiny_value;
feature1=repmat(feature1,nSam,1);
feature=feature./feature1;





end