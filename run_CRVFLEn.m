function results=run_CRVFLEn(seq, res_path, bSaveImage)

close all;
results = MEEMTrack_v1(seq.path, seq.ext, false, seq.init_rect, seq.startFrame,seq.endFrame);
end