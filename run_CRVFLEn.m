function results=run_CRVFLEn(seq, res_path, bSaveImage)

close all;
results = CRVFLEn(seq.path, seq.ext, false, seq.init_rect, seq.startFrame,seq.endFrame);
end
