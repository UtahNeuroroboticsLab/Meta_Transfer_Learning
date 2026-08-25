% Example driver for timestamp-aligning one KDF file to its NS5 recording.
% Edit only the session directory and filenames below.

pipelineDir = string(fileparts(mfilename('fullpath')));
addpath(pipelineDir);

sessionDir = "C:\path\to\recording-session";
kdfFile = fullfile(sessionDir, "TrainingData_<session-id>.kdf");
ns5File = fullfile(sessionDir, "<session-id>.ns5");
outDir = fullfile(sessionDir, "aligned_h5");

result = alignAndExportKDFNS5(sessionDir, kdfFile, ns5File, outDir);
disp(result)
