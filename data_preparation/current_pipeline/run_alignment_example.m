% Project locations after the MATLAB/Jupyter experiment reorganization.
projectDir = "C:\Users\Micah\utah-neuro\MATLAB_Jupyter";
repoDir = fullfile(projectDir, "00_data_preparation");
sessionDir = fullfile(repoDir, "session_20260723");
addpath(repoDir);

kdfFile = fullfile(sessionDir, "TrainingData_20260723-153348.kdf");
ns5File = fullfile(sessionDir, "20260723-153348.ns5");
outDir = fullfile(repoDir, "aligned_h5");

result = alignAndExportKDFNS5(sessionDir, kdfFile, ns5File, outDir);
disp(result)
