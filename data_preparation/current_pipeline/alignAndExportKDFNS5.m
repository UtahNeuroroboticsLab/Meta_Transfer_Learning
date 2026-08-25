function result = alignAndExportKDFNS5(sessionDir, kdfFile, ns5File, outDir)
%ALIGNANDEXPORTKDFNS5 Align a KDF recording to NS5 and export both to HDF5.
%
% result = alignAndExportKDFNS5(sessionDir, kdfFile, ns5File, outDir)
%
% KDF NIPTime and the calculated offset are 30 kHz NIP ticks. Therefore:
%   absolute NS5 sample = KDF NIPTime + NIP offset
% The exported NS5 begins at the first KDF timestamp and ends at the last.

arguments
    sessionDir (1,1) string {mustBeFolder}
    kdfFile (1,1) string {mustBeFile}
    ns5File (1,1) string {mustBeFile}
    outDir (1,1) string
end

if ~isfolder(outDir)
    mkdir(outDir);
end

[~, ~, ~, ~, nipTime] = readKDF(kdfFile);
if isempty(nipTime)
    error('Alignment:MissingNIPTime', 'The KDF contains no NIPTime records.');
end

nipTime = double(nipTime(:));
if any(~isfinite(nipTime)) || any(diff(nipTime) < 0)
    error('Alignment:InvalidNIPTime', ...
        'KDF NIPTime must be finite and monotonically nondecreasing.');
end

[ns5Folder, experimentID] = fileparts(ns5File);
ns2File = fullfile(ns5Folder, experimentID + ".ns2");
if ~isfile(ns2File)
    error('Alignment:MissingNS2', 'Required synchronization file not found: %s', ns2File);
end

syncCandidates = [ ...
    fullfile(sessionDir, "RecStart_" + experimentID + ".mat"), ...
    fullfile(sessionDir, "SSStruct_" + experimentID + ".mat"), ...
    fullfile(sessionDir, "RecStart-" + experimentID + ".mat"), ...
    fullfile(sessionDir, "SSStruct-" + experimentID + ".mat")];
syncFile = syncCandidates(find(isfile(syncCandidates), 1));
if isempty(syncFile)
    error('Alignment:MissingSyncFile', ...
        'No RecStart or SSStruct synchronization file found for %s.', experimentID);
end

nipOffset = double(CalculateNIPOffset(char(ns2File), char(syncFile)));
ns5Range = round([nipTime(1), nipTime(end)] + nipOffset);
if ns5Range(1) < 1
    error('Alignment:RangeBeforeFile', ...
        'Calculated first NS5 sample is %d; expected at least 1.', ns5Range(1));
end

baseName = erase(string(kdfFile), fileparts(kdfFile) + filesep);
baseName = regexprep(baseName, '\.kdf$', '', 'ignorecase');
kdfH5 = fullfile(outDir, baseName + "_kdf.h5");
ns5H5 = fullfile(outDir, baseName + "_ns5_aligned.h5");

convertKDFToH5(kdfFile, kdfH5);
convertNSxOutputToH5(ns5File, ns5H5, 'Range', ns5Range);

% Alignment metadata makes the mapping reproducible from either file.
for file = [kdfH5, ns5H5]
    h5writeatt(file, '/', 'nip_offset_samples', int64(nipOffset));
    h5writeatt(file, '/', 'ns5_first_absolute_sample', int64(ns5Range(1)));
    h5writeatt(file, '/', 'ns5_last_absolute_sample', int64(ns5Range(2)));
    h5writeatt(file, '/', 'kdf_first_nip_time', int64(nipTime(1)));
    h5writeatt(file, '/', 'kdf_last_nip_time', int64(nipTime(end)));
end

result = struct( ...
    'kdf_h5', kdfH5, ...
    'ns5_h5', ns5H5, ...
    'nip_offset_samples', nipOffset, ...
    'ns5_range', ns5Range, ...
    'local_ns5_index_for_each_kdf_record', ...
        round(nipTime - nipTime(1)) + 1);

fprintf('NIP offset: %d samples\n', nipOffset);
fprintf('Aligned NS5 range: [%d, %d]\n', ns5Range(1), ns5Range(2));
fprintf('For KDF record i: local NS5 row = NIPTime(i) - NIPTime(1) + 1\n');
end
