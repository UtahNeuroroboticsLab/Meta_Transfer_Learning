function convertKDFToH5(kdfFile, outFile)
%CONVERTKDFTOH5 Export every field in a KDF file to Python-friendly HDF5.
%
% Data in the KDF file are channels/features x records. The HDF5 datasets
% are written records x channels/features so h5py reads them naturally.

arguments
    kdfFile (1,1) string {mustBeFile}
    outFile (1,1) string
end

[kin, feat, targets, kalman, nipTime] = readKDF(kdfFile);

if isfile(outFile)
    delete(outFile);
end

% Dataset names match Data_Alignment.ipynb for drop-in use.
writeDataset(outFile, "/trainNIPtime", nipTime);
writeDataset(outFile, "/trainKin", kin);
writeDataset(outFile, "/trainFeat", feat);
writeDataset(outFile, "/trainTargets", targets);
writeDataset(outFile, "/trainKalman", kalman);

h5writeatt(outFile, '/', 'source_file', char(kdfFile));
h5writeatt(outFile, '/', 'data_layout', 'records_x_variables');
h5writeatt(outFile, '/', 'nip_time_units', '30_kHz_NIP_ticks');
h5writeatt(outFile, '/', 'nip_clock_hz', 30000);

fprintf('Wrote KDF HDF5: %s\n', outFile);
end

function writeDataset(outFile, dataset, data)
% HDF5 cannot create a zero-sized dataset via high-level MATLAB functions.
if isempty(data)
    return
end
data = single(data);
h5create(outFile, dataset, size(data), 'Datatype', 'single');
h5write(outFile, dataset, data);
end
