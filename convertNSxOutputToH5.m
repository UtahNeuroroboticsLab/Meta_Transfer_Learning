function convertNSxOutputToH5(nsxFile, outFile, varargin)
% convertNSxOutputToH5 Convert NSx data to Python-friendly HDF5.
%
% Example:
% convertNSxOutputToH5("file.ns5", "output.h5")
%
% Example with selected channels:
% convertNSxOutputToH5("file.ns5", "output_ch1_16.h5", "Channels", 1:16)

p = inputParser;
addRequired(p, 'nsxFile');
addRequired(p, 'outFile');
addParameter(p, 'Range', []);
addParameter(p, 'Channels', []);
parse(p, nsxFile, outFile, varargin{:});

nsxFile = char(p.Results.nsxFile);
outFile = char(p.Results.outFile);
range = p.Results.Range;
channels = p.Results.Channels;

if isempty(range)
    [Header, data] = fastNSxRead('File', nsxFile);
else
    [Header, data] = fastNSxRead('File', nsxFile, 'Range', range);
end

if ~isempty(channels)
    data = data(channels, :);
end

if exist(outFile, 'file')
    delete(outFile);
end

data = int16(data);

h5create(outFile, '/data', size(data), 'Datatype', 'int16');
h5write(outFile, '/data', data);

h5writeatt(outFile, '/', 'source_file', nsxFile);
h5writeatt(outFile, '/', 'data_layout', 'channels_x_samples');
h5writeatt(outFile, '/', 'data_dtype', 'int16');

if isfield(Header, 'Fs')
    h5writeatt(outFile, '/', 'Fs', Header.Fs);
end

if isfield(Header, 'nElec')
    h5writeatt(outFile, '/', 'nElec', Header.nElec);
end

if isfield(Header, 'nSamples')
    h5writeatt(outFile, '/', 'nSamples', Header.nSamples);
end

disp("Wrote HDF5: " + outFile)
disp("Data shape: " + size(data,1) + " channels x " + size(data,2) + " samples")
end