% run to align kdf and ns5 timestamps for trainTIME segment

function varargout = load_ns5_segments(session_dir_path, ns5_filepath, kdf_training_timestamps)

% --- Input Validation ---
arguments
    session_dir_path        (1, 1) string
    ns5_filepath            (1, 1) string {mustBeFile}
    kdf_training_timestamps (1, :) double
end

% --- Path Resolution ---
ns5_path = char(ns5_filepath);
ns2_path = regexprep(ns5_path, '\.ns5$', '.ns2');
[~, experiment_id] = fileparts(strip(session_dir_path, "right", filesep));

% --- Synchronization ---
fprintf("\nLOADING OFFSET SEGMENT\n\t")
nip_offset = calculate_offset_robustly(session_dir_path, experiment_id, ns2_path);

% --- Data Extraction Factory ---
read_segment_fcn = @(range) extract_and_scale_segment(ns5_path, range);

% --- Training Data Segment ---
fprintf("\nLOADING TRAINING SEGMENT\n\t")
training_range = [kdf_training_timestamps(1), kdf_training_timestamps(end)] + nip_offset;
varargout{1} = read_segment_fcn(training_range);
end   % <-- closes main function


function scaled_data = extract_and_scale_segment(file_path, nip_range)

[header] = unrl_utils.fastNSxRead('File', file_path);

if nip_range(2) > header.ChannelSamples
    error('NeuralData:RangeOutOfBounds', ...
        'Range [%d %d] exceeds available samples (%d)', ...
        nip_range(1), nip_range(2), header.ChannelSamples)
end

analog_range = double(header.MaxAnlgVal(1)) - double(header.MinAnlgVal(1));
digital_range = double(header.MaxDigVal(1)) - double(header.MinDigVal(1));
gain = analog_range / digital_range;

[~, raw_data] = unrl_utils.fastNSxRead('File', file_path, 'Range', nip_range);

scaled_data = single(raw_data(1:header.ChannelCount, :)') .* gain;

end


function offset = calculate_offset_robustly(folder_path, experiment_id, ns2_path)

file_patterns = {
    fullfile(folder_path, "RecStart_" + experiment_id + ".mat");
    fullfile(folder_path, "SSStruct_" + experiment_id + ".mat")
    };

for i = 1:length(file_patterns)
    if exist(file_patterns{i}, 'file')
        offset = project_utils.CalculateNIPOffset_bhm(ns2_path, file_patterns{i});
        return;
    end
end

error('NeuralData:SyncError', ...
    'Could not locate synchronization file for: %s', experiment_id);

end
