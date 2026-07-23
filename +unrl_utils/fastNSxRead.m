function varargout = fastNSxRead(varargin)
%FASTNSXREAD Package-compatible entry point used by load_ns5_segments.m.
% Delegates to the retained fastNSxRead2022 implementation.
[varargout{1:nargout}] = fastNSxRead2022(varargin{:});
end
