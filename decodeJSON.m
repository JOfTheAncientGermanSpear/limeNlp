function data = decodeJSON(json)
%   FOR INTERNAL USE ONLY -- This function is intentionally undocumented
%   and is intended for use only within the scope of functions and classes
%   in toolbox/matlab/external/interfaces/webservices/restful. Its behavior
%   may change, or the class itself may be removed in a future release.

% Copyright 2014 The MathWorks, Inc.

% decodeJSON parses a JSON string and returns the parsed results. JSON
% objects are converted to structures and JSON arrays are converted to cell
% arrays. JSON numbers are returned as a double. JSON strings are returned
% as a string. This function is a copy of mls.internal.fromJSON and will be
% removed in a future release. It has been modified to add the ability to
% convert JSON objects with non-standard names to valid MATLAB field names.

% The original code here was modified too. We only expect empty or
% character strings from a Web server. A Web services does not send a cell
% array of data. Sending an empty cell array does not make sense when no
% data is returned from a Web servce.

if exist(json, 'file'), json = fileread(json); end;

json = strtrim(json);
if isempty(json)
    data = '';
else
    data = parse_value(json, 1, 1);
end
end

%--------------------------------------------------------------------------

function [value, offset] = parse_value(json, offset, depth)
% Parse value from JSON data array.

value = [];
if offset <= numel(json)
    offset = consume_whitespace(json, offset);
    id = json(offset);
    offset = offset + 1;
    
    switch lower(id)
        case '"'
            [value, offset] = parse_string(json, offset);
            
        case '{'
            [value, offset] = parse_object(json, offset);
            
        case '['
            [value, offset] = parse_array(json, offset, depth);
            
        % true is the only top level type starting with 't'
        case 't'
            value = true;
            if numel(json) - offset >= 2
                offset = offset + 3;
            else
                ME = MException('MATLAB:connector:Platform:InvalidParseValue', ...
                    getString(message('MATLAB:connector:Platform:InvalidParseValue', 'TRUE', id, json)));
                ME.throw;
            end
            
        % false is the only top level type starting with 'f'
        case 'f'
            value = false;
            if numel(json) - offset >= 3
                offset = offset + 4;
            else
                ME = MException('MATLAB:connector:Platform:InvalidParseValue', ...
                    getString(message('MATLAB:connector:Platform:InvalidParseValue', 'FALSE', id, json)));
                ME.throw;
            end
            
        case 'n'
            value = [];
            if numel(json) - offset >= 2
                offset = offset + 3;
            else
                ME = MException('MATLAB:connector:Platform:InvalidParseValue', ...
                    getString(message('MATLAB:connector:Platform:InvalidParseValue', 'NULL', id, json)));
                ME.throw;
            end
            
        otherwise
            [value, offset] = parse_number(json, offset - 1); % Need to put the id back on the string
    end
end
end

%--------------------------------------------------------------------------

function [data, offset] = parse_array(json, offset, depth)
% Parse JSON array.

data = [];
while offset <= numel(json)
    if strcmp(json(offset),']') % Check if the array is closed
        data = reshape(data, [ones(1, depth - 1) numel(data) 1]);
        offset = offset + 1;
        try
            % do not try to convert a cell array to a matrix if it
            % contains characters
            if ~isempty(data) && iscell(data) && ~iscellstr(data)
                data = cell2mat(data);
            elseif isempty(data)
                data = [];
            end
        catch %#ok<CTCH>
        end
        if iscell(data) && depth > 1
            % Remove singleton dimensions in each element of data which 
            % can result from the earlier reshape.
            data = cellfun(@squeeze, data, 'UniformOutput', false);
        end
        return
    end
    
    [value, offset] = parse_value(json, offset, depth + 1);
    data{end+1} = value; %#ok<AGROW>
    
    offset = consume_whitespace(json, offset);
    if offset <= numel(json) && json(offset) == ','
        offset = offset + 1;
        offset = consume_whitespace(json, offset);
    end
end
end

%--------------------------------------------------------------------------

function [data, offset] = parse_object(json, offset)
% Parse a JSON object.

names = {};
values = {};
while offset < numel(json)
    id = json(offset);
    offset = offset + 1;
    
    switch id
        case '"' % Start a name/value pair
            [name, value, offset] = parse_name_value(json, offset);
            if isempty(name)
                ME = MException('MATLAB:connector:Platform:ParseEmptyName', ...
                    getString(message('MATLAB:connector:Platform:ParseEmptyName', json)));
                ME.throw;
            end
            % data.(name) = value;
            names{end+1}  = name;  %#ok<AGROW>
            values{end+1} = value; %#ok<AGROW>
            
        case '}' % End of object, so exit the function
            data = makeStructure(names, values);
            return
            
        otherwise % Ignore other characters
    end
end
data = makeStructure(names, values);
end

%--------------------------------------------------------------------------

function data = makeStructure(names, values)
% Create a structure from names and values cell arrays. Ensure the names
% are unique.

names = matlab.lang.makeUniqueStrings(names);
names = matlab.lang.makeValidName(names);
data = cell2struct(values, names, 2);
if isempty(data)
    data = struct;
end
end

%--------------------------------------------------------------------------

function [name, value, offset] = parse_name_value(json, offset)
name = [];
value = [];
if offset <= numel(json)
    [name, offset] = parse_string(json, offset);
    
    % Skip spaces and the : separator
    offset = consume_whitespace(json, offset);
    if offset > numel(json) || json(offset) ~= ':'
        ME = MException('MATLAB:connector:Platform:MissingColon', ...
            getString(message('MATLAB:connector:Platform:MissingColon', json)));
        ME.throw;
    end
    offset = offset + 1;
    offset = consume_whitespace(json, offset);
    [value, offset] = parse_value(json, offset, 1);
end
end

%--------------------------------------------------------------------------

function [string, offset] = parse_string(json, offset)
string = '';
while offset <= numel(json)
    letter = json(offset);
    offset = offset + 1;
    
    switch lower(letter)
        case '\' % Deal with escaped characters
            if offset <= numel(json)
                code = json(offset);
                offset = offset + 1;
                switch lower(code)
                    case '"'
                        new_char = '"';
                    case '\'
                        new_char = '\';
                    case '/'
                        new_char = '/';
                    case {'b' 'f' 'n' 'r' 't'}
                        new_char = sprintf(['\' code]);
                    case 'u'
                        if numel(json) - offset >= 4
                            new_char = char(hex2dec(json(offset:offset+3)));
                            offset = offset + 4;
                        end
                    otherwise
                        new_char = [];
                end
            end
            
        case '"' % Done with the string
            return
            
        otherwise
            new_char = letter;
    end
    % Append the new character
    string = [string new_char]; %#ok<AGROW>
end
end

%--------------------------------------------------------------------------

function [num, offset] = parse_number(json, offset)
num = [];
if offset <= numel(json)
    
    tempOffset = offset;
    testChar = json(tempOffset);
    while tempOffset <= numel(json) && ...
            ((testChar <= '9' && testChar >= '0') || ...
            testChar == '+' || testChar == '-' || ...
            testChar == '.' || testChar == 'e' || testChar == 'E')
        tempOffset = tempOffset + 1;
        if tempOffset <= numel(json)
            testChar = json(tempOffset);
        end
    end
    num_str = json(offset:tempOffset-1);
    %offset = tempOffset;   % this can result in an infinite loop for 
                            % badly formatted nummbers
    if tempOffset == offset
        offset = offset + 1;
    else
        offset = tempOffset;
    end
    num = str2double(strtrim(num_str));
end
end

%--------------------------------------------------------------------------

function offset = consume_whitespace(json, offset)

if offset <= numel(json)
    testChar = json(offset);
    while offset < numel(json) && isspace(testChar)
        offset = offset + 1;
        testChar = json(offset);
    end
end
end