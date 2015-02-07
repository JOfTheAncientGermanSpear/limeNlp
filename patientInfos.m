function table = patientInfos(jsonDir, numTake)
%function table = patientInfos(jsonDir, numTake)
    if nargin < 2
        numTake = inf;
    end
    
    jsonTags = {};
    function info = fileToPatientInfo(fileStruct)
        start = tic;

        id = regexpi(fileStruct.name, '[0-9]+', 'match');
        fullPath = [jsonDir filesep fileStruct.name];

        json = decodeJSON(fullPath);
        info = posByCount(json);
        jsonTags = union(jsonTags, fieldnames(info));

        info.id = id{:};

        sprintf('processed patient %s', info.id)
        toc(start)
    end

    patientFiles = dir([jsonDir filesep '*.json']);
    
    numPatientFiles = length(patientFiles);
    if numTake > numPatientFiles
        numTake = numPatientFiles;
    end
    
    patientFiles = patientFiles(1:numTake);
    
    infoCells = arrayfun(@fileToPatientInfo, patientFiles, 'UniformOutput', 0);
    
    template = {};
    template.id = '';
    for i = 1:length(jsonTags)
        template.(jsonTags{i}) = 0;
    end
    
    numInfos = length(infoCells);
    infos = repmat(template, 1, numInfos);
    
    for i = 1:numInfos
        infoCell = infoCells{i};
        fields = fieldnames(infoCell);
        numFields = length(fields);
        info = infos(i);
        for f = 1:numFields
            f = fields{f};
            info.(f) = infoCell.(f);
        end
        infos(i) = info;
    end
    
    table = struct2table(infos);
    
end