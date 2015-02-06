function table = patientTableNlp(patientDir)
    function info = fileToPatientInfo(fileStruct)
        id = regexpi(fileStruct.name, '[0-9]+', 'match');
        info.id = id{:};
        info.fullPath = [patientDir filesep fileStruct.name];
        info.json = decodeJSON(info.fullPath);
        sprintf(id)
    end

    patientFiles = dir([patientDir filesep '*.json']);
    table = arrayfun(@fileToPatientInfo, patientFiles);
end