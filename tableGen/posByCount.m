function counts = posByCount(limeJson)
%function posCounts = posByCount(limeJson)
    counts = {};
    sanitizedCategories = {};
    
    sentences = limeJson.sentences;
    numSentences = length(sentences);
    for i = 1:numSentences
        addPhraseConstituentCounts(sentences(i).phrase_structure);
    end

    function addPhraseConstituentCounts(phrase)
        isLeaf = ~isfield(phrase, 'constituents');
        %don't process leaves because they are words and not POS
        if ~isLeaf
            category = phrase.category;
            addToCounts(category);
            
            children = phrase.constituents;
            numChildren = length(children);
            for c = 1:numChildren
                addPhraseConstituentCounts(children(c));
            end
        end
    end

    function addToCounts(category)
        if ~all(isletter(category))
            origCategory = category;
            category = sanitizeFieldSub(origCategory);
            if ~isfield(sanitizedCategories, category)
                sanitizedCategories.(category) = 1;
                warning('catagory rename: %s => %s', origCategory, category);
            end
        end
        
        category = sanitizeFieldSub(category);
        
        if isfield(counts, category)
            counts.(category) = counts.(category) + 1;
        else
            counts.(category) = 1;
        end
    end

end

function field = sanitizeFieldSub(field)
    field = strrep(field, '-', 'dash');
    field = strrep(field, '$', 'dollar');
    field = strrep(field, '.', 'period');
    field = strrep(field, ':', 'colon');
    field = strrep(field, '(', 'leftParen');
    field = strrep(field, ')', 'rightParen');
    field = strrep(field, ',', 'comma');
end