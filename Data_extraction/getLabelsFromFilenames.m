function [labels] = getLabelsFromFilenames(filenames_array)
    labels = [];
    tokens = [];
    
    for i=1:length(filenames_array)
        str = filenames_array(i);
        [token, remainder] = strtok(str, '/');
        tokens = [tokens, token];
        %Continue splitting the remainder until there are no more tokens
        while remainder ~= ""
            [token, remainder] = strtok(remainder, '/');
            tokens = [tokens, token];
        end
        labels = [labels, tokens(end-1)];
    end
end