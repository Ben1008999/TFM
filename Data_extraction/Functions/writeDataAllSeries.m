function [] = writeDataAllSeries(filename_output, agregado)
    fileID = fopen(filename_output, 'w');
    labels = agregado{1,1};
    data = agregado{1,2};
    % Write the data to the file
    for i=1:length(labels)
        fprintf(fileID, '%s, ', labels(i));
        fprintf(fileID, '%f, ', data(i,:));
        fprintf(fileID, '\n');
    end
    % Close the file
    fclose(fileID);
end