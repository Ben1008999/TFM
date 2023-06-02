function [] = writeJSON(filenameOutput, JSON)
    fid = fopen(filenameOutput,'w');
    encodedJSON = jsonencode(JSON); 
    fprintf(fid, encodedJSON);
    fclose('all'); 
end