function [agregado] = addTimeSeriesToWeek(TimeSeries, agregado, bitsPaquetes)
    date = datestr(datetime(TimeSeries(1,1)+7200, 'convertfrom', 'posixtime'));
    [weekDay_numb, weekDay_char] = weekday(date);
    %1: domingo, 2: lunes, 3: martes..., 7: sábado
    weekDay_numb = weekDay_numb-2;
    if(weekDay_numb < 0)
        weekDay_numb = 6; %Domingo
    end
    [day, hourminsec] = strtok(date);
    [hour, minsec] = strtok(hourminsec, ':');
    hour = str2double(hour(2:end));
    [min, sec] = strtok(minsec, ':');
    min = str2double(min);
    sec = str2double(sec(2:end));
    index_sec = weekDay_numb + hour*60*60 + min*60 + sec;
    
    agregado = [agregado; zeros(1, size(agregado, 2))];
    agregado(end, index_sec:index_sec+size(TimeSeries,1)-1) = TimeSeries(:,bitsPaquetes)';
end