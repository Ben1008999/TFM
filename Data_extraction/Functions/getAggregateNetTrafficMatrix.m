function [agregado] = getAggregateNetTrafficMatrix(filenames, bitsPaquetes, domain)
    agregado = zeros(1, length(domain));
    agregado(1,:) = domain;
    for i=1:length(filenames)
        filename = filenames(i);
        time_serie = load(filename);
        agregado = addTimeSeriesToWeek(time_serie, agregado, bitsPaquetes);
    end
    agregado(find(agregado <= 0)) = NaN;
end