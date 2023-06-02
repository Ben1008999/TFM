function [domainFIT] = getDomainFIT(Tsventana, Granularidad_deteccion)
    domainFIT = [[-(Tsventana-1):0] + ceil(Granularidad_deteccion/2)]';
    stepdomainFIT = 1/(2*domainFIT(end));
    domainFIT = domainFIT*stepdomainFIT;
end