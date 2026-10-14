function p = CorrectFromDPrime(dPrime, whichTask)

if nargin < 2, whichTask = '2AFC'; end
if nargin < 1, dPrime = linspace(0, 6, 100); end

switch whichTask
    case '2AFC'
        % For 2AFC percent correct is the integral from zero to infinity of a
        % Gaussian with SD = sqrt(2) and mean = d'
        p =  1 - cdf('Normal', 0, dPrime, sqrt(2) );
    case 'YesNo'
        % Assuming the same number of signal and no-signal trials, for YesNo
        % percent correct is 1/2 Hits + 1/2 correct rejections. For an optimal
        % observer the criterion is placed at dPrime/2. Given symmetry of criterion
        % and variance of signal and noise distribution, percent correct is the
        % integral from dPrime/2 to infinity of N(dPrime,1), thus:
        p = 1 - cdf('Normal', dPrime/2, dPrime, 1);
    otherwise
        error('Invalid task specified. Use ''2AFC'' or ''YesNo''.');
end

end