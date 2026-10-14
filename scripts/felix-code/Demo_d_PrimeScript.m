% Demo script: We want a minimum change in d' to ensure reasonable effect
% size.
minP = 0.5;
maxP = 1.0;
N = 50;
min_d_Prime_Delta = 0.5;

% For a number of different levels of performance, how much increase do we
% require:
p = linspace(minP, maxP, N);
dat = DPrimeFromCorrect(p);
min_d_PrimeValues = dat.dPrime + min_d_Prime_Delta;

min_p_values = CorrectFromDPrime(min_d_PrimeValues, 'YesNo');
delta_p = min_p_values - p;

subplot(1,2,1)
h = plot(p, min_p_values, 'o'); axis square
line([minP maxP], [minP maxP])
xlabel('prediction performance for dimensionality D')
ylabel('minimally required prediction performance for dimensionality D+1')
title(['requested minimal change in d-Prime = ' num2str(min_d_Prime_Delta)])

subplot(1,2,2)
h = plot(p, delta_p, 'o'); axis square
xlabel('prediction performance for dimensionality D')
ylabel('minimally required better prediction performance for dimensionality D+1')
title(['requested minimal change in d-Prime = ' num2str(min_d_Prime_Delta)])
