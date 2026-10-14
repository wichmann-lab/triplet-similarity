function out = DPrimeFromCorrect(myPercentCorrect, myHitRate, myPs)

if nargin < 3, myPs = 0.5; end
if nargin < 2, myHitRate = myPercentCorrect; end

% 1. Calculate false alarm rate
out.falseAlarmRate = (myPercentCorrect - 1 + myPs*(1 - myHitRate))/(myPs - 1);

% 2. Calculate the criterion lambda
out.lambda = icdf('normal', 1-out.falseAlarmRate, 0, 1);

% 3. Calculate d-Prime
out.dPrime = out.lambda - icdf('normal', 1-myHitRate, 0, 1);

% 4. Add values into struct
out.percentCorrect = myPercentCorrect;
out.hitRate = myHitRate;
out.ps = myPs;

end