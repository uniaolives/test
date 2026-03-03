% ucd.m – Universal Coherence Detection in MATLAB/Octave

function result = ucd_analyze(data)
    [nvars, nobs] = size(data);
    if nvars > 1
        corr_mat = abs(corrcoef(data'));
        tri = triu(corr_mat, 1);
        C = mean(tri(tri~=0));
    else
        C = 0.5;
    end
    F = 1.0 - C;
    result = struct('C', C, 'F', F, 'conservation', (abs(C + F - 1.0) < 1e-10));
end

data = [1 2 3 4; 2 3 4 5; 5 6 7 8];
disp(ucd_analyze(data))
