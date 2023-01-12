function results = eeg_mec(eeg, list_freqs, fs, num_harms)
[num_targs, num_channel, num_smpls] = size(eeg);
y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms);
num_class = length(list_freqs);
r=zeros(1,num_class);
results=zeros(num_targs,1);
nFFT = 256;
order_ar = 5;
for targ_i = 1:1:num_targs
    temp = eeg(targ_i, :, :);
    sizetemp=size(temp);
    test_tmp = reshape(temp,sizetemp(2:3))';
    for class_i = 1:1:num_class
        refdata = squeeze(y_ref(class_i, :, :))';
        
        tildeY = test_tmp - refdata * inv(refdata'*refdata)*refdata'*test_tmp;
        Ytemp = tildeY'*tildeY;
        [V,D] = eig(Ytemp);
        D = diag(D);
        Dsum = sum(D);
        for idx = 1:num_channel
            if sum(D(1:idx))/Dsum>=0.05
                break
            end
            
        end
        
        W = V(:, 1:idx);
        for ii = 1:idx
            W(:,ii) = W(:,ii)/sqrt(D(ii));
        end
        S = test_tmp * W;
        P = zeros(num_harms*2, idx);
        for iComb = 1:idx % in the paper iComb <-> l
            for iHarm = 1:num_harms*2 % in the paper iHarm <-> k
                P(iHarm,iComb) = norm( refdata(:,iHarm)'*S(:,iComb));
            end % of loop over harmonicsList                     
        end % of loop over combinations (or "channels")
        P = P .^ 2;
        
        tildeS = tildeY*W;
        tildeS = tildeS';
        nPxxRows = ceil( (nFFT+1) / 2 );
        Pxx = zeros( nPxxRows, idx);
        for iComb = 1:idx
            pxx = myPyulear( tildeS(iComb,:), order_ar,nFFT);
            Pxx(:,iComb) = pxx(1:nPxxRows);
        end % of loop over combinations (or "channels")
        sigma = zeros(num_harms*2, idx);
        div = fs / nFFT;
        for iComb = 1:idx          % loop over the signals in S
            for iHarm =1:num_harms*2  % loop over the harmonicsList of the frequenciesList
                ind = round(list_freqs(class_i) * floor((iHarm - 1)/2+1) / div );
                sigma(iHarm,iComb) = mean( Pxx(max(1,ind-1):min(ind+1,nPxxRows),iComb) );
            end % of loop over harmonicsList
        end % of loop over combinations (channels in S)
        P = P./sigma;
        r(class_i) = mean(P(:));
    end % class_i
    
    [~, tau] = max(r);
    results(targ_i) = tau;
end % targ_i
end

function [ y_ref ] = cca_reference(list_freqs, fs, num_smpls, num_harms)
if nargin < 3 
    error('stats:cca_reference:LackOfInput',...
        'Not enough input arguments.');
end

if ~exist('num_harms', 'var') || isempty(num_harms), num_harms = 3; end

num_freqs = length(list_freqs);
tidx = (1:num_smpls)/fs;
for freq_i = 1:1:num_freqs
    tmp = [];
    for harm_i = 1:1:num_harms
        stim_freq = list_freqs(freq_i);
        tmp = [tmp;...
            sin(2*pi*tidx*harm_i*stim_freq);...
            cos(2*pi*tidx*harm_i*stim_freq)];
    end % harm_i
    y_ref(freq_i, 1:2*num_harms, 1:num_smpls) = tmp;
end % freq_i

end

function Pxx = myPyulear( x, order, nFFT )
%   PYULEAR Power Spectral Density (PSD) estimate via Yule-Walker's method.
%   Pxx = PYULEAR( X, ORDER ) returns the PSD of a discrete-time signal vector
%   X in the vector Pxx.  Pxx is the distribution of power per unit frequency.
%   The frequency is expressed in units of radians/sample.  ORDER is the
%   order of the autoregressive (AR) model used to produce the PSD.  PYULEAR
%   uses a default FFT length of 256 which determines the length of Pxx.
%
%   For real signals, PYULEAR returns the one-sided PSD by default; for
%   complex signals, it returns the two-sided PSD.  Note that a one-sided
%   PSD contains the total power of the input signal.
%
%   Pxx = PYULEAR(X,ORDER,NFFT) specifies the FFT length used to calculate
%   the PSD estimates.  For real X, Pxx has length (NFFT/2+1) if NFFT is
%   even, and (NFFT+1)/2 if NFFT is odd.  For complex X, Pxx always has
%   length NFFT.  If empty, the default NFFT is 256.
%

    % inlining the line:  R = biasedXcorrVec( x, order ); --------------------
    M = numel( x );
    x = x(:);

    if( order > M )
        order = M;
    end
    
    % Compute correlation via FFT
    X = fft( x, 2^nextpow2( 2*M - 1 ) );
    R = real( ifft( abs( X ).^2 ) );
    R = R(1:order+1,:) / M;
    %--------------------------------
    % inlining the line:  R = biasedXcorrVec( x, order ); --------------------
%     R = zeros( order, 1 );
%     xx = [x; x(1:order)]';
%     for i = 1:order+1,
%         R(i) = xx(i:i+M-1)*x;
%     end
%     R = R / M;
    %--------------------------------
    [a, v] = myLevinson( R, order );    
    
    % inlining the line h = myFreqz( a, nFFT ); --------
    na = order + 1;
    if( nFFT < na )
        % Data is larger than FFT points, wrap modulo nfft
        % inlining the line a = myDatawrap( a, nFFT ); ---------
        nValuesToPad = mod( numel( a ), nFFT );
        if( nValuesToPad > 0 )
            a = [a zeros( 1, nValuesToPad) ];
        end
        a = sum( reshape( a, nFFT, [] ), 2 )';
    end
    h = ( ones( 1, nFFT ) ./ fft( a, nFFT ) )';
    %-------------
    
    Sxx = v * abs( h ).^2; % This is the power spectrum [Power] (input variance*power of AR model)
    
    % Compute the 1-sided or 2-sided PSD [Power/freq], or mean-square [Power].
    % Also, compute the corresponding frequency and frequency units.
    %     [Pxx,w,units] = computepsd( Sxx, w, options.range, options.nfft, options.Fs, 'psd' );
    %     Pxx = computepsd( Sxx, w, 'onesided', nFFT, [], 'psd' );
    % Generate the one-sided spectrum [Power]
    if( rem( nFFT, 2) ) % nFFT is ODD
        Sxx_unscaled = Sxx(1:(nFFT+1)/2,:); % Take only [0,pi] or [0,pi)
        Sxx = [Sxx_unscaled(1,:); 2*Sxx_unscaled(2:end,:)];  % Only DC is a unique point and doesn't get doubled
    else % nFFT is EVEN
        Sxx_unscaled = Sxx(1:nFFT/2+1,:); % Take only [0,pi] or [0,pi)
        Sxx = [Sxx_unscaled(1,:); 2*Sxx_unscaled(2:end-1,:); Sxx_unscaled(end,:)]; % Don't double unique Nyquist point
    end
    
    Pxx = Sxx ./ (2.*pi); % Scale the power spectrum by 2*pi to obtain the psd

end % of myPyulear( x, order, nFFT )

function [A, E, K] = myLevinson( R, N )
% A = myLevinson( R, N ) solves the Hermitian Toeplitz system of equations
% 
%     [  R(1)   R(2)* ...  R(N)* ] [  A(2)  ]  = [  -R(2)  ]
%     [  R(2)   R(1)  ... R(N-1)*] [  A(3)  ]  = [  -R(3)  ]
%     [   .        .         .   ] [   .    ]  = [    .    ]
%     [ R(N-1) R(N-2) ...  R(2)* ] [  A(N)  ]  = [  -R(N)  ]
%     [  R(N)  R(N-1) ...  R(1)  ] [ A(N+1) ]  = [ -R(N+1) ]
% 
% (also known as the Yule-Walker AR equations) using the Levinson-
% Durbin recursion.  Input R is typically a vector of autocorrelation
% coefficients with lag 0 as the first element.
% 
% N is the order of the recursion; if omitted, N = LENGTH(R)-1.
% A will be a row vector of length N+1, with A(1) = 1.0.
% 
% [A, E] = myLevinson(...) returns the prediction error, E, of order N.
% 
% [A, E, K] = myLevinson(...) returns the reflection coefficients K as a
% column vector of length N.
% 
% If R is a matrix, levinson finds coefficients for each column of R,
% and returns them in the rows of A

    assert( isvector( R ) && (numel( R ) > N), 'R must be a vector of size greater than N' );

    R = R(:)';
    K = zeros( 1, N );
    A = zeros( 1, N+1 );
    A(1) = 1;
    prevA = A;
    
    E = R(1);
    
    for i = 1:N,
        
        s = 0;
        for j = 1:i-1,
            s = s + prevA(j+1) * R(i-j+1);
        end % of j loop
        
        ki = ( R(i+1) - s ) / E;
        K(i) = -ki;
        A(i+1) = ki;
        
        if( i > 1 )
            for j = 1:i-1,
                A(j+1) = prevA(j+1) - ki*prevA(i-j+1);
            end % of j-loop
        end
        
        E = (1 - ki*ki) * E;

        prevA = A;
    end
    
    A(2:end) = -A(2:end);
    
end % of myLevinson()