Fs = 4.41E4;                                % Sampling Frequency (Hz)
tmax = 5;                                   % Time Duration Of Signal (sec)
t = linspace(0, tmax, tmax*Fs);             % Time Vector
f = 220;                                    % Original Frequency
s = sin(2*pi*f*t + cos(15*pi*t)*pi/2);      % Original Signal
sound(s, Fs)                                % Listen To Original Signal
pause(tmax*2)                               % Wait For It To Finish
Fc = 261.6;                                 % Desired Output Frequency   
carrier = sin(2*pi*(Fc-f)*t);               % Generate Carrier 
sm = s .* carrier;                          % Modulate (Produces Upper & Lower Sidebands
Fn = Fs/2;                                  % Design High-Pass Filter To Eliminate Lower Sideband
Wp = Fc/Fn;
Ws = Wp*0.8;
[n,Wn] = buttord(Wp, Ws, 1, 10);
[b,a] = butter(n,Wn,'high');
[sos,g] = tf2sos(b,a);
smf = 2*filtfilt(sos,g,sm);                 % Output Is Suppressed-Carrier Upper Sideband Signal
sound(smf, Fs)                              % Listen To New Signal
Fn1 = Fs/2;                                 % Compute & Plot Fourier Series Of Both
Ft1 = fft(s)/length(s);
Fv1 = linspace(0, 1, fix(length(Ft1)/2)+1)*Fn1;
Ix1 = 1:length(Fv1);
Fn2 = Fs/2;
Ft2 = fft(smf)/length(smf);
Fv2 = linspace(0, 1, fix(length(Ft2)/2)+1)*Fn2;
Ix2 = 1:length(Fv2);
figure(1)
plot(Fv1, abs(Ft1(Ix1)))
grid
hold on
plot(Fv2, abs(Ft2(Ix2)))
hold off
axis([0  500   ylim])