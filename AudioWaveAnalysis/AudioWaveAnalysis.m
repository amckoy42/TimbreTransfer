clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 16;

% Read and plot signal.
[y, fs] = audioread('organfinale.wav');
audioMono = (y(:, 1) + y(:, 2)) / 2;
subplot(2, 2, 1);
plot(y, 'b-');
grid on;
title('Audio Waveform', 'FontSize', fontSize);
xlabel('Index', 'FontSize', fontSize);
ylabel('Signal Amplitude', 'FontSize', fontSize);

% Compute and plot spectrogram
subplot(2, 2, 2);
spectrogram(audioMono);
title('Spectrogram', 'FontSize', fontSize);

% Compute and plot power.
audioPower = pwelch(y);
subplot(2, 2, 3);
plot(audioPower, 'b-');
grid on;
xlim([0, 5000]);
title('P Welch Power', 'FontSize', fontSize);
xlabel('Frequency', 'FontSize', fontSize);
ylabel('Power', 'FontSize', fontSize);

% Compute and plot power.
pxx = periodogram(y);
subplot(2, 2, 4);
plot(pxx, 'b-');
grid on;
xlim([0, 5000]);
title('Periodogram', 'FontSize', fontSize);
xlabel('Frequency', 'FontSize', fontSize);
ylabel('Power', 'FontSize', fontSize);