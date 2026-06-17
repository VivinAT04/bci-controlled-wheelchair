%% BCI-Controlled Wheelchair Simulation
% Reads classifier predictions from results/predicted_commands.csv
% (produced by scripts/export_predictions.py) and simulates an
% intelligent wheelchair moving on a 2D grid, with safety logic:
%   - Confidence threshold: low-confidence predictions are overridden
%     to "Stop", since acting on an uncertain EEG classification is
%     unsafe for a real assistive device.
%   - Obstacle detection: the chair will not move into a wall cell.
%   - Emergency stop: any "Stop" command halts movement immediately
%     regardless of position.
%
% Run this script from MATLAB with the working folder set to the
% project root (so the relative path to the CSV resolves correctly).

clear; clc; close all;

%% --- Configuration ---
CONFIDENCE_THRESHOLD = 0.5;   % below this, override to Stop (safety rule)
GRID_SIZE = 20;                % 20x20 grid world
START_POS = [10, 10];          % starting [row, col]
START_HEADING = 0;             % 0=North, 90=East, 180=South, 270=West

% Simple obstacle layout: a wall segment to demonstrate collision logic
obstacles = false(GRID_SIZE, GRID_SIZE);
obstacles(5, 8:14) = true;     % a wall the chair must not cross

%% --- Load predicted commands from Python classifier output ---
csv_path = fullfile('results', 'predicted_commands.csv');
if ~isfile(csv_path)
    error('Cannot find %s. Run scripts/export_predictions.py first.', csv_path);
end

T = readtable(csv_path);
% Force text columns to MATLAB string arrays regardless of how readtable
% imported them (cell array of char vs string array varies by MATLAB
% version / import settings) - this avoids brace-indexing errors below.
T.command = string(T.command);
T.true_class = string(T.true_class);
T.predicted_class = string(T.predicted_class);

fprintf('Loaded %d predicted commands from %s\n', height(T), csv_path);

%% --- Simulate ---
pos = START_POS;
heading = START_HEADING;
chair_path = pos;                % record every visited cell (renamed from
                                  % "path" to avoid shadowing MATLAB's
                                  % built-in path() function)
n_overridden = 0;
n_blocked = 0;
n_emergency_stop = 0;

for i = 1:height(T)
    command = T.command(i);          % scalar string, e.g. "Move forward"
    confidence = T.confidence(i);

    % Safety rule 1: low-confidence prediction -> force Stop
    if confidence < CONFIDENCE_THRESHOLD
        command = "Stop";
        n_overridden = n_overridden + 1;
    end

    % Safety rule 2: emergency stop on explicit Stop command
    if command == "Stop"
        n_emergency_stop = n_emergency_stop + 1;
        chair_path = [chair_path; pos]; %#ok<AGROW>
        continue
    end

    % Compute proposed next position based on command + heading
    switch command
        case "Move forward"
            next_pos = pos + step_vector(heading);
        case "Turn left"
            heading = mod(heading - 90, 360);
            next_pos = pos;
        case "Turn right"
            heading = mod(heading + 90, 360);
            next_pos = pos;
        otherwise
            next_pos = pos;
    end

    % Safety rule 3: obstacle / boundary check before moving
    if is_valid_move(next_pos, GRID_SIZE, obstacles)
        pos = next_pos;
    else
        n_blocked = n_blocked + 1;
    end

    chair_path = [chair_path; pos]; %#ok<AGROW>
end

%% --- Report ---
fprintf('\n--- Simulation summary ---\n');
fprintf('Total commands processed: %d\n', height(T));
fprintf('Overridden to Stop (low confidence < %.2f): %d\n', CONFIDENCE_THRESHOLD, n_overridden);
fprintf('Explicit Stop commands executed: %d\n', n_emergency_stop);
fprintf('Moves blocked by obstacle/boundary: %d\n', n_blocked);
fprintf('Final position: [%d, %d], heading %d deg\n', pos(1), pos(2), heading);

%% --- Visualize ---
figure('Name', 'BCI Wheelchair Simulation', 'Color', 'w');
hold on;
[obs_r, obs_c] = find(obstacles);
scatter(obs_c, obs_r, 200, 'k', 's', 'filled', 'DisplayName', 'Obstacle');
plot(chair_path(:,2), chair_path(:,1), '-o', 'Color', [0.2 0.4 0.8], ...
    'MarkerSize', 3, 'DisplayName', 'Wheelchair path');
plot(START_POS(2), START_POS(1), 'g^', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'g', 'DisplayName', 'Start');
plot(pos(2), pos(1), 'r*', 'MarkerSize', 14, 'DisplayName', 'End');
xlim([0 GRID_SIZE+1]); ylim([0 GRID_SIZE+1]);
set(gca, 'YDir', 'reverse'); % row increases downward, like a grid map
axis equal; grid on;
xlabel('Column'); ylabel('Row');
title('Simulated wheelchair path from BCI classifier commands');
legend('Location', 'best');

%% --- Local functions ---
function v = step_vector(heading_deg)
    % Returns a [row, col] unit step for the given heading.
    switch heading_deg
        case 0,   v = [-1, 0];  % North = up
        case 90,  v = [0, 1];   % East = right
        case 180, v = [1, 0];   % South = down
        case 270, v = [0, -1];  % West = left
        otherwise, v = [0, 0];
    end
end

function ok = is_valid_move(pos, grid_size, obstacles)
    % Checks grid boundaries and obstacle collision.
    if any(pos < 1) || any(pos > grid_size)
        ok = false;
    elseif obstacles(pos(1), pos(2))
        ok = false;
    else
        ok = true;
    end
end