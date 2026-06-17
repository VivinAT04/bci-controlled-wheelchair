%% BCI-Controlled Wheelchair Simulation (Animated)
% Reads classifier predictions from results/predicted_commands.csv
% (produced by scripts/export_predictions.py) and animates an
% intelligent wheelchair moving step-by-step on a 2D grid, with safety
% logic:
%   - Confidence threshold: low-confidence predictions are overridden
%     to "Stop", since acting on an uncertain EEG classification is
%     unsafe for a real assistive device.
%   - Obstacle detection: the chair will not move into a wall cell.
%   - Emergency stop: any "Stop" command halts movement immediately
%     regardless of position.
%
% This script locates the CSV relative to its own file location, so it
% works correctly regardless of MATLAB's current working folder.

clear; clc; close all;

%% --- Configuration ---
CONFIDENCE_THRESHOLD = 0.5;   % below this, override to Stop (safety rule)
GRID_SIZE = 20;                % 20x20 grid world
START_POS = [10, 10];          % starting [row, col]
START_HEADING = 0;             % 0=North, 90=East, 180=South, 270=West
ANIMATION_PAUSE = 0.05;        % seconds between frames (lower = faster)
MAX_FRAMES = 150;              % cap animation length so it doesn't run too long

% Simple obstacle layout: a wall segment to demonstrate collision logic
obstacles = false(GRID_SIZE, GRID_SIZE);
obstacles(5, 8:14) = true;     % a wall the chair must not cross

%% --- Load predicted commands from Python classifier output ---
script_dir = fileparts(mfilename('fullpath'));
csv_path = fullfile(script_dir, 'results', 'predicted_commands.csv');

if ~isfile(csv_path)
    error('Cannot find %s. Run scripts/export_predictions.py first.', csv_path);
end

T = readtable(csv_path);
T.command = string(T.command);
T.true_class = string(T.true_class);
T.predicted_class = string(T.predicted_class);

fprintf('Loaded %d predicted commands from %s\n', height(T), csv_path);
n_steps = min(height(T), MAX_FRAMES);
fprintf('Animating first %d steps (set MAX_FRAMES to change this).\n', n_steps);

%% --- Set up the figure once, before the loop ---
fig = figure('Name', 'BCI Wheelchair Simulation (Live)', 'Color', 'w');
hold on;
[obs_r, obs_c] = find(obstacles);
scatter(obs_c, obs_r, 200, 'k', 's', 'filled', 'DisplayName', 'Obstacle');
plot(START_POS(2), START_POS(1), 'g^', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'g', 'DisplayName', 'Start');

% The trail (path so far) and the chair's current marker are separate
% graphics objects we update every frame, rather than redrawing the
% whole plot from scratch each time.
trail = plot(nan, nan, '-', 'Color', [0.2 0.4 0.8], 'LineWidth', 1.2, ...
    'DisplayName', 'Path so far');
chair_marker = plot(START_POS(2), START_POS(1), 'o', 'MarkerSize', 14, ...
    'MarkerFaceColor', [0.9 0.3 0.2], 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'DisplayName', 'Wheelchair');

xlim([0 GRID_SIZE+1]); ylim([0 GRID_SIZE+1]);
set(gca, 'YDir', 'reverse');
axis equal; grid on;
xlabel('Column'); ylabel('Row');
legend('Location', 'best');
status_text = title('Step 0 / 0 - Command: (starting)');

%% --- Simulate + animate, one step per loop iteration ---
pos = START_POS;
heading = START_HEADING;
chair_path = pos;
n_overridden = 0;
n_blocked = 0;
n_emergency_stop = 0;

for i = 1:n_steps
    if ~isvalid(fig)
        fprintf('Figure closed early, stopping animation.\n');
        break
    end

    command = T.command(i);
    confidence = T.confidence(i);

    if confidence < CONFIDENCE_THRESHOLD
        command = "Stop";
        n_overridden = n_overridden + 1;
    end

    if command == "Stop"
        n_emergency_stop = n_emergency_stop + 1;
        % No movement this step, but still redraw so the title updates
        % and the viewer can see the chair held still.
    else
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

        if is_valid_move(next_pos, GRID_SIZE, obstacles)
            pos = next_pos;
        else
            n_blocked = n_blocked + 1;
        end
    end

    chair_path = [chair_path; pos]; %#ok<AGROW>

    % --- Update the existing graphics objects (this is the animation) ---
    set(trail, 'XData', chair_path(:,2), 'YData', chair_path(:,1));
    set(chair_marker, 'XData', pos(2), 'YData', pos(1));
    set(status_text, 'String', ...
        sprintf('Step %d / %d  -  Command: %s  (confidence %.2f)', ...
        i, n_steps, command, confidence));
    drawnow;
    pause(ANIMATION_PAUSE);
end

%% --- Report ---
fprintf('\n--- Simulation summary ---\n');
fprintf('Steps animated: %d\n', i);
fprintf('Overridden to Stop (low confidence < %.2f): %d\n', CONFIDENCE_THRESHOLD, n_overridden);
fprintf('Explicit Stop commands executed: %d\n', n_emergency_stop);
fprintf('Moves blocked by obstacle/boundary: %d\n', n_blocked);
fprintf('Final position: [%d, %d], heading %d deg\n', pos(1), pos(2), heading);

%% --- Local functions ---
function v = step_vector(heading_deg)
    switch heading_deg
        case 0,   v = [-1, 0];  % North = up
        case 90,  v = [0, 1];   % East = right
        case 180, v = [1, 0];   % South = down
        case 270, v = [0, -1];  % West = left
        otherwise, v = [0, 0];
    end
end

function ok = is_valid_move(pos, grid_size, obstacles)
    if any(pos < 1) || any(pos > grid_size)
        ok = false;
    elseif obstacles(pos(1), pos(2))
        ok = false;
    else
        ok = true;
    end
end