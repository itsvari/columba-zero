# Bound capture without cancelling export

`tools/run_pi_validation.py` supervises one application process. It owns no camera
or I²C connection. The child application remains the sole owner of its camera and
pan-tilt bus. The supervisor starts no telemetry sidecar automatically.

Run from the repository root, using the same Python environment as the application:

```sh
python tools/run_pi_validation.py \
    --config /path/to/validated-profile.yaml \
    --duration-s 270 \
    --stop-file /tmp/monitor-campaign-unique-id.stop \
    --output /path/to/new-supervisor-directory \
    --hardware
```

`--hardware` is required when the selected profile enables following in hardware
mode. Simulation and profiles without hardware following do not require it. The
flag does not modify the profile. `AppConfig.load` validates the exact saved
configuration before the child starts. The command contains no shell:

```text
<current Python> -u -m app.main --config <supervisor directory>/config.yaml
```

`python -m tools.run_pi_validation` accepts the same options. Direct script
invocation does not require an editable installation or `PYTHONPATH` override.

The supervisor copies the selected configuration without changing its settings.
Paths inside it retain the application's working-directory semantics. Application
recordings still go to the configured `output.directory`; `--output` stores the
supervisor's evidence. It must be a new directory with an existing parent. The
stop-file must be absent, including any symlink, and have an existing parent or
be directly inside the new supervisor directory. It cannot name a supervisor
output file. Choose a fresh stop-file for every invocation. Input or configuration
failure can leave the new directory with its configuration snapshot, but starts
no child.

After the playlist and host controls finish, request a stop by creating that
invocation's unique file:

```sh
touch /tmp/monitor-campaign-unique-id.stop
```

The host may perform this step through its existing SSH connection. Keep the
supervisor alive until it finishes. Do not add a separate timer that sends TERM
to the application, and do not send a second TERM to stop export. The supervisor
also accepts SIGINT or SIGTERM itself, including repeated signals.

The first observed stop trigger is a unique-file request, a supervisor signal,
or the capture deadline. If several are pending in one poll, the signal takes
precedence, then the file, then the deadline. The supervisor sends the child TERM
once. Subsequent requests only contribute to its recorded signal count. It polls
every 0.1 seconds, so deadline enforcement has polling and operating-system
scheduling delay. The capture budget starts immediately before launch and
includes application startup. A 270-second budget leaves 69 seconds around a
201-second playlist for startup and host controls; it does not synchronise the
playlist to camera exposure.

After TERM, the supervisor gives the child a separate deadline of
`preview.export_timeout_s + 30` seconds from that request. Cleanup and export
share this deadline. Crossing the original capture deadline during export sends
no further TERM. If the child exceeds the cleanup deadline, the supervisor sends
KILL to the child's process group and allows at most five further seconds to reap
the child. A separate process session prevents signals sent to the supervisor's
terminal group from reaching the child directly. Supervisor exceptions use the
same one-TERM, bounded-cleanup path. A child that exits on its own needs no TERM.

The supervisor does not infer application phases from console text. A child that
starts exporting independently before a supervisor stop request may receive the
first TERM during its own export. Configure the campaign so the supervisor owns
the capture stop. SIGKILL of the supervisor, power loss and an unschedulable
operating system cannot execute its cleanup path.

The output directory retains these files:

| File | Contents |
| --- | --- |
| `config.yaml` | Exact selected configuration bytes used by the child. |
| `console.log` | Combined child stdout and stderr through process exit, including export. |
| `manifest.json` | Command, working directory, source and configuration hashes, supervisor and child PIDs, Unix and monotonic times, capture and export deadlines, stop reason, TERM count, supervisor signal count, first signal, forced-KILL status, child exit status and any supervisor exception. |

The manifest is replaced atomically when the child starts, when stop is requested,
and when supervision finishes. Its source SHA-256 covers sorted `app/**/*.py`
paths and their content hashes, followed by `tools/run_pi_validation.py`. It does
not hash model weights, calibration, dependencies or external files named by the
configuration. Preserve those campaign inputs separately. `term_sent_count`
records attempts, including a child-exit race where the process has already gone.
Exit status 0 means the child exited successfully. Status 1 means the child failed
or required KILL. Invalid input and I/O failures report status 2. An unexpected
supervisor exception propagates after bounded cleanup.

To collect Pi health, launch the existing read-only sidecar separately after
`manifest.json` has a non-null `child_pid`:

```sh
python -m tools.pi_telemetry --process-id CHILD_PID \
    --output /path/to/new-telemetry.jsonl --duration-s 1300
```

The sidecar can cover export as well as capture. Choose its duration for the
capture budget plus the configured export timeout and grace.

## Verify without a camera or bus

These commands were run locally against a fake application subprocess. They open
no camera, I²C bus, SSH connection or browser:

```sh
python tools/run_pi_validation.py --help
python -m pytest tests/test_pi_validation_runner.py
python -m ruff check tools/run_pi_validation.py tests/test_pi_validation_runner.py
python -m ruff format --check tools/run_pi_validation.py tests/test_pi_validation_runner.py
python -m mypy --strict tools/run_pi_validation.py tests/test_pi_validation_runner.py
```

Use the project's environment with its development dependencies installed. The
regression first demonstrates a fake export cancelling with status 3 after two
TERMs. The supervised case receives a stop-file request, repeated supervisor
signals and the capture deadline during export, but the child observes one TERM
and completes with status 0. Further fixtures check cleanup-deadline KILL,
supervisor-exception cleanup, natural child exit, hardware opt-in and invalid
inputs before launch. These fixtures establish local process-control behaviour;
they do not establish Pi movement, timing, output-mode or endurance acceptance.
