"""Module for interfacing with the Bpod Finite State Machine."""

import logging
import re
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from pydantic import validate_call
from serial import SerialException
from serial.tools.list_ports import comports

from bpod_core import __version__ as bpod_core_version
from bpod_core.com import ExtendedSerial
from bpod_core.fsm import StateMachine

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer  # noqa: F401

PROJECT_NAME = 'bpod-core'
VENDOR_IDS_BPOD = [0x16C0]  # vendor IDs of supported Bpod devices
MIN_BPOD_FW_VERSION = (23, 0)  # minimum supported firmware version (major, minor)
MIN_BPOD_HW_VERSION = 3  # minimum supported hardware version
MAX_BPOD_HW_VERSION = 4  # maximum supported hardware version
CHANNEL_TYPES_INPUT = {
    b'U': 'Serial',
    b'X': 'SoftCode',
    b'Z': 'SoftCodeApp',
    b'F': 'FlexIO',
    b'D': 'Digital',
    b'B': 'BNC',
    b'W': 'Wire',
    b'P': 'Port',
}
CHANNEL_TYPES_OUTPUT = CHANNEL_TYPES_INPUT.copy()
CHANNEL_TYPES_OUTPUT.update({b'V': 'Valve', b'P': 'PWM'})
N_SERIAL_EVENTS_DEFAULT = 15
VALID_OPERATORS = ['exit', '>exit', '>back']

logger = logging.getLogger(__name__)


class VersionInfo(NamedTuple):
    """Represents the Bpod's on-board hardware configuration."""

    firmware: tuple[int, int]
    """Firmware version (major, minor)"""
    machine: int
    """Machine type (numerical)"""
    pcb: int | None
    """PCB revision, if applicable"""


class HardwareConfiguration(NamedTuple):
    """Represents the Bpod's on-board hardware configuration."""

    max_states: int
    """Maximum number of supported states in a single state machine description."""
    cycle_period: int
    """Period of the state machine's refresh cycle during a trial in microseconds."""
    max_serial_events: int
    """Maximum number of behavior events allocatable among connected modules."""
    max_bytes_per_serial_message: int
    """Maximum number of bytes allowed per serial message."""
    n_global_timers: int
    """Number of global timers supported."""
    n_global_counters: int
    """Number of global counters supported."""
    n_conditions: int
    """Number of condition-events supported."""
    n_inputs: int
    """Number of input channels."""
    input_description: bytes
    """Array indicating the state machine's onboard input channel types."""
    n_outputs: int
    """Number of channels in the state machine's output channel description array."""
    output_description: bytes
    """Array indicating the state machine's onboard output channel types."""

    @property
    def cycle_frequency(self) -> int:
        """Frequency of the state machine's refresh cycle during a trial in Hertz."""
        return 1000000 // self.cycle_period

    @property
    def n_modules(self) -> int:
        """Number of modules supported by the state machine."""
        return self.input_description.count(b'U')


class BpodError(Exception):
    """
    Exception class for Bpod-related errors.

    This exception is raised when an error specific to the Bpod device or its
    operations occurs.
    """


class Bpod:
    """Bpod class for interfacing with the Bpod Finite State Machine."""

    _version: VersionInfo
    _hardware: HardwareConfiguration
    serial0: ExtendedSerial
    """Primary serial device for communication with the Bpod."""
    serial1: ExtendedSerial | None = None
    """Secondary serial device for communication with the Bpod."""
    serial2: ExtendedSerial | None = None
    """Tertiary serial device for communication with the Bpod - used by Bpod 2+ only."""
    inputs: NamedTuple
    """Available input channels."""
    outputs: NamedTuple
    """Available output channels."""
    modules: NamedTuple
    """Available modules."""
    event_names: list[str]
    """List of event names."""
    output_actions: list[str]
    """List of output actions."""

    @validate_call
    def __init__(self, port: str | None = None, serial_number: str | None = None):
        logger.info(f'bpod_core {bpod_core_version}')

        # initialize members
        self.event_names = []
        self.output_actions = []

        # identify Bpod by port or serial number
        port, self._serial_number = self._identify_bpod(port, serial_number)

        # open primary serial port
        self.serial0 = ExtendedSerial()
        self.serial0.port = port
        self.open()

        # get firmware version and machine type; enforce version requirements
        self._get_version_info()

        # get the Bpod's onboard hardware configuration
        self._get_hardware_configuration()

        # configure input and output channels
        self._configure_io()

        # detect additional serial ports
        self._detect_additional_serial_ports()

        # update modules
        self.update_modules()

        # log hardware information
        machine = {3: 'r2.0-2.5', 4: '2+ r1.0'}.get(self.version.machine, 'unknown')
        logger.info(f'Connected to Bpod Finite State Machine {machine} on {self.port}')
        logger.info(
            f'Firmware Version {"{}.{}".format(*self.version.firmware)}, '
            f'Serial Number {self._serial_number}, PCB Revision {self.version.pcb}'
        )

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, type, value, traceback):
        """Exit context and close connection."""
        self.close()

    def __del__(self):
        self.close()

    def _sends_discovery_byte(
        self,
        port: str,
        byte: bytes = b'\xde',
        timeout: float = 0.11,
        trigger: bytes | None = None,
    ) -> bool:
        r"""Check if the device on the given port sends a discovery byte.

        Parameters
        ----------
        port : str
            The name of the serial port to check (e.g., '/dev/ttyUSB0' or 'COM3').
        byte : bytes, optional
            The discovery byte to expect from the device. Defaults to b'\\xde'.
        timeout : float, optional
            Timeout period (in seconds) for the serial read operation. Defaults to 0.11.
        trigger : bytes, optional
            An optional command to send on serial0 before reading from the given device.

        Returns
        -------
        bool
            Whether the given device responded with the expected discovery byte or not.
        """
        try:
            with ExtendedSerial(port, timeout=timeout) as ser:
                if trigger is not None and getattr(self, 'serial0', None) is not None:
                    self.serial0.write(trigger)
                return ser.read(1) == byte
        except SerialException:
            return False

    def _identify_bpod(
        self, port: str | None = None, serial_number: str | None = None
    ) -> tuple[str, str | None]:
        """
        Try to identify a supported Bpod based on port or serial number.

        If neither port nor serial number are provided, this function will attempt to
        detect a supported Bpod automatically.

        Parameters
        ----------
        port : str | None, optional
            The port of the device.
        serial_number : str | None, optional
            The serial number of the device.

        Returns
        -------
        str
            the port of the device
        str | None
            the serial number of the device

        Raises
        ------
        BpodError
            If no Bpod is found or the indicated device is not supported.
        """
        # If no port or serial number provided, try to automagically find an idle Bpod
        if port is None and serial_number is None:
            try:
                port_info = next(
                    p
                    for p in comports()
                    if getattr(p, 'vid', None) in VENDOR_IDS_BPOD
                    and self._sends_discovery_byte(p.device)
                )
            except StopIteration as e:
                raise BpodError('No available Bpod found') from e
            return port_info.device, port_info.serial_number

        # Else, if a serial number was provided, try to match it with a serial device
        elif serial_number is not None:
            try:
                port_info = next(
                    p
                    for p in comports()
                    if p.serial_number == serial_number
                    and self._sends_discovery_byte(p.device)
                )
            except (StopIteration, AttributeError) as e:
                raise BpodError(f'No device with serial number {serial_number}') from e

        # Else, assure that the provided port exists and the device could be a Bpod
        else:
            try:
                port_info = next(p for p in comports() if p.device == port)
            except (StopIteration, AttributeError) as e:
                raise BpodError(f'Port not found: {port}') from e

        if port_info.vid not in VENDOR_IDS_BPOD:
            raise BpodError('Device is not a supported Bpod')
        return port_info.device, port_info.serial_number

    def _get_version_info(self) -> None:
        """
        Retrieve firmware version and machine type information from the Bpod.

        This method queries the Bpod to obtain its firmware version, machine type, and
        PCB revision. It also validates that the hardware and firmware versions meet
        the minimum requirements. If the versions are not supported, an Exception is
        raised.

        Raises
        ------
        BpodError
            If the hardware version or firmware version is not supported.
        """
        logger.debug('Retrieving version information')
        v_major, machine_type = self.serial0.query_struct(b'F', '<2H')
        v_minor = self.serial0.query_struct(b'f', '<H')[0] if v_major > 22 else 0
        v_firmware = (v_major, v_minor)
        if not (MIN_BPOD_HW_VERSION <= machine_type <= MAX_BPOD_HW_VERSION):
            raise BpodError(
                f'The hardware version of the Bpod on {self.port} is not supported.'
            )
        if v_firmware < MIN_BPOD_FW_VERSION:
            raise BpodError(
                f'The Bpod on {self.port} uses firmware v{v_major}.{v_minor} '
                f'which is not supported. Please update the device to '
                f'firmware v{MIN_BPOD_FW_VERSION[0]}.{MIN_BPOD_FW_VERSION[1]} or later.'
            )
        pcv_rev = self.serial0.query_struct(b'v', '<B')[0] if v_major > 22 else None
        self._version = VersionInfo(v_firmware, machine_type, pcv_rev)

    def _get_hardware_configuration(self) -> None:
        """Retrieve the Bpod's onboard hardware configuration."""
        logger.debug('Retrieving onboard hardware configuration')
        if self.version.firmware > (22, 0):
            hardware_conf = list(self.serial0.query_struct(b'H', '<2H6B'))
        else:
            hardware_conf = list(self.serial0.query_struct(b'H', '<2H5B'))
            hardware_conf.insert(-4, 3)  # max bytes per serial msg always = 3
        hardware_conf.extend(self.serial0.read_struct(f'<{hardware_conf[-1]}s1B'))
        hardware_conf.extend(self.serial0.read_struct(f'<{hardware_conf[-1]}s'))
        self._hardware = HardwareConfiguration(*hardware_conf)

    def _configure_io(self) -> None:
        """Configure the input and output channels of the Bpod."""
        logger.debug('Configuring I/O')
        for description, channel_class, channel_names in (
            (self._hardware.input_description, Input, CHANNEL_TYPES_INPUT),
            (self._hardware.output_description, Output, CHANNEL_TYPES_OUTPUT),
        ):
            n_channels = len(description)
            io_class = f'{channel_class.__name__.lower()}s'
            channels = []
            types = []

            # loop over the description array and create channels
            for idx, io_key in enumerate(struct.unpack(f'<{n_channels}c', description)):
                if io_key not in channel_names:
                    raise RuntimeError(f'Unknown {io_class[:-1]} type: {io_key}')
                n = description[:idx].count(io_key) + 1
                name = f'{channel_names[io_key]}{n}'
                channels.append(channel_class(self, name, io_key, idx))
                types.append((name, channel_class))

            # store channels to NamedTuple and set the latter as a class attribute
            named_tuple = NamedTuple(io_class, types)._make(channels)
            setattr(self, io_class, named_tuple)

        # set the enabled state of the input channels
        self._set_enable_inputs()

    def _detect_additional_serial_ports(self) -> None:
        """Detect additional USB-serial ports."""
        logger.debug('Detecting additional USB-serial ports')

        # First, assemble a list of candidate ports
        candidate_ports = [
            p.device
            for p in comports()
            if p.serial_number == self._serial_number and p.device != self.port
        ]

        # Exclude those devices from the list that are already sending a discovery byte
        # NB: this should not be necessary, as we already filter for devices with
        #     identical USB serial number.
        # for port in candidate_ports:
        #     if self._sends_discovery_byte(port):
        #         candidate_ports.remove(port)

        # Find secondary USB-serial port
        if self._version.firmware >= (23, 0):
            for port in candidate_ports:
                if self._sends_discovery_byte(port, bytes([222]), trigger=b'{'):
                    self.serial1 = ExtendedSerial()
                    self.serial1.port = port
                    candidate_ports.remove(port)
                    logger.debug(f'Detected secondary USB-serial port: {port}')
                    break
            if self.serial1 is None:
                raise BpodError('Could not detect secondary serial port')

        # State Machine 2+ uses a third USB-serial port for FlexIO
        if self.version.machine == 4:
            for port in candidate_ports:
                if self._sends_discovery_byte(port, bytes([223]), trigger=b'}'):
                    self.serial2 = ExtendedSerial()
                    self.serial2.port = port
                    logger.debug(f'Detected tertiary USB-serial port: {port}')
                    break
            if self.serial2 is None:
                raise BpodError('Could not detect tertiary serial port')

    def _handshake(self):
        """
        Perform a handshake with the Bpod.

        Raises
        ------
        BpodException
            If the handshake fails.
        """
        try:
            self.serial0.timeout = 0.2
            if not self.serial0.verify(b'6', b'5'):
                raise BpodError(f'Handshake with device on {self.port} failed')
            self.serial0.timeout = None
        except SerialException as e:
            raise BpodError(f'Handshake with device on {self.port} failed') from e
        finally:
            self.serial0.reset_input_buffer()
        logger.debug(f'Handshake with Bpod on {self.port} successful')

    def _test_psram(self) -> bool:
        """
        Test the Bpod's PSRAM.

        Returns
        -------
        bool
            True if the PSRAM test passed, False otherwise.
        """
        return self.serial0.verify(b'_')

    def _set_enable_inputs(self) -> bool:
        logger.debug('Updating enabled state of input channels')
        enable = [i.enabled for i in self.inputs]
        self.serial0.write_struct(f'<c{self._hardware.n_inputs}?', b'E', *enable)
        return self.serial0.read(1) == b'\x01'

    def _reset_session_clock(self) -> bool:
        logger.debug('Resetting session clock')
        return self.serial0.verify(b'*')

    def _disable_all_module_relays(self):
        for module in self.modules:
            module.set_relay(False)

    def _compile_event_names(self):
        n_serial_events = sum([len(m.event_names) for m in self.modules])
        n_softcodes = self._hardware.max_serial_events - n_serial_events
        n_usb = self._hardware.input_description.count(b'X')
        n_usb_ext = self._hardware.input_description.count(b'Z')
        n_softcodes_per_usb_channel = n_softcodes // (n_usb + n_usb_ext)
        n_app_softcodes = n_usb_ext * n_softcodes_per_usb_channel

        event_names = []
        module_idx = 0
        flexio_idx = 0
        port_idx = 0
        bnc_idx = 0
        wire_idx = 0
        for input in self.inputs:
            match input.io_type:
                case b'U':  # Serial / modules
                    n = self.modules[module_idx].event_names
                    module_idx += 1
                case b'X':  # SoftCode
                    n = (f'SoftCode{i + 1}' for i in range(n_softcodes_per_usb_channel))
                case b'Z':
                    n = (f'APP_SoftCode{i + 1}' for i in range(n_app_softcodes))
                case b'F':
                    n = (f'Flex{flexio_idx}_{i + 1}' for i in range(2))
                    flexio_idx += 1
                case b'P':
                    n = (f'Port{port_idx}_{state}' for state in ('High', 'Low'))
                    port_idx += 1
                case b'B':
                    n = (f'BNC{bnc_idx}_{state}' for state in ('High', 'Low'))
                    bnc_idx += 1
                case b'W':
                    n = (f'Wire{bnc_idx}_{state}' for state in ('High', 'Low'))
                    wire_idx += 1
                case _:
                    n = []
            event_names.extend(n)

        event_names.extend(
            f'GlobalTimer{i + 1}_Start' for i in range(self._hardware.n_global_timers)
        )
        event_names.extend(
            f'GlobalTimer{i + 1}_End' for i in range(self._hardware.n_global_timers)
        )
        event_names.extend(
            f'GlobalCounter{i + 1}_End' for i in range(self._hardware.n_global_counters)
        )
        event_names.extend(
            f'Condition{i + 1}' for i in range(self._hardware.n_conditions)
        )
        event_names.append('Tup')

        output_actions = []
        module_idx = 0
        flexio_idx = 0
        valve_idx = 0
        pwm_idx = 0
        bnc_idx = 0
        wire_idx = 0
        for output in self.outputs:
            match output.io_type:
                case b'U':
                    n = self.modules[module_idx].name
                    module_idx += 1
                case b'X':
                    n = 'SoftCode'
                case b'Z':
                    n = 'APP_SoftCode'
                case b'F':
                    n = f'Flex{flexio_idx + 1}'
                    flexio_idx += 1
                case b'V':
                    n = f'Valve{valve_idx + 1}'
                    valve_idx += 1
                case b'P':
                    n = f'PWM{pwm_idx + 1}'
                    pwm_idx += 1
                case b'B':
                    n = f'BNC{bnc_idx + 1}'
                    bnc_idx += 1
                case b'W':
                    n = f'Wire{bnc_idx + 1}'
                    wire_idx += 1
                case _:
                    n = ()
            output_actions.append(n)
        output_actions.extend(
            ['GlobalTimerTrig', 'GlobalTimerCancel', 'GlobalCounterReset']
        )
        if self.version.machine == 4:
            output_actions.extend(['AnalogThreshEnable', 'AnalogThreshDisable'])

        self.event_names = event_names
        self.output_actions = output_actions

    @property
    def port(self) -> str | None:
        """The port of the Bpod's primary serial device."""
        return self.serial0.port

    @property
    def version(self) -> VersionInfo:
        """Version information of the Bpod's firmware and hardware."""
        return self._version

    def open(self):
        """
        Open the connection to the Bpod.

        Raises
        ------
        SerialException
            If the port could not be opened.
        BpodException
            If the handshake fails.
        """
        if self.serial0.is_open:
            return
        self.serial0.open()
        self._handshake()

    def close(self):
        """Close the connection to the Bpod."""
        if hasattr(self, 'serial0') and self.serial0.is_open:
            self.serial0.write(b'Z')
            self.serial0.close()

    def set_status_led(self, enabled: bool) -> bool:
        """
        Enable or disable the Bpod's status LED.

        Parameters
        ----------
        enabled : bool
            True to enable the status LED, False to disable.

        Returns
        -------
        bool
            True if the operation was successful, False otherwise.
        """
        self.serial0.write_struct('<c?', b':', enabled)
        return self.serial0.verify(b'')

    def update_modules(self):
        """Update the list of connected modules and their configurations."""
        # self._disable_all_module_relays()
        self.serial0.write(b'M')
        modules = []
        for idx in range(self._hardware.n_modules):
            # check connection state
            if not (is_connected := self.serial0.read_struct('<?')[0]):
                module_name = f'{CHANNEL_TYPES_INPUT[b"U"]}{idx + 1}'
                modules.append(Module(_bpod=self, index=idx, name=module_name))
                continue

            # read further information if module is connected
            n_events = N_SERIAL_EVENTS_DEFAULT
            firmware_version, n_chars = self.serial0.read_struct('<IB')
            base_name, more_info = self.serial0.read_struct(f'<{n_chars}s?')
            base_name = base_name.decode('UTF8')
            custom_event_names = list()
            while more_info:
                match self.serial0.read(1):
                    case b'#':
                        n_events = self.serial0.read_struct('<B')[0]
                    case b'E':
                        n_event_names = self.serial0.read_struct('<B')[0]
                        for _ in range(n_event_names):
                            n_chars = self.serial0.read_struct('<B')[0]
                            event_name = self.serial0.read_struct(f'<{n_chars}s')[0]
                            custom_event_names.append(event_name.decode('UTF8'))
                more_info = self.serial0.read_struct('<?')[0]

            # create module name with trailing index
            matches = [re.match(rf'^{base_name}(\d$)', m.name) for m in modules]
            index = max([int(m.group(1)) for m in matches if m is not None] + [0])
            module_name = f'{base_name}{index + 1}'
            logger.debug(f'Detected {module_name} on module port {idx + 1}')

            # create instance of Module
            modules.append(
                Module(
                    _bpod=self,
                    index=idx,
                    name=module_name,
                    is_connected=is_connected,
                    firmware_version=firmware_version,
                    n_events=n_events,
                    _custom_event_names=custom_event_names,
                )
            )

        # create NamedTuple and store as class attribute
        self.modules = NamedTuple('modules', [(m.name, Module) for m in modules])._make(
            modules
        )

        # update event names
        self._compile_event_names()

    def validate_state_machine(self, state_machine: StateMachine) -> None:
        """
        Validate the provided state machine for compatibility with the hardware.

        Parameters
        ----------
        state_machine : StateMachine
            The state machine to validate.

        Raises
        ------
        ValueError
            If the state machine is invalid or not compatible with the hardware.
        """
        self.send_state_machine(state_machine, validate_only=True)

    def send_state_machine(
        self,
        state_machine: StateMachine,
        run_asap: bool = True,
        validate_only: bool = False,
    ):
        """
        Send a state machine to the Bpod.

        This method compiles the provided state machine into a byte array format
        compatible with the Bpod and sends it to the device. It also validates the
        state machine for compatibility with the hardware before sending.

        Parameters
        ----------
        state_machine : StateMachine
            The state machine to be sent to the Bpod device.
        run_asap : bool, optional
            If True, the state machine will run immediately after the current one has
            finished. Default is True.
        validate_only : bool, optional
            If True, the state machine is only validated and not sent to the device.
            Default is False.

        Raises
        ------
        ValueError
            If the state machine is invalid or exceeds hardware limitations.
        """
        # Disable all active module relays
        if not validate_only:
            self._disable_all_module_relays()

        # Ensure that the state machine has at least one state
        if (n_states := len(state_machine.states)) == 0:
            raise ValueError('State machine has no states')

        # Check if '>back' operator is being used
        targets_used = {
            target
            for state in state_machine.states.values()
            for target in state.state_change_conditions.values()
        }
        use_back_op = '>back' in targets_used

        # Validate the maximum number of states (excluding '>exit' and '>back')
        max_states = self._hardware.max_states - 1 - use_back_op
        if n_states > max_states:
            raise ValueError(f'State machine has more than {max_states} states')

        # Validate states
        valid_targets = list(state_machine.states.keys()) + VALID_OPERATORS
        for state_name, state in state_machine.states.items():
            for condition_name, target in state.state_change_conditions.items():
                if target not in valid_targets:
                    target_type = 'operator' if target[0] == '>' else 'target state'
                    raise ValueError(
                        f"Invalid {target_type} '{target}' for state change condition "
                        f"'{condition_name}' in state '{state_name}'"
                    )
                if condition_name not in self.event_names:
                    raise ValueError(
                        f"Invalid state change condition '{condition_name}' in state "
                        f"'{state_name}'"
                    )
            actions = set(state.output_actions.keys())
            if invalid_actions := actions.difference(self.output_actions):
                raise ValueError(
                    f"Invalid output action '{invalid_actions.pop()}' "
                    f"in state '{state_name}'"
                )

        # Number of global timers, global counters and conditions used by state machine.
        # These correspond to the respective highest id + 1, i.e., all elements with
        # smaller ids will be included irrespective of their use.
        n_global_timers = max(state_machine.global_timers.keys(), default=-1) + 1
        n_global_counters = max(state_machine.global_counters.keys(), default=-1) + 1
        n_conditions = max(state_machine.conditions.keys(), default=-1) + 1
        for name, value, key in (
            ('global timers', n_global_timers, 'n_global_timers'),
            ('global counters', n_global_counters, 'n_global_counters'),
            ('conditions', n_conditions, 'n_conditions'),
        ):
            if value > (maximum_value := getattr(self._hardware, key)):
                raise ValueError(
                    f'Too many {name} in state machine. Hardware supports a maximum '
                    f'of {maximum_value} {name}.'
                )

        # Compile list of physical channels
        # TODO: this is ugly
        physical_output_channels = [m.name for m in self.modules] + [
            o.name for o in self.outputs if o.io_type != b'U'
        ]
        physical_input_channels = [m.name for m in self.modules] + [
            o.name for o in self.inputs if o.io_type != b'U'
        ]

        # Validate global timers
        for timer_id, timer in state_machine.global_timers.items():
            if timer.channel not in physical_output_channels + [None]:
                raise ValueError(
                    f"Invalid channel '{timer.channel}' for global timer {timer_id}"
                )

        # TODO: validate global timer onset triggers
        # TODO: validate global counters
        # TODO: validate conditions
        # TODO: Check that sync channel is not used as state output

        # return here if we're only validating the state machine
        if validate_only:
            return

        # compile dicts of indices to resolve strings to integers
        target_indices = {
            k: v for v, k in enumerate([*state_machine.states.keys(), 'exit'])
        }
        target_indices.update({'exit': n_states, '>exit': n_states})
        target_indices.update({'>back': 255} if use_back_op else {})
        event_indices = {k: v for v, k in enumerate(self.event_names)}
        action_indices = {k: v for v, k in enumerate(self.output_actions)}

        # Initialize bytearray. This will be appended to in the following sections.
        byte_array = bytearray(
            (n_states, n_global_timers, n_global_counters, n_conditions)
        )

        # Compile target indices for state timers and append to bytearray
        # Target indices default to the respective state's index unless 'Tup' is used
        for state_idx, state in enumerate(state_machine.states.values()):
            for event, target in state.state_change_conditions.items():
                if event == 'Tup':
                    byte_array.append(target_indices[target])
                    break
            else:
                byte_array.append(state_idx)

        # Helper function for appending events and their target indices to bytearray
        def append_events(event0: str, event1: str) -> None:
            idx0 = event_indices[event0]
            idx1 = event_indices[event1]
            for state in state_machine.states.values():
                counter_idx = len(byte_array)
                byte_array.append(0)
                for event, target in state.state_change_conditions.items():
                    if idx0 <= (key_idx := event_indices[event]) < idx1:
                        byte_array[counter_idx] += 1
                        byte_array.extend((key_idx - idx0, target_indices[target]))

        # Append input events to bytearray (i.e., events on physical input channels)
        append_events(self.event_names[0], 'GlobalTimer1_Start')

        # Append output actions and their values to bytearray
        i1 = action_indices['GlobalTimerTrig']
        for state in state_machine.states.values():
            counter_pos = len(byte_array)
            byte_array.append(0)
            for action, value in state.output_actions.items():
                if (key_idx := action_indices[action]) < i1:
                    byte_array[counter_pos] += 1
                    byte_array.extend((key_idx, value))

        # Append remaining events
        append_events('GlobalTimer1_Start', 'GlobalTimer1_End')  # global timer start
        append_events('GlobalTimer1_End', 'GlobalCounter1_End')  # global timer end
        append_events('GlobalCounter1_End', 'Condition1')  # global counter end
        append_events('Condition1', 'Tup')  # conditions

        # Compile indices for global timers channels
        timer_channel_indices = {k: v for v, k in enumerate(physical_output_channels)}
        timer_channel_indices[None] = 254

        # Append values for global timers to bytearray
        idx0 = len(byte_array)
        byte_array.extend((254,) * n_global_timers + (0,) * n_global_timers * 4)
        for timer_id, global_timer in state_machine.global_timers.items():
            offset = idx0 + timer_id
            byte_array[offset : offset + 5 * n_global_timers : n_global_timers] = (
                timer_channel_indices[global_timer.channel],
                255 if global_timer.value_on == 0 else global_timer.value_on,  # TODO
                255 if global_timer.value_off == 0 else global_timer.value_off,  # TODO
                global_timer.loop,
                global_timer.send_events,
            )

        # Append global counter events to bytearray
        idx0 = len(byte_array)
        byte_array.extend((254,) * n_global_counters)
        for counter_id, global_counter in state_machine.global_counters.items():
            byte_array[idx0 + counter_id] = event_indices[global_counter.event]

        # Compile indices for condition channels
        global_timers = [
            f'GlobalTimer{i + 1}' for i in range(self._hardware.n_global_timers)
        ]
        condition_channel_indices = {
            k: v for v, k in enumerate(physical_input_channels + global_timers)
        }

        # Append values for conditions to bytearray
        idx0 = len(byte_array)
        byte_array.extend((0,) * n_global_counters * 2)
        for condition_id, condition in state_machine.conditions.items():
            offset = idx0 + condition_id
            byte_array[offset : offset + 2 * n_conditions : n_conditions] = (
                condition_channel_indices[condition.channel],
                condition.value,
            )

        # Append global counter resets
        # TODO: one uint8 per state, default = 0?
        byte_array.extend(
            s.output_actions.get('GlobalCounterReset', 0)
            for s in state_machine.states.values()
        )

        # Helper function for packing a collection of integers into byte_array
        def pack_values(values: list[int], format_str: str) -> None:
            byte_array.extend(struct.pack(f'<{len(values)}{format_str}', *values))

        # The format of the next values depends on the number of global timers
        if self._hardware.n_global_timers > 16:
            format_string = 'I'  # uint32
        elif self._hardware.n_global_timers > 8:
            format_string = 'H'  # uint16
        else:
            format_string = 'B'  # uint8

        # Pack global timer triggers and cancels into bytearray
        for key in ('GlobalTimerTrig', 'GlobalTimerCancel'):
            pack_values(
                [s.output_actions.get(key, 0) for s in state_machine.states.values()],
                format_string,
            )

        # Pack global timer onset triggers into bytearray
        pack_values(
            [
                getattr(state_machine.global_timers.get(idx, {}), 'onset_trigger', 0)
                for idx in range(n_global_timers)
            ],
            format_string,
        )

        # Pack state timers
        pack_values(
            [
                round(s.timer * self._hardware.cycle_frequency)
                for s in state_machine.states.values()
            ],
            'I',  # uint32
        )

        # Pack global timer durations, onset delays and loop intervals
        for key in ('duration', 'onset_delay', 'loop_interval'):
            pack_values(
                [
                    round(
                        getattr(state_machine.global_timers.get(idx, {}), key, 0)
                        * self._hardware.cycle_frequency
                    )
                    for idx in range(n_global_timers)
                ],
                'I',  # uint32
            )

        # Pack global counter thresholds
        pack_values(
            [
                getattr(state_machine.global_counters.get(idx, {}), 'threshold', 0)
                for idx in range(n_global_counters)
            ],
            'I',  # uint32
        )

        # Send to state machine
        logger.debug('Sending state machine definition to Bpod')
        n_bytes = len(byte_array)
        self.serial0.write_struct(
            f'<c2?H{n_bytes}s', b'C', run_asap, use_back_op, n_bytes, byte_array
        )


class Channel(ABC):
    """Abstract base class representing a channel on the Bpod device."""

    @abstractmethod
    def __init__(self, bpod: Bpod, name: str, io_key: bytes, index: int):
        """
        Abstract base class representing a channel on the Bpod device.

        Parameters
        ----------
        bpod : Bpod
            The Bpod instance associated with the channel.
        name : str
            The name of the channel.
        io_key : bytes
            The I/O type of the channel (e.g., b'B', b'V', b'P').
        index : int
            The index of the channel.
        """
        self.name = name
        self.io_type = io_key
        self.index = index
        self._serial0 = bpod.serial0

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Input(Channel):
    """Input channel class representing a digital input channel."""

    def __init__(self, bpod: Bpod, name: str, io_key: bytes, index: int):
        """
        Input channel class representing a digital input channel.

        Parameters
        ----------
        bpod : Bpod
            The Bpod instance associated with the channel.
        name : str
            The name of the channel.
        io_key : bytes
            The I/O type of the channel (e.g., b'B', b'V', b'P').
        index : int
            The index of the channel.
        """
        super().__init__(bpod, name, io_key, index)
        self._set_enable_inputs = bpod._set_enable_inputs
        self._enabled = io_key in (b'PBWF')  # Enable Port, BNC, Wire and FlexIO inputs

    def read(self) -> bool:
        """
        Read the state of the input channel.

        Returns
        -------
        bool
            True if the input channel is active, False otherwise.
        """
        return self._serial0.verify([b'I', self.index])

    def override(self, state: bool) -> None:
        """
        Override the state of the input channel.

        Parameters
        ----------
        state : bool
            The state to set for the input channel.
        """
        self._serial0.write_struct('<cB', b'V', state)

    def enable(self, enabled: bool) -> bool:
        """
        Enable or disable the input channel.

        Parameters
        ----------
        enabled : bool
            True to enable the input channel, False to disable.

        Returns
        -------
        bool
            True if the operation was success, False otherwise.
        """
        if self.io_type not in b'FDBWVP':
            logger.warning(
                f'{"En" if enabled else "Dis"}abling input `{self.name}` has no effect'
            )
        self._enabled = enabled
        success = self._set_enable_inputs()
        return success

    @property
    def enabled(self) -> bool:
        """
        Check if the input channel is enabled.

        Returns
        -------
        bool
            True if the input channel is enabled, False otherwise.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        """
        Enable or disable the input channel.

        Parameters
        ----------
        enabled : bool
            True to enable the input channel, False to disable.
        """
        self.enable(enabled)


class Output(Channel):
    """Output channel class representing a digital output channel."""

    def __init__(self, bpod: Bpod, name: str, io_key: bytes, index: int):
        """
        Output channel class representing a digital output channel.

        Parameters
        ----------
        bpod : Bpod
            The Bpod instance associated with the channel.
        name : str
            The name of the channel.
        io_key : bytes
            The I/O type of the channel (e.g., b'B', b'V', b'P').
        index : int
            The index of the channel.
        """
        super().__init__(bpod, name, io_key, index)

    def override(self, state: bool | int) -> None:
        """
        Override the state of the output channel.

        Parameters
        ----------
        state : bool or int
            The state to set for the output channel. For binary I/O types, provide a
            bool. For pulse width modulation (PWM) I/O types, provide an int (0-255).
        """
        if isinstance(state, int) and self.io_type in (b'D', b'B', b'W'):
            state = state > 0
        self._serial0.write_struct('<c2B', b'O', self.index, state)


@dataclass
class Module:
    """Represents a Bpod module with its configuration and event names."""

    _bpod: Bpod
    """A reference to the Bpod."""

    index: int
    """The index of the module."""

    name: str
    """The name of the module."""

    is_connected: bool = False
    """Whether the module is connected."""

    firmware_version: int | None = None
    """The firmware version of the module."""

    n_events: int = N_SERIAL_EVENTS_DEFAULT
    """The number of events assigned to the module."""

    _custom_event_names: list[str] = field(default_factory=list)
    """A list of custom event names."""

    def __post_init__(self) -> None:
        self._relay_enabled = False
        self._define_event_names()

    def _define_event_names(self):
        """Define the module's event names."""
        self.event_names = []
        for idx in range(self.n_events):
            if len(self._custom_event_names) > idx:
                self.event_names.append(f'{self.name}_{self._custom_event_names[idx]}')
            else:
                self.event_names.append(f'{self.name}_{idx + 1}')

    @validate_call
    def set_relay(self, enable: bool) -> None:
        """
        Enable or disable the serial relay for the module.

        Parameters
        ----------
        enable : bool
            True to enable the relay, False to disable it.
        """
        if enable == self._relay_enabled:
            return
        if enable is True:
            self._bpod._disable_all_module_relays()
        logger.info(f'{"En" if enable else "Dis"}abling relay for module {self.name}')
        self._bpod.serial0.write_struct('<cB?', b'J', self.index, enable)
        self._relay_enabled = enable

    @property
    def relay(self) -> bool:
        """The current state of the serial relay."""
        return self._relay_enabled

    @relay.setter
    def relay(self, state: bool) -> None:
        """The current state of the serial relay."""
        self.set_relay(state)
