"""Module for interfacing with the Bpod Finite State Machine."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

from pydantic import validate_call
from serial import SerialException
from serial.tools.list_ports import comports

from bpod_core import __version__ as bpod_core_version
from bpod_core.com import ExtendedSerial

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer  # noqa: F401

PROJECT_NAME = 'bpod-core'
VIDS_BPOD = [0x16C0]  # vendor IDs of supported Bpod devices
MIN_BPOD_FW_VERSION = (23, 0)  # minimum supported firmware version (major, minor)
MIN_BPOD_HW_VERSION = 3  # minimum supported hardware version
MAX_BPOD_HW_VERSION = 4  # maximum supported hardware version

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
    timer_period: int
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


class BpodError(Exception):
    """
    Exception class for Bpod-related errors.

    This exception is raised when an error specific to the Bpod device or its
    operations occurs.
    """


class Bpod:
    """Bpod class for interfacing with the Bpod Finite State Machine."""

    _version: VersionInfo
    _hardware_config: HardwareConfiguration
    serial0: ExtendedSerial
    """Primary serial device for communication with the Bpod."""
    serial1: ExtendedSerial
    """Secondary serial device for communication with the Bpod."""
    serial2: ExtendedSerial | None = None
    """Tertiary serial device for communication with the Bpod - used by Bpod 2+ only."""
    inputs: NamedTuple
    outputs: NamedTuple

    @validate_call
    def __init__(self, port: str | None = None, serial_number: str | None = None):
        logger.info(f'bpod_core {bpod_core_version}')

        # identify Bpod by port or serial number
        port, serial_number = self._identify_bpod(port, serial_number)

        # open primary serial port
        self.serial0 = ExtendedSerial()
        self.serial0.port = port
        self.open()

        # get firmware version and machine type; enforce version requirements
        self._get_version_info()

        # get the Bpod's onboard hardware configuration
        self._get_hardware_configuration()

        # configure input and output channels
        self._configure_channels()

        # detect additional serial ports
        self._detect_additional_serial_ports()

        # log hardware information
        machine = {3: 'r2.0-2.5', 4: '2+ r1.0'}.get(self.version.machine, 'unknown')
        logger.info(f'Connected to Bpod Finite State Machine {machine} on {self.port}')
        logger.info(
            f'Firmware Version {"{}.{}".format(*self.version.firmware)}, '
            f'Serial Number {serial_number}, PCB Revision {self.version.pcb}'
        )

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, type, value, traceback):
        """Exit context and close connection."""
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def _identify_bpod(
        port: str | None = None, serial_number: str | None = None
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

        def sends_discovery_byte(port: str) -> bool:
            """Check if the device on the given port sends a discovery byte."""
            try:
                with ExtendedSerial(port, timeout=0.15) as ser:
                    return ser.read(1) == bytes([222])
            except SerialException:
                return False

        # If no port or serial number provided, try to automagically find an idle Bpod
        if port is None and serial_number is None:
            try:
                port_info = next(
                    p
                    for p in comports()
                    if getattr(p, 'vid', None) in VIDS_BPOD
                    and sends_discovery_byte(p.device)
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
                    and sends_discovery_byte(p.device)
                )
            except (StopIteration, AttributeError) as e:
                raise BpodError(f'No device with serial number {serial_number}') from e

        # Else, assure that the provided port exists and the device could be a Bpod
        else:
            try:
                port_info = next(p for p in comports() if p.device == port)
            except (StopIteration, AttributeError) as e:
                raise BpodError(f'Port not found: {port}') from e

        if port_info.vid not in VIDS_BPOD:
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
        if self.version.firmware > (22, 0):
            hardware_conf = list(self.serial0.query_struct(b'H', '<2H6B'))
        else:
            hardware_conf = list(self.serial0.query_struct(b'H', '<2H5B'))
            hardware_conf.insert(-4, 3)  # max bytes per serial msg always = 3
        hardware_conf.extend(self.serial0.read_struct(f'<{hardware_conf[-1]}s1B'))
        hardware_conf.extend(self.serial0.read_struct(f'<{hardware_conf[-1]}s'))
        self._hardware_config = HardwareConfiguration(*hardware_conf)

    def _configure_channels(self) -> None:
        type_dict = {
            b'U': 'UART',
            b'X': 'USB',
            b'Z': 'USB_ext',
            b'F': 'FlexIO',
            b'S': 'SPI',
            b'D': 'Digital',
            b'B': 'BNC',
            b'W': 'Wire',
            b'V': 'Valve',
            b'P': 'Port',
        }

        logger.debug('Configuring I/O ports')
        for description, channel_class in (
            (self._hardware_config.input_description, Input),
            (self._hardware_config.output_description, Output),
        ):
            channels = []
            types = []
            class_name = f'{channel_class.__name__.lower()}s'

            for idx in range(len(description)):
                io_key = description[idx : idx + 1]
                if bytes(io_key) in type_dict:
                    n = description[:idx].count(io_key) + 1
                    name = f'{type_dict[io_key]}{n}'
                    channels.append(channel_class(self, name, io_key, idx))
                    types.append((name, channel_class))
                else:
                    raise RuntimeError(f'Unknown {class_name[:-1]} type: {io_key}')

            setattr(self, class_name, NamedTuple(class_name, types)._make(channels))

        # set the enabled state of the inputs
        self._set_enable_inputs()

    def _detect_additional_serial_ports(self) -> None:
        """Detect additional USB-serial ports."""
        # First, assemble a list of candidate ports
        candidate_ports = [
            p.device for p in comports() if p.vid in VIDS_BPOD and p.device != self.port
        ]

        # Exclude those devices from the list that are already sending a discovery byte
        for port in candidate_ports:
            try:
                with ExtendedSerial(port, timeout=0.15) as ser:
                    if ser.read(1) == bytes([222]):
                        candidate_ports.remove(port)
            except SerialException:
                pass

        # Find second USB-serial port
        for port in candidate_ports:
            try:
                with ExtendedSerial(port, timeout=0.15) as ser:
                    self.serial0.write(b'{')
                    if ser.read(1) == bytes([222]):
                        ser.reset_input_buffer()
                        ser.timeout = None
                        self.serial1 = ser
                        candidate_ports.remove(port)
                        break
            except SerialException:
                pass

        # State Machine 2+ uses a third USB-serial port
        if self.version.machine == 4:
            for port in candidate_ports:
                try:
                    with ExtendedSerial(port, timeout=0.15) as ser:
                        self.serial0.write(b'}')
                        if ser.read(1) == bytes([223]):
                            ser.reset_input_buffer()
                            ser.timeout = None
                            self.serial2 = ser
                            break
                except SerialException:
                    pass

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

    def _set_enable_inputs(self) -> bool:
        enable = [i.enabled for i in self.inputs]
        self.serial0.write_struct(f'<c{self._hardware_config.n_inputs}?', b'E', *enable)
        return self.serial0.read(1) == b'\x01'

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

    def update_modules(self):
        self.serial0.write(b'M')
        print(f'{self.n_modules} modules')
        for module_index in range(self._hardware_config.input_description.count(b'U')):
            connected = self.serial0.read(1) == b'\x01'
            if not connected:
                continue
            firmware_version, n = self.serial0.read_struct('<IB')
            module_name, more_info = self.serial0.read_struct(f'<{n}s?')
            while more_info:
                match self.serial0.read(1):
                    case b'#':
                        n_serial_events = self.serial0.read_struct('<B')[0]
                    case b'E':
                        n_event_names = self.serial0.read(0)
                        event_names = []
                        for i_event_name in range(n_event_names):
                            n = self.serial0.read_struct('<B')[0]
                            event_names.append(self.serial0.read_struct(f'<{n}s'))
                more_info = self.serial0.read(1) == b'\x01'


class Channel(ABC):
    """Abstract base class representing a channel on the Bpod device."""

    @abstractmethod
    def __init__(self, bpod: Bpod, name: str, io_type: bytes, index: int):
        """
        Abstract base class representing a channel on the Bpod device.

        Parameters
        ----------
        bpod : Bpod
            The Bpod instance associated with the channel.
        name : str
            The name of the channel.
        io_type : bytes
            The I/O type of the channel (e.g., b'B', b'V', b'P').
        index : int
            The index of the channel.
        """
        self.name = name
        self.io_type = io_type
        self.index = index
        self._serial0 = bpod.serial0

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Input(Channel):
    """Input channel class representing a digital input channel."""

    def __init__(self, bpod: Bpod, name: str, io_type: bytes, index: int):
        """
        Input channel class representing a digital input channel.

        Parameters
        ----------
        bpod : Bpod
            The Bpod instance associated with the channel.
        name : str
            The name of the channel.
        io_type : bytes
            The I/O type of the channel (e.g., b'B', b'V', b'P').
        index : int
            The index of the channel.
        """
        super().__init__(bpod, name, io_type, index)
        self._set_enable_inputs = bpod._set_enable_inputs
        self._enabled = True

    def read(self) -> bool:
        """
        Read the state of the input channel.

        Returns
        -------
        bool
            True if the input channel is active, False otherwise.
        """
        return self._serial0.verify([b'I', self.index], b'\x01')

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

    def __init__(self, bpod: Bpod, name: str, io_type: bytes, index: int):
        """
        Output channel class representing a digital output channel.

        Parameters
        ----------
        bpod : Bpod
            The Bpod instance associated with the channel.
        name : str
            The name of the channel.
        io_type : bytes
            The I/O type of the channel (e.g., b'B', b'V', b'P').
        index : int
            The index of the channel.
        """
        super().__init__(bpod, name, io_type, index)

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
