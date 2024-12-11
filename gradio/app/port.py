"""
File: port.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Utility functions to check and free ports by terminating processes holding them.
License: MIT License
"""

import socket
import psutil
from typing import Iterable, Union


def is_port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def free_ports(ports: Union[int, Iterable[int]]) -> None:
    ports_to_free = {ports} if isinstance(ports, int) else set(ports)

    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            connections = proc.net_connections(kind="inet")
            for conn in connections:
                if conn.laddr.port in ports_to_free:
                    proc.terminate()
                    proc.wait()
                    ports_to_free.discard(conn.laddr.port)
                    if not ports_to_free:
                        return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
