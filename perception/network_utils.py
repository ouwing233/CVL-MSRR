import socket

from .config import bind_PORT


def create_udp_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind(('', bind_PORT))
    s.settimeout(3)
    print('Bind UDP on ', bind_PORT, '...')
    return s


def UDP_receive(s):
    try:
        data, addr = s.recvfrom(1024)
        data = data.decode("utf-8", "ignore")
        message = 'Received from (%s:%s),%s' % (addr[0], addr[1], data)
        print(message)
    except Exception as e:
        print(e)


def UDP_send(s, content, server_ip, port):
    s.sendto(content.encode("utf-8"), (server_ip, port))
