import signal

class InterruptHandler(object):

    def __init__(self, sig=signal.SIGUSR1):
        self.sig = sig
        self.signal_received = False
        self._original_handler = signal.getsignal(self.sig)

    def __enter__(self):

        def handler(signum, frame):
            self.signal_received = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        signal.signal(self.sig, self._original_handler)

    def reset(self):

        self.signal_received = False

        return True

if __name__ == '__main__':
    with InterruptHandler() as h:
        while True:
            if h.signal_received:
                print(1)
                h.reset()
