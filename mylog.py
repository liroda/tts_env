import functools
import logging
import logging.handlers

__all__ = [
    'logger',
]


class TTSLogger(object):
    def __init__(self, name="tts"):

        self.logger = logging.getLogger(name)


        self.format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')



        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)
        self.filehandler = logging.handlers.TimedRotatingFileHandler(filename='/app/deploy/logs',when="MIDNIGHT", interval=1, backupCount=5)
        self.filehandler.suffix = "%Y-%m-%d.log"

        self.filehandler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logger.addHandler(self.filehandler)

        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def __call__(self, log_level: str, msg: str):
        self.logger.log(log_level, msg)


ttslogger = TTSLogger()
